from pathlib import Path
from typing import Any, Union, overload

import torch
from torch.utils.data import IterableDataset as TorchIterableDataset
from datasets import IterableDataset as HFIterableDataset
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase
from typing_extensions import Self

from turbo_alignment.common.data.io import count_lines, read_jsonl
from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.dataset.base import AlignmentDataset
from turbo_alignment.dataset.chat import (
    ChatDatasetRecord,
    ChatMessage,
)
from turbo_alignment.dataset.chat.chat import SplitChatDataset
from turbo_alignment.dataset.pair_preferences.models import PairPreferenceRecord
from turbo_alignment.dataset.registry import PairPreferenceDatasetTypeRegistry
from turbo_alignment.settings.datasets.base import (
    DatasetSourceSettings,
    DatasetStrategy,
)
from turbo_alignment.settings.datasets.pair_preference import (
    PairPreferenceDatasetSettings,
)

logger = get_project_logger()


@PairPreferenceDatasetTypeRegistry.register(DatasetStrategy.TRAIN)
class PairPreferenceDataset(AlignmentDataset[PairPreferenceRecord]):
    def __init__(
            self,
            source: DatasetSourceSettings,
            settings: PairPreferenceDatasetSettings,
            tokenizer: PreTrainedTokenizerBase,
            seed: int,
            read: bool = True,
    ):
        # Initialize base class (without reading yet)
        super().__init__(source=source, settings=settings, tokenizer=tokenizer, seed=seed)
        self.settings: PairPreferenceDatasetSettings = settings
        
        self._add_labels = settings.add_labels
        # Force keep_end=True for chat settings as required
        self.settings.chat_settings.keep_end = True
        
        self._hf_settings = settings.hf_settings

        # Reuse SplitChatDataset for tokenization logic
        self._chat_dataset = SplitChatDataset(
            source=source,
            settings=settings.chat_settings,
            tokenizer=tokenizer,
            seed=seed,
            read=False,
        )

        self._dataset = self._load_dataset()

        # Dynamically mixin TorchIterableDataset so Trainer detects it correctly
        class IterablePairPreferenceDataset(self.__class__, TorchIterableDataset):
            pass
        self.__class__ = IterablePairPreferenceDataset

    def _build_chat_record(self, record: PairPreferenceRecord, answer: ChatMessage) -> ChatDatasetRecord:
        context_messages = [ChatMessage(role=msg.role, content=msg.content, disable_loss=True)
                            for msg in record.context]
        return ChatDatasetRecord(id=record.id, messages=context_messages + [answer])

    def _process_batch(self, batch: dict[str, list]) -> dict[str, list]:
        """
        Process a batch of records using SplitChatDataset logic.
        
        Args:
            batch: Dict with lists of values for each field (HF format)
            
        Returns:
            Dict with lists of processed values
        """
        batch_size = len(batch["context"])
        records = []
        
        # 1. Convert HF batch to PairPreferenceRecord objects
        for i in range(batch_size):
            # context is a list of dicts (messages), need to ensure it parses correctly
            # batch['context'][i] is a list of dicts, which Pydantic should handle
            records.append(PairPreferenceRecord(
                id=batch.get("id", [""] * batch_size)[i],
                context=batch["context"][i],
                answer_w=batch["answer_w"][i],
                answer_l=batch["answer_l"][i],
                precomputed_margin=batch.get("precomputed_margin", [None] * batch_size)[i],
            ))

        # 2. Build chat records for chosen and rejected
        chosen_chat_records = [
            self._build_chat_record(r, ChatMessage(role=r.answer_w.role, content=r.answer_w.content))
            for r in records
        ]
        rejected_chat_records = [
            self._build_chat_record(r, ChatMessage(role=r.answer_l.role, content=r.answer_l.content))
            for r in records
        ]

        # 3. Use existing SplitChatDataset to tokenize
        tokenized_chosen = self._chat_dataset.convert_records(chosen_chat_records)
        tokenized_rejected = self._chat_dataset.convert_records(rejected_chat_records)

        # 4. Reassemble results
        results = {
            "id": [],
            "inputs_context": [],
            "inputs_chosen": [],
            "inputs_rejected": [],
            "precomputed_margin": [],
            "_valid": [],
        }

        for i in range(batch_size):
            chosen_tok = tokenized_chosen[i]
            rejected_tok = tokenized_rejected[i]
            
            if chosen_tok is not None and rejected_tok is not None:
                results["id"].append(records[i].id)
                # Convert tensors to lists for Arrow storage
                # Note: SplitChatDataset returns 1D tensors, so no need to squeeze
                results["inputs_context"].append(chosen_tok['context_ids'].tolist()) 
                results["inputs_chosen"].append(chosen_tok['answer_ids'].tolist())
                results["inputs_rejected"].append(rejected_tok['answer_ids'].tolist())
                results["precomputed_margin"].append(records[i].precomputed_margin)
                results["_valid"].append(True)
            else:
                # Invalid record (filtered out by truncation logic)
                results["id"].append("")
                results["inputs_context"].append([])
                results["inputs_chosen"].append([])
                results["inputs_rejected"].append([])
                results["precomputed_margin"].append(None)
                results["_valid"].append(False)
        
        return results

    def _load_dataset(self) -> HFIterableDataset:

        if self.source.records_data:
            raise NotImplementedError("Loading from records_data (in-memory list) not fully supported with HF datasets yet.")
        
        data_path = self.source.records_path or self.source.train_path
        if not data_path:
             raise ValueError('At least one of records_data and records_path should be not None')
        
        data_path = Path(data_path)
        logger.info(f"Loading dataset from {data_path} in producer-consumer mode (HF machinery)...")
        
        # 1. Load as streaming dataset (lazy)
        # HF handles sharding automatically for JSON/JSONL when using num_workers
        raw_dataset = load_dataset(
            "json",
            data_files=str(data_path),
            split="train",
            streaming=True,
        )

        # 2. Apply tokenization on-the-fly
        # This will run in worker processes
        
        # safely determine columns to remove
        remove_columns = None
        if hasattr(raw_dataset, 'features') and raw_dataset.features is not None:
            remove_columns = list(raw_dataset.features.keys())
            
        processed_dataset = raw_dataset.map(
            self._process_batch,
            batched=True,
            batch_size=self._hf_settings.tokenization_batch_size,
            remove_columns=remove_columns
        )

        # 3. Filter invalid
        processed_dataset = processed_dataset.filter(lambda x: x["_valid"])
        processed_dataset = processed_dataset.remove_columns(["_valid"])

        # 4. Shuffle buffer
        # Essential for streaming to avoid correlation
        return processed_dataset.shuffle(
            seed=self.seed,
            buffer_size=self._hf_settings.streaming_buffer_size,
        )

    def __len__(self) -> int:
        if hasattr(self, '_len'):
            return self._len

        length = 0
        if self.source.num_samples is not None:
            length = self.source.num_samples
        elif self.source.n_rows is not None:
            length = self.source.n_rows
        elif self.source.records_path:
            try:
                length = count_lines(self.source.records_path)
            except Exception:
                logger.warning(f"Could not count lines in {self.source.records_path}")
                length = 0

        self._len = length
        return length

    def _post_process_item(self, item: dict[str, Any]) -> dict[str, Any]:
        # Return raw lists (from HF dataset) to reduce overhead in worker processes.
        # Tensor conversion will happen in the collator.
        margin = item["precomputed_margin"]
        if margin is None:
            margin = 0.0

        return {
            "id": item["id"],
            "inputs_context": torch.tensor(item["inputs_context"], dtype=torch.long),
            "inputs_chosen": torch.tensor(item["inputs_chosen"], dtype=torch.long),
            "inputs_rejected": torch.tensor(item["inputs_rejected"], dtype=torch.long),
            "precomputed_margin": torch.tensor(margin, dtype=torch.float),
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        raise NotImplementedError("Indexing not supported in streaming mode.")

    def __iter__(self):
        for item in self._dataset:
            yield self._post_process_item(item)

    # Kept for compatibility with base class, though unused by new logic
    def convert_records(self, records: list[PairPreferenceRecord]) -> list[dict[str, Any] | None]:
        # Build chat records for chosen and rejected using helper method
        chosen_chat_records = [
            self._build_chat_record(record, ChatMessage(role=record.answer_w.role, content=record.answer_w.content))
            for record in records]
        rejected_chat_records = [
            self._build_chat_record(record, ChatMessage(role=record.answer_l.role, content=record.answer_l.content))
            for record in records]

        tokenized_chosen = self._chat_dataset.convert_records(chosen_chat_records)
        tokenized_rejected = self._chat_dataset.convert_records(rejected_chat_records)

        output: list[dict[str, Any] | None] = []
        for record, chosen_tok, rejected_tok in zip(records, tokenized_chosen, tokenized_rejected):
            if not (chosen_tok and rejected_tok):
                continue

            output.append({
                'id': record.id,
                'inputs_context': chosen_tok['context_ids'],
                'inputs_chosen': chosen_tok['answer_ids'],
                'inputs_rejected': rejected_tok['answer_ids'],
                'precomputed_margin': record.precomputed_margin,
            })

        return output

    @staticmethod
    @overload
    def _read_records(records: Path) -> list[PairPreferenceRecord]:
        ...

    @staticmethod
    @overload
    def _read_records(records: list[dict]) -> list[PairPreferenceRecord]:
        ...

    @staticmethod
    def _read_records(records) -> list[PairPreferenceRecord]:
        if isinstance(records, Path):
            return [PairPreferenceRecord(**record) for record in read_jsonl(records)]
        if isinstance(records, list):
            return [PairPreferenceRecord(**record) for record in records]
        raise NotImplementedError

    def get_slice(self, start: int, end: int) -> Self:
        raise NotImplementedError("Slicing not supported in streaming mode.")
