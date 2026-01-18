from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

from turbo_alignment.settings.datasets.base import (
    BaseDatasetSettings,
    DatasetType,
    MultiDatasetSettings,
)
from turbo_alignment.settings.datasets.chat import ChatDatasetSettings


class HFDatasetSettings(BaseModel):
    """Settings for HuggingFace datasets integration."""

    cache_dir: Optional[str] = Field(
        default=None,
        description="Directory for caching tokenized datasets. Uses HF default if None."
    )

    num_proc: Optional[int] = Field(
        default=None,
        description="Number of processes for parallel tokenization. None = auto."
    )

    tokenization_batch_size: int = Field(
        default=1000,
        description="Batch size for .map() tokenization calls"
    )

    keep_in_memory: bool = Field(
        default=False,
        description="Force keeping dataset in memory (for small datasets)"
    )

    streaming_buffer_size: int = Field(
        default=10000,
        description="Shuffle buffer size when using streaming mode"
    )


class PairPreferenceDatasetSettings(BaseDatasetSettings):
    dataset_type: Literal[DatasetType.PAIR_PREFERENCES] = DatasetType.PAIR_PREFERENCES
    chat_settings: ChatDatasetSettings
    add_labels: bool = True
    hf_settings: HFDatasetSettings = Field(
        default_factory=HFDatasetSettings,
        description="HuggingFace datasets configuration"
    )


class PairPreferenceMultiDatasetSettings(PairPreferenceDatasetSettings, MultiDatasetSettings):
    ...
