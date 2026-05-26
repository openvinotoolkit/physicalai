# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Feature descriptors used by the inference package.

An :class:`InferenceFeature` captures the static metadata for a single
input or output tensor of an exported policy: its semantic category
(:class:`InferenceFeatureType`), its tensor ``shape`` and the ``name``
under which it is exchanged with the runtime.
"""

from dataclasses import dataclass
from enum import StrEnum


class InferenceFeatureType(StrEnum):
    """Semantic category of an :class:`InferenceFeature`."""

    VISUAL = "VISUAL"
    ACTION = "ACTION"
    STATE = "STATE"
    LANGUAGE = "LANG"
    COMMON = "COMMON"


@dataclass(frozen=True)
class InferenceFeature:
    """Static description of a single inference input or output tensor.

    Attributes:
        ftype: Semantic category of the feature.
        shape: Tensor shape, excluding the batch dimension.
        name: Identifier used to reference the feature at runtime.
    """

    ftype: InferenceFeatureType
    shape: tuple[int, ...]
    name: str
