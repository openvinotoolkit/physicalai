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
    LANGUAGE = "LANGUAGE"
    COMMON = "COMMON"


class InferenceFeatureDtype(StrEnum):
    """Data type of an :class:`InferenceFeature`."""

    FLOAT32 = "float32"
    INT64 = "int64"
    STRING = "string"

    @classmethod
    def from_torch(cls, dtype: str) -> "InferenceFeatureDtype":
        """Return the :class:`InferenceFeatureDtype` matching a torch dtype string.

        Raises:
            ValueError: If ``dtype`` does not map to a supported
                :class:`InferenceFeatureDtype`.
        """
        name = dtype.removeprefix("torch.")
        mapping = {
            "float32": cls.FLOAT32,
            "float": cls.FLOAT32,
            "int64": cls.INT64,
            "long": cls.INT64,
        }
        try:
            return mapping[name]
        except KeyError as exc:
            msg = f"Unsupported torch dtype: {dtype!r}"
            raise ValueError(msg) from exc

    @classmethod
    def from_numpy(cls, dtype: str) -> "InferenceFeatureDtype":
        """Return the :class:`InferenceFeatureDtype` matching a numpy dtype string.

        Raises:
            ValueError: If ``dtype`` does not map to a supported
                :class:`InferenceFeatureDtype`.
        """
        mapping = {
            "float32": cls.FLOAT32,
            "int64": cls.INT64,
            "str": cls.STRING,
            "str_": cls.STRING,
            "bytes": cls.STRING,
            "bytes_": cls.STRING,
            "object": cls.STRING,
        }
        if dtype in mapping:
            return mapping[dtype]
        if dtype.startswith(("<U", ">U", "U", "<S", ">S", "S")):
            return cls.STRING
        msg = f"Unsupported numpy dtype: {dtype!r}"
        raise ValueError(msg)


@dataclass(frozen=True)
class InferenceFeature:
    """Static description of a single inference input or output tensor.

    Attributes:
        ftype: Semantic category of the feature.
        shape: Tensor shape, excluding the batch dimension.
        name: Identifier used to reference the feature at runtime.
        dtype: Data type of the feature.
    """

    ftype: InferenceFeatureType
    shape: tuple[int, ...]
    name: str
    dtype: InferenceFeatureDtype
