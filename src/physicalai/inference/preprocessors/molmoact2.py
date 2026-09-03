# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""NumPy observation preprocessing for MolmoAct2 inference."""

from __future__ import annotations

import re
from typing import Any

import cv2
import numpy as np
from typing_extensions import override

from physicalai.inference.constants import IMAGES, STATE, TASK
from physicalai.inference.preprocessors.base import Preprocessor
from physicalai.inference.preprocessors.stats_normalizer import StatsNormalizer

SO101_JOINT_SIGNS = (1.0, -1.0, 1.0, 1.0, 1.0, 1.0)
SO101_JOINT_OFFSETS = (0.0, 90.0, 90.0, 0.0, 0.0, 0.0)

_TRAILING_PUNCTUATION = ".,!?;:"
_IMAGE_NDIM = 4
_RGB_CHANNELS = 3
_PREFIX_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"^(?:task|instruction|language[_ ]instruction|goal)\s*[:\-]\s*",
        r"^(?:the\s+task\s+is\s+to|your\s+task\s+is\s+to)\s+",
    )
)


def _normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    for pattern in _PREFIX_PATTERNS:
        text = pattern.sub("", text, count=1).strip()
    return text.rstrip(_TRAILING_PUNCTUATION).strip().lower()


def normalization_stats(stats: dict[str, Any]) -> dict[str, np.ndarray]:
    """Convert manifest statistics to Studio buffer dtypes.

    Returns:
        Float32 statistics and a boolean normalization mask.
    """
    return {
        name: np.asarray(value, dtype=np.bool_ if name == "mask" else np.float32)
        for name, value in stats.items()
        if value is not None
    }


def _wrap(value: str, start: str, end: str, *, enabled: bool) -> str:
    if not value or not enabled or (value.startswith(start) and value.endswith(end)):
        return value
    return f"{start}{value}{end}"


def _discrete_state(state: np.ndarray, num_tokens: int) -> str:
    if num_tokens <= 0:
        msg = f"num_state_tokens must be > 0, got {num_tokens}."
        raise ValueError(msg)
    state = np.nan_to_num(np.asarray(state, dtype=np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
    token_ids = np.rint((np.clip(state, -1.0, 1.0) + 1.0) / 2.0 * (num_tokens - 1)).astype(np.int64)
    payload = "".join(f"<state_{int(token)}>" for token in token_ids.reshape(-1))
    return f"<state_start>{payload}<state_end>"


def _build_prompt(
    task: str,
    state: np.ndarray,
    *,
    num_state_tokens: int,
    setup_type: str,
    control_mode: str,
    add_setup_tokens: bool,
    add_control_tokens: bool,
    num_images: int,
) -> str:
    setup = _wrap(setup_type, "<setup_start>", "<setup_end>", enabled=add_setup_tokens)
    control = _wrap(control_mode, "<control_start>", "<control_end>", enabled=add_control_tokens)
    prompt = (
        f"The task is to {task}. The setup is {setup}. "
        f"The current state of the robot is {_discrete_state(state, num_state_tokens)}. "
        f"The expected control mode is {control}. "
        "Given these, what action should the robot take to complete the task?"
    )
    if num_images == 1:
        image_prefix = "<|image|>"
    else:
        image_prefix = "".join(f"Image {index + 1}<|image|>" for index in range(num_images))
    return f"{image_prefix}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<action_output>"


class JointFrameTransform:
    """Map leading joints between robot and checkpoint frames."""

    def __init__(
        self,
        signs: list[float] | None = None,
        offsets: list[float] | None = None,
    ) -> None:
        """Use Studio SO101 defaults unless compatible overrides are supplied.

        Raises:
            ValueError: If signs and offsets have different lengths.
        """
        signs = list(SO101_JOINT_SIGNS) if signs is None else signs
        offsets = list(SO101_JOINT_OFFSETS) if offsets is None else offsets
        if len(signs) != len(offsets):
            msg = f"joint_signs ({len(signs)}) and joint_offsets ({len(offsets)}) must match"
            raise ValueError(msg)
        self.signs = np.asarray(signs, dtype=np.float32)
        self.offsets = np.asarray(offsets, dtype=np.float32)

    def apply(self, values: np.ndarray, *, inverse: bool) -> np.ndarray:
        """Apply the forward or inverse affine transform.

        Returns:
            A transformed copy of the input values.
        """
        count = min(self.signs.size, values.shape[-1])
        output = np.array(values, copy=True)
        joints = values[..., :count]
        output[..., :count] = (
            self.signs[:count] * (joints - self.offsets[:count])
            if inverse
            else self.signs[:count] * joints + self.offsets[:count]
        )
        return output


class MolmoAct2Preprocessor(Preprocessor):
    """Prepare normalized prompts and packed images before tokenization."""

    def __init__(
        self,
        *,
        image_keys: list[str],
        state_stats: dict[str, Any] | None = None,
        normalization_mode: str = "QUANTILES",
        image_size: tuple[int, int] = (378, 378),
        num_state_tokens: int = 256,
        setup_type: str = "",
        control_mode: str = "",
        add_setup_tokens: bool = True,
        add_control_tokens: bool = True,
        adapt_to_so101: bool = False,
        joint_signs: list[float] | None = None,
        joint_offsets: list[float] | None = None,
    ) -> None:
        """Store observation preprocessing settings.

        Raises:
            ValueError: If no state tokens are available.
        """
        if num_state_tokens <= 0:
            msg = f"num_state_tokens must be > 0, got {num_state_tokens}."
            raise ValueError(msg)
        self.image_keys = list(image_keys)
        self.image_size = tuple(image_size)
        self.num_state_tokens = num_state_tokens
        self.setup_type = setup_type
        self.control_mode = control_mode
        self.add_setup_tokens = add_setup_tokens
        self.add_control_tokens = add_control_tokens
        self.joint_transform = JointFrameTransform(joint_signs, joint_offsets) if adapt_to_so101 else None
        self.normalizer = (
            StatsNormalizer(
                stats={STATE: normalization_stats(state_stats)},
                mode=normalization_mode.lower(),
                features=[STATE],
            )
            if state_stats
            else None
        )

    @override
    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Prepare one observation batch for tokenizer inference.

        Returns:
            Inputs with packed images and encoded prompt text.

        Raises:
            ValueError: If a required state, task, or image input is invalid.
        """
        outputs = dict(inputs)
        state = outputs.get(STATE, outputs.get(f"observation.{STATE}"))
        if state is None:
            msg = f"MolmoAct2 requires {STATE!r} in its input"
            raise ValueError(msg)
        state = np.asarray(state, dtype=np.float32)
        if state.ndim == 1:
            state = state[None, :]
        if self.joint_transform is not None:
            state = self.joint_transform.apply(state, inverse=False)
        if self.normalizer is not None:
            state = self.normalizer({STATE: state})[STATE]
        state = np.clip(state, -1.0, 1.0)

        images = self._images(outputs, batch_size=state.shape[0])
        tasks = self._tasks(outputs, batch_size=state.shape[0])
        outputs[IMAGES] = np.stack([self._resize(image) for image in images])
        outputs[TASK] = [
            _build_prompt(
                tasks[index],
                state[index],
                num_state_tokens=self.num_state_tokens,
                setup_type=self.setup_type,
                control_mode=self.control_mode,
                add_setup_tokens=self.add_setup_tokens,
                add_control_tokens=self.add_control_tokens,
                num_images=len(images),
            )
            for index in range(state.shape[0])
        ]
        outputs.pop(STATE, None)
        outputs.pop(f"observation.{STATE}", None)
        return outputs

    def _images(self, inputs: dict[str, Any], *, batch_size: int) -> list[np.ndarray]:
        images = self._raw_images(inputs)
        output = []
        for image in images:
            if image.ndim != _IMAGE_NDIM:
                msg = f"Expected BCHW or BHWC image with 3 channels, got {image.shape}"
                raise ValueError(msg)
            if image.shape[1] == _RGB_CHANNELS:
                canonical = image
            elif image.shape[-1] == _RGB_CHANNELS:
                canonical = image.transpose(0, 3, 1, 2)
            else:
                msg = f"Expected BCHW or BHWC image with 3 channels, got {image.shape}"
                raise ValueError(msg)
            if canonical.shape[0] != batch_size:
                msg = f"Image batch size mismatch: expected {batch_size}, got {canonical.shape[0]}"
                raise ValueError(msg)
            output.append(canonical)
        return output

    def _raw_images(self, inputs: dict[str, Any]) -> list[np.ndarray]:
        container = inputs.get(IMAGES)
        images: list[np.ndarray] = []
        for name in self.image_keys:
            key = name if name.startswith(f"{IMAGES}.") else f"{IMAGES}.{name}"
            if key in inputs:
                images.append(np.asarray(inputs[key]))
            elif isinstance(container, dict) and name.removeprefix(f"{IMAGES}.") in container:
                images.append(np.asarray(container[name.removeprefix(f"{IMAGES}.")]))
        if not self.image_keys and isinstance(container, np.ndarray):
            images = [container]
        elif not self.image_keys and isinstance(container, dict):
            keys = sorted(key for key in container if "is_pad" not in str(key))
            images = [np.asarray(container[key]) for key in keys]
        elif not self.image_keys and not images:
            keys = sorted(key for key in inputs if key.startswith(f"{IMAGES}.") and "is_pad" not in key)
            images = [np.asarray(inputs[key]) for key in keys]
        if not images:
            msg = "MolmoAct2 requires at least one image input"
            raise ValueError(msg)
        return images

    @staticmethod
    def _tasks(inputs: dict[str, Any], *, batch_size: int) -> list[str]:
        source = inputs.get(TASK, inputs.get(f"observation.{TASK}", inputs.get("observation.language")))
        if source is None:
            msg = f"MolmoAct2 requires {TASK!r} in its input"
            raise ValueError(msg)
        tasks = [source] * batch_size if isinstance(source, str) else np.asarray(source).reshape(-1).tolist()
        if len(tasks) == 1 and batch_size > 1:
            tasks *= batch_size
        if len(tasks) != batch_size:
            msg = f"Expected {batch_size} task strings, got {len(tasks)}"
            raise ValueError(msg)
        return [_normalize_text(str(task)) for task in tasks]

    def _resize(self, images: np.ndarray) -> np.ndarray:
        height, width = self.image_size
        output = []
        for image in images:
            if image.dtype == np.uint8:
                pixels = image
            elif np.issubdtype(image.dtype, np.floating):
                pixels = image.astype(np.float32)
                if float(pixels.max()) <= 1.0:
                    pixels *= 255.0
                pixels = np.clip(pixels, 0.0, 255.0).astype(np.uint8)
            else:
                msg = f"Unsupported image dtype: {image.dtype}"
                raise ValueError(msg)
            resized = cv2.resize(pixels.transpose(1, 2, 0), (width, height), interpolation=cv2.INTER_LINEAR_EXACT)
            output.append(resized.transpose(2, 0, 1).astype(np.float32) / 255.0)
        return np.stack(output)


__all__ = [
    "SO101_JOINT_OFFSETS",
    "SO101_JOINT_SIGNS",
    "JointFrameTransform",
    "MolmoAct2Preprocessor",
    "normalization_stats",
]
