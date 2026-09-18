# Configure Preprocessing and Postprocessing

Preprocessors run before model execution. Postprocessors run after model execution. Together, they define the input and output adaptation around the runner.

The manifest can declare both stages.

```yaml
model:
  runner:
    type: action_chunking
    chunk_size: 50
  artifacts:
    openvino: model.xml
  preprocessors:
    - type: normalize
      artifact: stats.safetensors
  postprocessors:
    - type: denormalize
      artifact: stats.safetensors
```

The same components can also be declared with explicit class paths.

```yaml
preprocessors:
  - class_path: physicalai.inference.preprocessors.StatsNormalizer
    init_args:
      artifact: stats.safetensors
```

Pipeline shape:

```text
observation
  -> preprocessors
  -> runner
  -> postprocessors
  -> action output
```

Use `type` for registered built-in components. Use `class_path` when you want an explicit import path.

## Specify the input image layout

`resize` (`ResizePreprocessor`) and `smolvla_resize` (`ResizeSmolVLA`) require
`image_layout`. Set it to the axis order of the images entering that preprocessor:

- `BCHW`: batch, channels, height, width.
- `BHWC`: batch, height, width, channels.

All camera images passed to the same preprocessor must use the configured layout.
The layout is not inferred from image dimensions, including ambiguous shapes such
as `(1, 3, 3, 3)`.

For example, configure resizing for channels-last input:

```yaml
preprocessors:
  - type: resize
    image_resolution: [224, 224]
    image_layout: BHWC
```

For a SmolVLA pipeline with channels-first input, use:

```yaml
preprocessors:
  - type: smolvla_resize
    image_resolution: [512, 512]
    image_layout: BCHW
```

When using `class_path`, put `image_layout` inside `init_args`. When constructing
either class in Python, pass it as a keyword argument, for example:

```python
from physicalai.inference.preprocessors import ResizePreprocessor

resize = ResizePreprocessor(image_resolution=(224, 224), image_layout="BHWC")
```

Existing manifests, configuration files, and Python calls that omit `image_layout`
must be updated. Choose the layout from the image producer or the preceding
preprocessor; there is no default or automatic detection. Unbatched `CHW` and
`HWC` layouts are not supported.

The setting describes the input only. `ResizePreprocessor` still returns `BCHW`
images, and `ResizeSmolVLA` still stacks images as
`(cameras, batch, channels, height, width)` with camera masks.
