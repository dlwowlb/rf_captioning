# RF Captioning Examples

This folder contains example scripts for using HY-Motion and RF-Genesis together.

## Project Structure

```
rf_captioning/
├── examples/                    # Example scripts (YOU ARE HERE)
│   ├── motion_to_doppler.py    # Full pipeline: Text → Motion → RF Doppler
│   ├── hymotion_only.py        # HY-Motion standalone
│   └── rfgenesis_only.py       # RF-Genesis standalone
├── HY-Motion-1.0/              # HY-Motion model
│   ├── hymotion/               # Main package
│   │   ├── utils/
│   │   │   └── t2m_runtime.py  # T2MRuntime class
│   │   ├── pipeline/
│   │   └── ...
│   └── ckpts/                  # Model weights
└── RF-Genesis/                 # RF-Genesis simulator
    ├── genesis/                # Main package
    │   ├── raytracing/
    │   │   ├── pathtracer.py   # trace() function
    │   │   └── signal_generator.py
    │   ├── object_diffusion/
    │   │   └── object_diff.py  # generate() function
    │   └── visualization/
    │       └── visualize.py    # save_video() function
    └── models/
        └── TI1843_config.json  # Radar config
```

## Examples

### 1. Full Pipeline: Text → Motion → RF Doppler

```bash
# From project root
python examples/motion_to_doppler.py \
    --prompt "a person walking forward" \
    --duration 3.0 \
    --output-dir output/test
```

This script:
1. Uses HY-Motion to generate SMPL motion from text
2. Converts motion to RF-Genesis format
3. Runs RF simulation with ray tracing
4. Generates Doppler visualizations

### 2. HY-Motion Only

```bash
python examples/hymotion_only.py \
    --prompt "a person waving hands" \
    --duration 2.0 \
    --output-dir output/motion_test
```

This script generates motion from text and saves:
- `motion_data.npy` - Raw SMPL parameters
- `motion_visualization.png` - Motion trajectory plot
- `metadata.json` - Generation metadata

### 3. RF-Genesis Only

```bash
python examples/rfgenesis_only.py \
    --prompt "a person running in place" \
    --output-dir output/rf_test
```

This script uses RF-Genesis's built-in MDM to generate motion and simulate RF signals.

## Prerequisites

### HY-Motion Setup

```bash
cd HY-Motion-1.0
pip install -r requirements.txt

# Download model weights
huggingface-cli download tencent/HY-Motion-1.0 \
    --include 'HY-Motion-1.0/*' \
    --local-dir ckpts/tencent
```

### RF-Genesis Setup

```bash
cd RF-Genesis
pip install -r requirements.txt
sh setup.sh
```

## Common Issues

### Import Errors

If you get import errors, ensure you run scripts from the project root:

```bash
cd /path/to/rf_captioning
python examples/motion_to_doppler.py --help
```

### Path Issues

The example scripts automatically resolve paths relative to their location.
If running from a different directory, use absolute paths:

```bash
python /path/to/examples/motion_to_doppler.py \
    --model-path /path/to/HY-Motion-1.0/ckpts/tencent/HY-Motion-1.0
```

### CUDA Out of Memory

For visualization steps, the scripts switch to CPU to avoid OOM:

```python
torch.set_default_device('cpu')
```

## API Reference

### HY-Motion Key Classes

```python
from hymotion.utils.t2m_runtime import T2MRuntime

runtime = T2MRuntime(
    config_path="path/to/config.yml",
    ckpt_name="path/to/latest.ckpt",
    device_ids=[0],  # GPU IDs
)

html, fbx_files, model_output = runtime.generate_motion(
    text="a person walking",
    seeds_csv="42",
    duration=3.0,
    cfg_scale=5.0,
    output_format="dict",
)
```

### RF-Genesis Key Functions

```python
from genesis.object_diffusion import object_diff
from genesis.raytracing import pathtracer, signal_generator
from genesis.visualization import visualize

# 1. Generate motion (uses MDM internally)
object_diff.generate(prompt, output_dir)

# 2. Ray tracing (run from genesis/ directory)
body_pir, body_aux = pathtracer.trace("path/to/obj_diff.npz")

# 3. Generate radar frames
radar_frames = signal_generator.generate_signal_frames(
    body_pir, body_aux, env_pir,
    radar_config="models/TI1843_config.json"
)

# 4. Visualize
visualize.save_video(config, radar_path, motion_path, output_path)
```

## Output Files

| File | Description |
|------|-------------|
| `obj_diff.npz` | SMPL pose parameters |
| `radar_frames.npy` | Raw radar signal frames |
| `output.mp4` | Visualization video |
| `range_doppler_samples.png` | Range-Doppler frame samples |
| `doppler_spectrogram.png` | Micro-Doppler spectrogram |
