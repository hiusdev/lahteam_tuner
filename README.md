# 🎨 LahTeam Tuner

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Helper library for LoRA training on Google Colab** - Supports FLUX.2, Qwen Image, and Z-Image Turbo.

[🇻🇳 Tiếng Việt](README.vi.md)

## ✨ Features

- 🚀 **Auto download models** - Single command to download VAE, Text Encoder, DiT
- ⚙️ **Auto config** - Automatically detect model type and create appropriate configuration
- 📊 **Dataset config** - Generate TOML config for training dataset
- 🔧 **Training helpers** - Build command line arguments automatically
- 📝 **Logger patch** - Replace logger with print() for Colab compatibility

## 📦 Supported Models

| Model | Type | Description |
|-------|------|-------------|
| `z_image_turbo` | Z-Image | Z-Image Turbo model |
| `qwen_image` | Qwen | Qwen Image generation |
| `qwen_image_edit` | Qwen | Qwen Image editing |
| `qwen_image_edit_2509` | Qwen | Qwen Image editing v2509 |
| `flux2_dev` | FLUX.2 | FLUX.2 Dev (Mistral 3) |
| `flux2_klein_4b` | FLUX.2 | FLUX.2 Klein 4B |
| `flux2_klein_base_4b` | FLUX.2 | FLUX.2 Klein Base 4B ⭐ |
| `flux2_klein_9b` | FLUX.2 | FLUX.2 Klein 9B |
| `flux2_klein_base_9b` | FLUX.2 | FLUX.2 Klein Base 9B ⭐ |

> ⭐ **Recommended**: Use `klein_base_*` for LoRA training

## 🚀 Installation

```python
!git clone https://github.com/hiusdev/lahteam_tuner
!pip install -q -e lahteam_tuner
```

## 📚 API Reference

### Download Functions

```python
from lahteam_tuner import download_model, download_flux2_model, download_qwen_model

# Auto-detect model type
paths = download_model("flux2_klein_base_4b", "/content/models", hf_token="...")
# Returns: {"vae_path": ..., "text_encoders_path": ..., "dit_path": ...}

# Or call directly
paths = download_flux2_model("flux2_klein_base_4b", "/content/models")
paths = download_qwen_model("z_image_turbo", "/content/models")
```

### Dataset Config

```python
from lahteam_tuner import create_dataset_config

path = create_dataset_config(
    data_dir="/path/to/images",
    control_dir="/path/to/control",  # Optional
    config_dir="/path/to/config",
    resolution=(1024, 1024),
    batch_size=2,
    default_repeats=10,
    caption_extension=".txt",
    enable_bucket=True,
    model_type="flux2_klein_base_4b"
)
```

### Model Config

```python
from lahteam_tuner import get_model_config, MODEL_CONFIG, FLUX2_CONFIG

config = get_model_config("flux2_klein_base_4b")
# Returns:
# {
#     "vae": {...},
#     "text_encoder": {...},
#     "dit": {...},
#     "network_module": "networks.lora_flux_2",
#     "script_prefix": "flux_2",
#     "model_version": "klein-base-4b",
#     "training_params": {
#         "timestep_sampling": "shift",
#         "fp8_base": True,
#         ...
#     }
# }
```

### Utility Functions

```python
from lahteam_tuner import (
    # Model info
    is_flux2_model,          # Check if FLUX.2 model
    is_edit_model,           # Check if Edit model
    get_flux2_version,       # Get version string (dev, klein-4b, ...)
    get_script_prefix,       # Get script prefix (flux_2, qwen_image, ...)
    get_network_module,      # Get network module path
    
    # Training
    build_train_args,        # Build command line arguments from dict
    generate_sample_prompts, # Generate sample prompts file
    
    # Logger patch
    patch_logger_files,      # Patch logger to print() for Colab
    
    # File utils
    ensure_dir,              # Create directory if not exists
    find_images_in_folder,   # Find images in folder
    count_images_in_folder,  # Count images in folder
    read_caption,            # Read caption file
    write_caption,           # Write caption file
    
    # Dataset
    download_and_unzip,      # Download and extract dataset from URL
    get_repeats_from_folder_name,  # Get repeats from folder name (10_name)
)
```

## 📁 Library Structure

```
lahteam_tuner/
├── __init__.py      # Public API exports
├── config.py        # Model configurations (QWEN, FLUX.2)
├── download.py      # Model download functions
├── utils.py         # Helper utilities
├── setup.py         # Package setup
├── LICENSE          # MIT License
└── README.md
```

## 📝 License

MIT License - See [LICENSE](LICENSE) file for details.

## 👥 Authors

- **LahTeam.VN** - [Website](https://lahteam.vn)

## 🙏 Credits

- [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner) - Base training framework
- [Comfy-Org](https://huggingface.co/Comfy-Org) - FLUX.2 model hosting
- [Qwen](https://huggingface.co/Qwen) - Qwen Image models

---

<p align="center">
  <sub>🤖 This project was developed with AI assistance (Claude/Gemini)</sub>
</p>
