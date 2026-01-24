# 🎨 LahTeam Tuner

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Helper library cho việc train LoRA trên Google Colab** - Hỗ trợ FLUX.2, Qwen Image, và Z-Image Turbo.

## ✨ Tính năng

- 🚀 **Tự động tải model** - Chỉ 1 lệnh để tải VAE, Text Encoder, DiT
- ⚙️ **Auto config** - Tự động detect model type và tạo cấu hình phù hợp
- 📊 **Dataset config** - Tạo TOML config cho dataset training
- 🔧 **Training helpers** - Build command line arguments tự động
- 📝 **Logger patch** - Thay thế logger thành print() cho Colab

## 📦 Models được hỗ trợ

| Model | Type | Description |
|-------|------|-------------|
| `z_image_turbo` | Z-Image | Mô hình Z-Image Turbo |
| `qwen_image` | Qwen | Qwen Image generation |
| `qwen_image_edit` | Qwen | Qwen Image editing |
| `qwen_image_edit_2509` | Qwen | Qwen Image editing v2509 |
| `flux2_dev` | FLUX.2 | FLUX.2 Dev (Mistral 3) |
| `flux2_klein_4b` | FLUX.2 | FLUX.2 Klein 4B |
| `flux2_klein_base_4b` | FLUX.2 | FLUX.2 Klein Base 4B ⭐ |
| `flux2_klein_9b` | FLUX.2 | FLUX.2 Klein 9B |
| `flux2_klein_base_9b` | FLUX.2 | FLUX.2 Klein Base 9B ⭐ |

> ⭐ **Recommended**: Sử dụng `klein_base_*` cho training LoRA

## 🚀 Cài đặt

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

# Hoặc gọi trực tiếp
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
    is_flux2_model,          # Kiểm tra có phải FLUX.2 không
    is_edit_model,           # Kiểm tra có phải Edit model không
    get_flux2_version,       # Lấy version string (dev, klein-4b, ...)
    get_script_prefix,       # Lấy prefix cho script (flux_2, qwen_image, ...)
    get_network_module,      # Lấy network module path
    
    # Training
    build_train_args,        # Build command line arguments từ dict
    generate_sample_prompts, # Tạo file sample prompts
    
    # Logger patch
    patch_logger_files,      # Patch logger thành print() cho Colab
    
    # File utils
    ensure_dir,              # Tạo thư mục nếu chưa tồn tại
    find_images_in_folder,   # Tìm ảnh trong folder
    count_images_in_folder,  # Đếm số ảnh trong folder
    read_caption,            # Đọc file caption
    write_caption,           # Ghi file caption
    
    # Dataset
    download_and_unzip,      # Tải và giải nén dataset từ URL
    get_repeats_from_folder_name,  # Lấy repeats từ tên folder (10_name)
)
```

## 📁 Cấu trúc thư viện

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

MIT License - Xem file [LICENSE](LICENSE) để biết thêm chi tiết.

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
