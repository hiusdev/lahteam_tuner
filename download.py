"""
LahTeam Musubi Tuner - Model Download Functions
Tải model đơn giản, chỉ cần gọi 1 function.

Usage:
    from lahteam_tuner.download import download_flux2_model, download_qwen_model
    
    # FLUX.2
    paths = download_flux2_model("flux2_klein_4b", "/content/models")
    
    # Qwen / Z-Image
    paths = download_qwen_model("z_image_turbo", "/content/models")
"""

import os
from typing import Optional, Dict

from .config import MODEL_CONFIG, FLUX2_CONFIG
from .utils import download_component


def download_flux2_model(
    model_type: str,
    model_dir: str,
    hf_token: Optional[str] = None
) -> Dict[str, str]:
    """
    Tải FLUX.2 model (VAE, Text Encoder, DiT).
    
    Args:
        model_type: "flux2_dev", "flux2_klein_4b", "flux2_klein_base_4b", "flux2_klein_9b", "flux2_klein_base_9b"
        model_dir: Thư mục lưu model (e.g. "/content/models")
        hf_token: HuggingFace token (optional)
    
    Returns:
        dict: {"vae_path": ..., "text_encoders_path": ..., "dit_path": ...}
    
    Example:
        paths = download_flux2_model("flux2_klein_base_4b", "/content/models")
        print(paths["dit_path"])
    """
    if not model_type.startswith("flux2_"):
        raise ValueError(f"model_type phải bắt đầu bằng 'flux2_'. Nhận được: {model_type}")
    
    config = FLUX2_CONFIG.get(model_type)
    if not config:
        available = list(FLUX2_CONFIG.keys())
        raise ValueError(f"Không tìm thấy config cho: {model_type}. Có sẵn: {available}")
    
    print(f"\n🚀 Đang tải FLUX.2: {model_type}")
    print("=" * 60)
    
    vae_path = download_component("VAE", model_dir, config['vae'], hf_token)
    text_encoders_path = download_component("Text Encoder", model_dir, config['text_encoder'], hf_token)
    dit_path = download_component("DiT", model_dir, config['dit'], hf_token)
    
    print("=" * 60)
    print("🎉 Tải FLUX.2 hoàn tất!")
    print(f"📁 VAE: {vae_path}")
    print(f"📁 Text Encoder: {text_encoders_path}")
    print(f"📁 DiT: {dit_path}")
    
    return {
        "vae_path": vae_path,
        "text_encoders_path": text_encoders_path,
        "dit_path": dit_path
    }


def download_qwen_model(
    model_type: str,
    model_dir: str,
    hf_token: Optional[str] = None
) -> Dict[str, str]:
    """
    Tải Qwen / Z-Image model (VAE, Text Encoder, DiT).
    
    Args:
        model_type: "qwen_image", "qwen_image_edit", "qwen_image_edit_2509", "z_image_turbo"
        model_dir: Thư mục lưu model (e.g. "/content/models")
        hf_token: HuggingFace token (optional)
    
    Returns:
        dict: {"vae_path": ..., "text_encoders_path": ..., "dit_path": ...}
    
    Example:
        paths = download_qwen_model("z_image_turbo", "/content/models")
        print(paths["dit_path"])
    """
    if model_type.startswith("flux2_"):
        raise ValueError(f"Dùng download_flux2_model() cho FLUX.2. Nhận được: {model_type}")
    
    config = MODEL_CONFIG.get(model_type)
    if not config:
        available = list(MODEL_CONFIG.keys())
        raise ValueError(f"Không tìm thấy config cho: {model_type}. Có sẵn: {available}")
    
    print(f"\n📦 Đang tải: {model_type}")
    print("=" * 60)
    
    vae_path = download_component("VAE", model_dir, config['vae'], hf_token)
    text_encoders_path = download_component("Text Encoder", model_dir, config['text_encoder'], hf_token)
    dit_path = download_component("DiT", model_dir, config['dit'], hf_token)
    
    print("=" * 60)
    print("🎉 Tải model hoàn tất!")
    print(f"📁 VAE: {vae_path}")
    print(f"📁 Text Encoder: {text_encoders_path}")
    print(f"📁 DiT: {dit_path}")
    
    return {
        "vae_path": vae_path,
        "text_encoders_path": text_encoders_path,
        "dit_path": dit_path
    }


def download_model(
    model_type: str,
    model_dir: str,
    hf_token: Optional[str] = None
) -> Dict[str, str]:
    """
    Tải model tự động (phát hiện FLUX.2 hoặc Qwen).
    
    Args:
        model_type: Bất kỳ model type nào
        model_dir: Thư mục lưu model
        hf_token: HuggingFace token
    
    Returns:
        dict: {"vae_path": ..., "text_encoders_path": ..., "dit_path": ...}
    """
    if model_type.startswith("flux2_"):
        return download_flux2_model(model_type, model_dir, hf_token)
    else:
        return download_qwen_model(model_type, model_dir, hf_token)
