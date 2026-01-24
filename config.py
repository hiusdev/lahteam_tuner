"""
Model configurations cho các loại model được hỗ trợ.
"""

# ==============================================================================
# QWEN / Z-IMAGE CONFIG
# ==============================================================================

MODEL_CONFIG = {
    "qwen_image": {
        "vae": {
            "type": "folder",
            "repo_id": "Qwen/Qwen-Image", 
            "folder": "vae",
            "local_folder": "Qwen-Image/vae"
        },
        "text_encoder": {
            "type": "file",
            "repo_id": "Comfy-Org/Qwen-Image_ComfyUI", 
            "filename": "qwen_2.5_vl_7b.safetensors", 
            "subfolder": "split_files/text_encoders"
        },
        "dit": {
            "type": "file",
            "repo_id": "Comfy-Org/Qwen-Image_ComfyUI", 
            "filename": "qwen_image_fp8_e4m3fn.safetensors", 
            "subfolder": "split_files/diffusion_models"
        },
        "network_module": "networks.lora_qwen_image",
        "script_prefix": "qwen_image",
        "model_version": "original",
        # Training params từ docs chính thức
        "training_params": {
            "timestep_sampling": "shift",
            "weighting_scheme": "none",
            "discrete_flow_shift": 2.2,
            "fp8_base": True,
            "fp8_vl": True,  # Recommended for < 16GB VRAM
        }
    },
    "qwen_image_edit": {
        "vae": {
            "type": "folder",
            "repo_id": "Qwen/Qwen-Image", 
            "folder": "vae",
            "local_folder": "Qwen-Image/vae"
        },
        "text_encoder": {
            "type": "file",
            "repo_id": "Comfy-Org/Qwen-Image_ComfyUI", 
            "filename": "qwen_2.5_vl_7b.safetensors", 
            "subfolder": "split_files/text_encoders"
        },
        "dit": {
            "type": "file",
            "repo_id": "Comfy-Org/Qwen-Image-Edit_ComfyUI", 
            "filename": "qwen_image_edit_fp8_e4m3fn.safetensors", 
            "subfolder": "split_files/diffusion_models"
        },
        "network_module": "networks.lora_qwen_image",
        "script_prefix": "qwen_image",
        "model_version": "edit",
        "is_edit": True,
        "training_params": {
            "timestep_sampling": "shift",
            "weighting_scheme": "none",
            "discrete_flow_shift": 2.2,
            "fp8_base": False,  # Edit mode không dùng fp8_base
            "control_resolution": [1024, 1024],
        }
    },
    "qwen_image_edit_2509": {
        "vae": {
            "type": "folder",
            "repo_id": "Qwen/Qwen-Image", 
            "folder": "vae",
            "local_folder": "Qwen-Image/vae"
        },
        "text_encoder": {
            "type": "file",
            "repo_id": "Comfy-Org/Qwen-Image_ComfyUI", 
            "filename": "qwen_2.5_vl_7b.safetensors", 
            "subfolder": "split_files/text_encoders"
        },
        "dit": {
            "type": "file",
            "repo_id": "Comfy-Org/Qwen-Image-Edit_ComfyUI", 
            "filename": "qwen_image_edit_2509_bf16.safetensors", 
            "subfolder": "split_files/diffusion_models"
        },
        "network_module": "networks.lora_qwen_image",
        "script_prefix": "qwen_image",
        "model_version": "edit-2509",
        "is_edit": True,
        "training_params": {
            "timestep_sampling": "shift",
            "weighting_scheme": "none",
            "discrete_flow_shift": 2.2,
            "fp8_base": False,
            "control_resolution": [1024, 1024],
        }
    },
    "z_image_turbo": {
        "vae": {
            "type": "file",
            "repo_id": "Comfy-Org/z_image_turbo", 
            "filename": "ae.safetensors", 
            "subfolder": "split_files/vae"
        },
        "text_encoder": {
            "type": "file",
            "repo_id": "Comfy-Org/z_image_turbo", 
            "filename": "qwen_3_4b.safetensors", 
            "subfolder": "split_files/text_encoders"
        },
        "dit": {
            "type": "file",
            "repo_id": "ostris/Z-Image-De-Turbo", 
            "filename": "z_image_de_turbo_v1_bf16.safetensors", 
            "subfolder": ""
        },
        "network_module": "networks.lora_zimage",
        "script_prefix": "zimage",
        "model_version": None,
        "training_params": {
            "timestep_sampling": "shift",
            "weighting_scheme": "none",
            "discrete_flow_shift": 2.0,
            "fp8_base": True,
        }
    }
}


# ==============================================================================
# FLUX.2 CONFIG
# ==============================================================================

FLUX2_CONFIG = {
    "flux2_dev": {
        "vae": {
            "type": "file",
            "repo_id": "black-forest-labs/FLUX.2-dev",
            "filename": "ae.safetensors"
        },
        "text_encoder": {
            "type": "folder",
            "repo_id": "black-forest-labs/FLUX.2-dev",
            "folder": "text_encoder",
            "local_folder": "FLUX2-dev/text_encoder"
        },
        "dit": {
            "type": "file",
            "repo_id": "black-forest-labs/FLUX.2-dev",
            "filename": "flux2-dev.safetensors"
        },
        "model_version": "dev",
        "max_blocks_to_swap": 29,
    },
    "flux2_klein_4b": {
        "vae": {
            "type": "file",
            "repo_id": "Comfy-Org/vae-text-encorder-for-flux-klein-4b",
            "subfolder": "split_files/vae",
            "filename": "flux2-vae.safetensors"
        },
        "text_encoder": {
            "type": "file",
            "repo_id": "Comfy-Org/vae-text-encorder-for-flux-klein-4b",
            "subfolder": "split_files/text_encoders",
            "filename": "qwen_3_4b.safetensors"
        },
        "dit": {
            "type": "file",
            "repo_id": "Comfy-Org/vae-text-encorder-for-flux-klein-4b",
            "filename": "flux-2-klein-4b.safetensors",
            "subfolder": "split_files/diffusion_models"
        },
        "model_version": "klein-4b",
        "max_blocks_to_swap": 13,
    },
    "flux2_klein_base_4b": {
        "vae": {
            "type": "file",
            "repo_id": "Comfy-Org/vae-text-encorder-for-flux-klein-4b",
            "subfolder": "split_files/vae",
            "filename": "flux2-vae.safetensors"
        },
        "text_encoder": {
            "type": "file",
            "repo_id": "Comfy-Org/vae-text-encorder-for-flux-klein-4b",
            "subfolder": "split_files/text_encoders",
            "filename": "qwen_3_4b.safetensors"
        },
        "dit": {
            "type": "file",
            "repo_id": "Comfy-Org/vae-text-encorder-for-flux-klein-4b",
            "filename": "flux-2-klein-base-4b.safetensors",
            "subfolder": "split_files/diffusion_models"
        },
        "model_version": "klein-base-4b",
        "max_blocks_to_swap": 13,
    },
    "flux2_klein_9b": {
        "vae": {
            "type": "file",
            "repo_id": "Comfy-Org/vae-text-encorder-for-flux-klein-9b",
            "subfolder": "split_files/vae",
            "filename": "flux2-vae.safetensors"
        },
        "text_encoder": {
            "type": "file",
            "repo_id": "Comfy-Org/vae-text-encorder-for-flux-klein-9b",
            "subfolder": "split_files/text_encoders",
            "filename": "qwen_3_4b.safetensors"
        },
        "dit": {
            "type": "file",
            "repo_id": "Comfy-Org/vae-text-encorder-for-flux-klein-9b",
            "filename": "flux-2-klein-9b.safetensors",
            "subfolder": "split_files/diffusion_models"
        },
        "model_version": "klein-9b",
        "max_blocks_to_swap": 16,
    },
    "flux2_klein_base_9b": {
        "vae": {
            "type": "file",
            "repo_id": "Comfy-Org/vae-text-encorder-for-flux-klein-9b",
            "subfolder": "split_files/vae",
            "filename": "flux2-vae.safetensors"
        },
        "text_encoder": {
            "type": "file",
            "repo_id": "Comfy-Org/vae-text-encorder-for-flux-klein-9b",
            "subfolder": "split_files/text_encoders",
            "filename": "qwen_3_4b.safetensors"
        },
        "dit": {
            "type": "file",
            "repo_id": "Comfy-Org/vae-text-encorder-for-flux-klein-9b",
            "filename": "flux-2-klein-base-9b.safetensors",
            "subfolder": "split_files/diffusion_models"
        },
        "model_version": "klein-base-9b",
        "max_blocks_to_swap": 16,
    }
}

# Thêm thông tin chung cho FLUX.2
for key in FLUX2_CONFIG:
    FLUX2_CONFIG[key]["network_module"] = "networks.lora_flux_2"
    FLUX2_CONFIG[key]["script_prefix"] = "flux_2"
    FLUX2_CONFIG[key]["is_flux2"] = True
    # Training params từ docs chính thức
    FLUX2_CONFIG[key]["training_params"] = {
        "timestep_sampling": "flux2_shift",
        "weighting_scheme": "none",
        "fp8_base": True,
        "fp8_scaled": True,  # Recommended with fp8_base for DiT
        "mixed_precision": "bf16",
        # fp8_text_encoder: không dùng cho dev (Mistral 3)
        "fp8_text_encoder": key != "flux2_dev",
        # Control resolution theo docs: 2024x2024 for single, 1024x1024 for multi
        "control_resolution": [2024, 2024],
        "no_resize_control": True,  # FLUX.2 official inference settings
    }


def get_model_config(model_type: str) -> dict:
    """
    Lấy config cho model_type.
    
    Args:
        model_type: Loại model (z_image_turbo, qwen_image, flux2_*, ...)
    
    Returns:
        dict: Config của model
    """
    if model_type.startswith("flux2_"):
        config = FLUX2_CONFIG.get(model_type)
    else:
        config = MODEL_CONFIG.get(model_type)
    
    if not config:
        raise ValueError(f"Không tìm thấy config cho model: {model_type}")
    
    return config
