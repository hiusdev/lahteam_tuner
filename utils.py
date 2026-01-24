"""
LahTeam Musubi Tuner - Helper Functions
Contains reusable helper functions.

Usage in Colab:
    !git clone https://github.com/hiusdev/lahteam_tuner
    !pip install -q -e lahteam_tuner
    from lahteam_tuner import download_component, patch_logger_files
"""

import os
import re
from typing import Optional, Dict, List, Any


# ==============================================================================
# DOWNLOAD HELPERS
# ==============================================================================

def download_component(
    component_name: str, 
    base_dir: str, 
    config: dict,
    hf_token: Optional[str] = None
) -> Optional[str]:
    """
    Download a model component (file or folder) from HuggingFace.
    
    Args:
        component_name: Component name (VAE, Text Encoder, DiT)
        base_dir: Base directory to save
        config: Dict containing {type, repo_id, filename/folder, subfolder, local_folder}
        hf_token: HuggingFace token (optional)
    
    Returns:
        str: Path to downloaded file/folder, or None if error
    
    Example:
        config = {
            "type": "file",
            "repo_id": "Comfy-Org/z_image_turbo",
            "filename": "ae.safetensors",
            "subfolder": "split_files/vae"
        }
        path = download_component("VAE", "/content/models", config)
    """
    from huggingface_hub import hf_hub_download, snapshot_download
    
    download_type = config.get("type", "file")
    repo_id = config["repo_id"]
    
    if download_type == "folder":
        folder = config.get("folder", "")
        local_folder = config.get("local_folder", folder)
        full_path = os.path.join(base_dir, local_folder)
        
        if os.path.exists(full_path) and os.listdir(full_path):
            print(f"✅ [{component_name.upper()}] Folder already exists: {local_folder}")
            return full_path
        
        print(f"⏳ [{component_name.upper()}] Downloading folder '{folder}' from {repo_id}...")
        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=f"{folder}/*" if folder else None,
                local_dir=base_dir,
                local_dir_use_symlinks=False,
                token=hf_token
            )
            print(f"   -> Folder download complete!")
            return full_path
        except Exception as e:
            print(f"❌ Error downloading [{component_name.upper()}]: {e}")
            return None
    else:
        subfolder = config.get('subfolder', '')
        filename = config['filename']
        full_path = os.path.join(base_dir, subfolder, filename) if subfolder else os.path.join(base_dir, filename)

        if os.path.exists(full_path):
            print(f"✅ [{component_name.upper()}] File already exists: {filename}")
            return full_path

        print(f"⏳ [{component_name.upper()}] Downloading '{filename}'...")
        os.makedirs(os.path.dirname(full_path) if os.path.dirname(full_path) else base_dir, exist_ok=True)
        
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=f"{subfolder}/{filename}" if subfolder else filename,
                local_dir=base_dir,
                token=hf_token
            )
            print(f"   -> Download complete!")
            return full_path
        except Exception as e:
            print(f"❌ Error downloading [{component_name.upper()}]: {e}")
            return None


def download_and_unzip(
    url: str, 
    unzip_dir: str, 
    zip_filename: str = "temp.zip",
    hf_token: Optional[str] = None
) -> bool:
    """
    Download zip file from URL and extract.
    
    Args:
        url: URL of zip file
        unzip_dir: Directory to extract to
        zip_filename: Temporary zip filename
        hf_token: Token for private repos
    
    Returns:
        bool: True if successful
    """
    import subprocess
    import zipfile
    
    if not url:
        return False
    
    if os.path.exists(unzip_dir) and any(os.scandir(unzip_dir)):
        print(f"✅ Directory '{unzip_dir}' already has data. Skipping.")
        return True
    
    print(f"⏳ Downloading from: {url}")
    zip_path = f"/content/{zip_filename}"
    header = f'"Authorization: Bearer {hf_token}"' if hf_token else ''
    
    cmd = f'aria2c --console-log-level=error -c -x 16 -k 1M -s 16 --header={header} -d /content -o {zip_filename} "{url}"'
    subprocess.run(cmd, shell=True)
    
    if os.path.exists(zip_path):
        print(f"📦 Extracting...")
        os.makedirs(unzip_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(unzip_dir)
        os.remove(zip_path)
        print(f"✅ Extraction complete!")
        return True
    
    print(f"❌ Download failed!")
    return False


# ==============================================================================
# LOGGER PATCH HELPERS
# ==============================================================================

def replace_logger_with_print(file_path: str) -> bool:
    """
    Replace logger.info/warning/error/debug with print().
    
    Args:
        file_path: Path to Python file
    
    Returns:
        bool: True if changes were made
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    
    # Pattern 1: logger.info(...) - simple, single line
    # Handle nested parentheses carefully
    patterns = [
        # logger.info("message")
        (r'logger\.(info|warning|error|debug)\(([^()]*)\)', r'print(\2)'),
        # logger.info(f"message {var}")
        (r'logger\.(info|warning|error|debug)\((f"[^"]*")\)', r'print(\2)'),
        (r"logger\.(info|warning|error|debug)\((f'[^']*')\)", r'print(\2)'),
        # logger.info("message", extra=...)
        (r'logger\.(info|warning|error|debug)\(("[^"]*"),\s*extra=[^)]+\)', r'print(\2)'),
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    # Fallback: simple replacement for complex cases
    # User may need to fix some cases manually
    content = re.sub(r'logger\.info\(', 'print(', content)
    content = re.sub(r'logger\.warning\(', 'print("[WARNING]",', content)
    content = re.sub(r'logger\.error\(', 'print("[ERROR]",', content)
    content = re.sub(r'logger\.debug\(', 'print("[DEBUG]",', content)
    
    if content != original:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False


def patch_logger_files(repo_dir: str, verbose: bool = False):
    """
    Patch all Python files in directory: replace logger with print.
    
    Args:
        repo_dir: Root directory to patch
        verbose: Print details of patched files
    """
    count = 0
    for root, _, files in os.walk(repo_dir):
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)

                try:
                    changed = replace_logger_with_print(path)
                    if verbose:
                        print(f"⏳ Patching: {path}")
                    if changed:
                        count += 1
                except Exception as e:
                    if verbose:
                        print(f"⚠️ Error: {path}: {e}")
    print(f"🔧 Patched {count} files")


# ==============================================================================
# DATASET CONFIG HELPERS
# ==============================================================================

def find_images_in_folder(path: str) -> bool:
    """Check if folder contains images."""
    import glob
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']:
        if glob.glob(os.path.join(path, ext)):
            return True
    return False


def get_repeats_from_folder_name(folder_name: str, default: int = 10) -> int:
    """
    Get repeats count from folder name (format: 10_name).
    
    Args:
        folder_name: Folder name
        default: Default value if not found
    
    Returns:
        int: Repeats count
    """
    parts = folder_name.split('_')
    if len(parts) > 1 and parts[0].isdigit():
        return int(parts[0])
    return default


# ==============================================================================
# TRAIN COMMAND HELPERS
# ==============================================================================

def build_train_args(config: Dict[str, Any]) -> str:
    """
    Build command line arguments from dict config.
    
    Args:
        config: Dict containing training parameters
    
    Returns:
        str: Arguments string
    
    Example:
        config = {"dit": "/path/dit", "learning_rate": 1e-4, "fp8_base": True}
        args = build_train_args(config)
        # -> '--dit="/path/dit" --learning_rate="0.0001" --fp8_base'
    """
    args = ""
    for k, v in config.items():
        if v is None or v is False:
            continue
        if isinstance(v, list):
            for item in v:
                args += f' --{k}="{item}"'
        elif isinstance(v, bool) and v is True:
            args += f" --{k}"
        else:
            args += f' --{k}="{v}"'
    return args


def get_script_prefix(model_type: str) -> str:
    """
    Get script prefix based on model_type.
    
    Args:
        model_type: Model type
    
    Returns:
        str: Prefix (flux_2, zimage, qwen_image)
    """
    if model_type.startswith("flux2_"):
        return "flux_2"
    elif model_type == "z_image_turbo":
        return "zimage"
    else:
        return "qwen_image"


def get_network_module(model_type: str) -> str:
    """
    Get network module based on model_type.
    
    Args:
        model_type: Model type
    
    Returns:
        str: Network module path
    """
    if model_type.startswith("flux2_"):
        return "networks.lora_flux_2"
    elif model_type == "z_image_turbo":
        return "networks.lora_zimage"
    else:
        return "networks.lora_qwen_image"


# ==============================================================================
# MODEL INFO HELPERS
# ==============================================================================

# FLUX.2 Version mapping
FLUX2_VERSION_MAP = {
    "flux2_dev": "dev",
    "flux2_klein_4b": "klein-4b",
    "flux2_klein_base_4b": "klein-base-4b",
    "flux2_klein_9b": "klein-9b",
    "flux2_klein_base_9b": "klein-base-9b"
}

# Max blocks_to_swap for FLUX.2
FLUX2_BLOCKS_LIMIT = {
    "flux2_dev": 29,
    "flux2_klein_4b": 13,
    "flux2_klein_base_4b": 13,
    "flux2_klein_9b": 16,
    "flux2_klein_base_9b": 16,
}


def get_flux2_version(model_type: str) -> str:
    """Get FLUX.2 version string from model_type."""
    return FLUX2_VERSION_MAP.get(model_type, "")


def get_max_blocks_to_swap(model_type: str) -> int:
    """Get blocks_to_swap limit for model_type."""
    return FLUX2_BLOCKS_LIMIT.get(model_type, 36)


def is_flux2_model(model_type: str) -> bool:
    """Check if model is FLUX.2."""
    return model_type.startswith("flux2_")


def is_edit_model(model_type: str) -> bool:
    """Check if model is Edit model."""
    return model_type in ["qwen_image_edit", "qwen_image_edit_2509"]


def get_train_script(model_type: str) -> str:
    """Get training script filename."""
    if model_type.startswith("flux2_"):
        return "flux_2_train_network.py"
    elif model_type == "z_image_turbo":
        return "zimage_train_network.py"
    else:
        return "qwen_image_train_network.py"


def get_cache_script_prefix(model_type: str) -> str:
    """Get prefix for cache scripts (latents & text encoder)."""
    return get_script_prefix(model_type)


# ==============================================================================
# DATASET CONFIG HELPERS (TOML)
# ==============================================================================

def create_dataset_config(
    data_dir: str,
    control_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    config_dir: str = "/content/fine_tune/config",
    resolution: tuple = (960, 544),
    control_resolution: Optional[tuple] = None,
    batch_size: int = 1,
    default_repeats: int = 1,
    caption_extension: str = ".txt",
    enable_bucket: bool = True,
    bucket_no_upscale: bool = False,
    no_resize_control: bool = False,
    model_type: str = "qwen_image"
) -> str:
    """
    Create dataset_config.toml in official musubi-tuner format.
    
    Args:
        data_dir: Directory containing images (can have subfolders 10_name)
        control_dir: Control images directory (for FLUX.2/Qwen Edit)
        cache_dir: Cache directory (default = data_dir)
        config_dir: Directory to save config file
        resolution: (width, height) default [960, 544]
        control_resolution: Resolution for control images
                           - Qwen Edit: recommended [1024, 1024]
                           - FLUX.2 single control: [2024, 2024]
                           - FLUX.2 multi control: [1024, 1024]
        batch_size: Batch size, default 1
        default_repeats: Default repeats count, default 1
        caption_extension: Caption file extension
        enable_bucket: Enable bucketing
        bucket_no_upscale: Don't upscale when bucketing
        no_resize_control: Don't resize control images (for FLUX.2)
        model_type: Model type for auto-detecting settings
    
    Returns:
        str: Path to created config file
    
    Example:
        # Qwen Edit
        path = create_dataset_config(
            data_dir="/content/data/output",
            control_dir="/content/data/input",
            control_resolution=(1024, 1024),
            model_type="qwen_image_edit"
        )
        
        # FLUX.2
        path = create_dataset_config(
            data_dir="/content/data/output",
            control_dir="/content/data/input",
            control_resolution=(2024, 2024),
            no_resize_control=True,
            model_type="flux2_klein_base_4b"
        )
    """
    import toml
    
    # Auto-detect settings based on model_type
    is_flux2 = model_type.startswith("flux2_")
    is_edit = model_type in ["qwen_image_edit", "qwen_image_edit_2509"]
    
    # Set default control_resolution based on model
    if control_dir and control_resolution is None:
        if is_edit:
            control_resolution = (1024, 1024)  # Recommended for Qwen Edit
        elif is_flux2:
            control_resolution = (2024, 2024)  # Single control for FLUX.2
    
    datasets = []
    
    # Scan subfolders
    try:
        subfolders = sorted([f.path for f in os.scandir(data_dir) if f.is_dir()])
    except FileNotFoundError:
        subfolders = []
    
    # Add root dir if has images
    paths = subfolders.copy()
    if find_images_in_folder(data_dir):
        paths.insert(0, data_dir)
    
    for idx, folder_path in enumerate(paths):
        if not find_images_in_folder(folder_path):
            continue
        
        folder_name = os.path.basename(folder_path)
        if folder_path == data_dir:
            repeats = default_repeats
        else:
            repeats = get_repeats_from_folder_name(folder_name, default_repeats)
        
        # Create dataset entry
        dataset_item = {
            "image_directory": folder_path,
            "num_repeats": repeats
        }
        
        # Cache directory (different for each dataset)
        if cache_dir:
            dataset_cache = os.path.join(cache_dir, f"cache_{idx}")
            dataset_item["cache_directory"] = dataset_cache
        
        # Control directory (for Edit mode or FLUX.2)
        if control_dir and (is_edit or is_flux2):
            dataset_item["control_directory"] = control_dir
            
            if control_resolution:
                dataset_item["control_resolution"] = list(control_resolution)
            
            if no_resize_control:
                dataset_item["no_resize_control"] = True
        
        print(f"   ✅ Dataset: '{folder_name}' | Repeats: {repeats}")
        datasets.append(dataset_item)
    
    if not datasets:
        raise ValueError("No image data found!")
    
    # Build full config
    full_config = {
        "general": {
            "resolution": list(resolution),
            "caption_extension": caption_extension,
            "batch_size": batch_size,
            "enable_bucket": enable_bucket,
            "bucket_no_upscale": bucket_no_upscale,
        },
        "datasets": datasets
    }
    
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "dataset_config.toml")
    
    # Write TOML with correct format
    with open(config_path, "w", encoding='utf-8') as f:
        # Write general section first
        f.write("[general]\n")
        for key, value in full_config["general"].items():
            if isinstance(value, bool):
                f.write(f"{key} = {str(value).lower()}\n")
            elif isinstance(value, list):
                f.write(f"{key} = {value}\n")
            elif isinstance(value, str):
                f.write(f'{key} = "{value}"\n')
            else:
                f.write(f"{key} = {value}\n")
        
        f.write("\n")
        
        # Write each dataset
        for ds in datasets:
            f.write("[[datasets]]\n")
            for key, value in ds.items():
                if isinstance(value, bool):
                    f.write(f"{key} = {str(value).lower()}\n")
                elif isinstance(value, list):
                    f.write(f"{key} = {value}\n")
                elif isinstance(value, str):
                    f.write(f'{key} = "{value}"\n')
                else:
                    f.write(f"{key} = {value}\n")
            f.write("\n")
    
    print("-" * 60)
    print(f"✅ Created: {config_path}")
    return config_path


# ==============================================================================
# SAMPLE GENERATION HELPERS
# ==============================================================================

def generate_sample_prompts(
    dataset_config_path: str,
    output_path: str,
    samples_per_dataset: int = 2,
    sample_width: int = 1024,
    sample_height: int = 1024,
    sample_steps: int = 28,
    control_dir: Optional[str] = None
) -> str:
    """
    Generate sample prompts file from dataset config.
    
    Args:
        dataset_config_path: Path to dataset_config.toml
        output_path: Path to save samples.txt
        samples_per_dataset: Number of samples per dataset
        sample_width/height: Sample dimensions
        sample_steps: Number of sampling steps
        control_dir: Control images directory (if needed)
    
    Returns:
        str: Path to samples.txt file
    """
    import toml
    import glob
    import random
    
    with open(dataset_config_path, "r", encoding='utf-8') as f:
        config_data = toml.load(f)
    
    datasets = config_data.get('datasets', [])
    lines = []
    
    for ds in datasets:
        img_dir = ds.get('image_directory')
        ctrl_dir = ds.get('control_directory', control_dir)
        
        if not img_dir or not os.path.exists(img_dir):
            continue
        
        images = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp'):
            images.extend(glob.glob(os.path.join(img_dir, ext)))
        
        if images:
            selected = random.sample(images, min(samples_per_dataset, len(images)))
            for img in selected:
                cap_path = os.path.splitext(img)[0] + ".txt"
                caption = ""
                if os.path.exists(cap_path):
                    with open(cap_path, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                
                if caption:
                    sample_line = f"{caption} --w {sample_width} --h {sample_height} --d {sample_steps}"
                    
                    # Add control image if available
                    if ctrl_dir:
                        basename = os.path.splitext(os.path.basename(img))[0]
                        ctrl_candidates = glob.glob(os.path.join(ctrl_dir, f"{basename}.*"))
                        if ctrl_candidates:
                            sample_line += f" --ci {ctrl_candidates[0]}"
                    
                    lines.append(sample_line)
    
    if lines:
        with open(output_path, "w", encoding='utf-8') as f:
            f.write("\n".join(lines))
        print(f"✅ Created {len(lines)} sample prompts: {output_path}")
    
    return output_path


# ==============================================================================
# COLAB STORE HELPERS
# ==============================================================================

def store_variables(var_dict: Dict[str, Any]):
    """
    Store variables to IPython %store (runs in Colab).
    
    Args:
        var_dict: Dict {variable_name: value}
    """
    try:
        from IPython import get_ipython
        from IPython.utils import capture
        
        ipython = get_ipython()
        if ipython is None:
            return
        
        g = ipython.user_ns
        for name, value in var_dict.items():
            g[name] = value
        
        with capture.capture_output():
            for name in var_dict.keys():
                ipython.run_line_magic('store', name)
        
        print(f"✅ Stored {len(var_dict)} variables")
    except Exception as e:
        print(f"⚠️ Cannot store variables: {e}")


def restore_variables(*var_names: str) -> Dict[str, Any]:
    """
    Restore variables from IPython %store.
    
    Args:
        var_names: Variable names to restore
    
    Returns:
        dict: {variable_name: value}
    """
    try:
        from IPython import get_ipython
        from IPython.utils import capture
        
        ipython = get_ipython()
        if ipython is None:
            return {}
        
        with capture.capture_output():
            for name in var_names:
                ipython.run_line_magic('store', f'-r {name}')
        
        result = {}
        for name in var_names:
            if name in ipython.user_ns:
                result[name] = ipython.user_ns[name]
        
        return result
    except Exception as e:
        print(f"⚠️ Cannot restore variables: {e}")
        return {}


# ==============================================================================
# MISC HELPERS
# ==============================================================================

def count_images_in_folder(folder: str) -> int:
    """Count images in folder."""
    import glob
    count = 0
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']:
        count += len(glob.glob(os.path.join(folder, ext)))
    return count


def get_all_image_paths(folder: str, recursive: bool = False) -> List[str]:
    """Get all image paths in folder."""
    import glob
    images = []
    pattern = "**/" if recursive else ""
    for ext in ['jpg', 'jpeg', 'png', 'webp', 'bmp']:
        images.extend(glob.glob(os.path.join(folder, f"{pattern}*.{ext}"), recursive=recursive))
    return sorted(images)


def ensure_dir(path: str) -> str:
    """Create directory if not exists and return path."""
    os.makedirs(path, exist_ok=True)
    return path


def read_caption(image_path: str, extension: str = ".txt") -> str:
    """Read caption file corresponding to image."""
    cap_path = os.path.splitext(image_path)[0] + extension
    if os.path.exists(cap_path):
        with open(cap_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return ""


def write_caption(image_path: str, caption: str, extension: str = ".txt"):
    """Write caption file corresponding to image."""
    cap_path = os.path.splitext(image_path)[0] + extension
    with open(cap_path, 'w', encoding='utf-8') as f:
        f.write(caption)
