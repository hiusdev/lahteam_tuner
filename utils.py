"""
LahTeam Musubi Tuner - Helper Functions
Chỉ chứa các function hỗ trợ có thể tái sử dụng.

Usage trong Colab:
    !git clone https://github.com/LahTeam/colab_musubi_tuner
    import sys
    sys.path.append('/content/colab_musubi_tuner')
    from lahteam_tuner.utils import download_component, patch_logger_files
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
    Tải một thành phần mô hình (file hoặc folder) từ HuggingFace.
    
    Args:
        component_name: Tên thành phần (VAE, Text Encoder, DiT)
        base_dir: Thư mục gốc để lưu
        config: Dict chứa {type, repo_id, filename/folder, subfolder, local_folder}
        hf_token: HuggingFace token (optional)
    
    Returns:
        str: Đường dẫn đến file/folder đã tải, hoặc None nếu lỗi
    
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
            print(f"✅ [{component_name.upper()}] Folder đã tồn tại: {local_folder}")
            return full_path
        
        print(f"⏳ [{component_name.upper()}] Đang tải folder '{folder}' từ {repo_id}...")
        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=f"{folder}/*" if folder else None,
                local_dir=base_dir,
                local_dir_use_symlinks=False,
                token=hf_token
            )
            print(f"   -> Tải folder thành công!")
            return full_path
        except Exception as e:
            print(f"❌ Lỗi khi tải [{component_name.upper()}]: {e}")
            return None
    else:
        subfolder = config.get('subfolder', '')
        filename = config['filename']
        full_path = os.path.join(base_dir, subfolder, filename) if subfolder else os.path.join(base_dir, filename)

        if os.path.exists(full_path):
            print(f"✅ [{component_name.upper()}] File đã tồn tại: {filename}")
            return full_path

        print(f"⏳ [{component_name.upper()}] Đang tải '{filename}'...")
        os.makedirs(os.path.dirname(full_path) if os.path.dirname(full_path) else base_dir, exist_ok=True)
        
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=f"{subfolder}/{filename}" if subfolder else filename,
                local_dir=base_dir,
                token=hf_token
            )
            print(f"   -> Tải thành công!")
            return full_path
        except Exception as e:
            print(f"❌ Lỗi khi tải [{component_name.upper()}]: {e}")
            return None


def download_and_unzip(
    url: str, 
    unzip_dir: str, 
    zip_filename: str = "temp.zip",
    hf_token: Optional[str] = None
) -> bool:
    """
    Tải file zip từ URL và giải nén.
    
    Args:
        url: URL file zip
        unzip_dir: Thư mục giải nén
        zip_filename: Tên file zip tạm
        hf_token: Token cho private repos
    
    Returns:
        bool: True nếu thành công
    """
    import subprocess
    import zipfile
    
    if not url:
        return False
    
    if os.path.exists(unzip_dir) and any(os.scandir(unzip_dir)):
        print(f"✅ Thư mục '{unzip_dir}' đã có dữ liệu. Bỏ qua.")
        return True
    
    print(f"⏳ Đang tải từ: {url}")
    zip_path = f"/content/{zip_filename}"
    header = f'"Authorization: Bearer {hf_token}"' if hf_token else ''
    
    cmd = f'aria2c --console-log-level=error -c -x 16 -k 1M -s 16 --header={header} -d /content -o {zip_filename} "{url}"'
    subprocess.run(cmd, shell=True)
    
    if os.path.exists(zip_path):
        print(f"📦 Đang giải nén...")
        os.makedirs(unzip_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(unzip_dir)
        os.remove(zip_path)
        print(f"✅ Giải nén hoàn tất!")
        return True
    
    print(f"❌ Tải thất bại!")
    return False


# ==============================================================================
# LOGGER PATCH HELPERS
# ==============================================================================

def replace_logger_with_print(file_path: str) -> bool:
    """
    Thay thế logger.info/warning/error/debug thành print().
    
    Args:
        file_path: Đường dẫn file Python
    
    Returns:
        bool: True nếu có thay đổi
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pattern = r'\blogger\.(info|warning|error|debug)\s*\((.*?)\)'
    replaced = re.sub(pattern, r'print(\2)', content, flags=re.DOTALL)
    
    if replaced != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(replaced)
        return True
    return False


def remove_setup_logging_calls(file_path: str) -> bool:
    """
    Xóa các lời gọi setup_logging().
    
    Args:
        file_path: Đường dẫn file Python
    
    Returns:
        bool: True nếu có thay đổi
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    removed = False
    
    for line in lines:
        stripped = line.strip()
        if stripped in ["setup_logging()", "setup_logging(args, reset=True)"]:
            removed = True
            continue
        new_lines.append(line)
    
    if removed:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
    return removed


def patch_logger_files(repo_dir: str, verbose: bool = False):
    """
    Patch tất cả file Python trong thư mục: thay logger thành print, xóa setup_logging.
    
    Args:
        repo_dir: Thư mục gốc cần patch
        verbose: In chi tiết các file đã sửa
    """
    count = 0
    for root, _, files in os.walk(repo_dir):
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                try:
                    changed = replace_logger_with_print(path)
                    changed2 = remove_setup_logging_calls(path)
                    if (changed or changed2) and verbose:
                        print(f"✅ Patched: {path}")
                        count += 1
                except Exception as e:
                    if verbose:
                        print(f"⚠️ Error: {path}: {e}")
    print(f"🔧 Đã patch {count} files")


# ==============================================================================
# DATASET CONFIG HELPERS
# ==============================================================================

def find_images_in_folder(path: str) -> bool:
    """Kiểm tra folder có chứa ảnh không."""
    import glob
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']:
        if glob.glob(os.path.join(path, ext)):
            return True
    return False


def get_repeats_from_folder_name(folder_name: str, default: int = 10) -> int:
    """
    Lấy số repeats từ tên folder (format: 10_name).
    
    Args:
        folder_name: Tên folder
        default: Giá trị mặc định nếu không tìm thấy
    
    Returns:
        int: Số repeats
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
    Build command line arguments từ dict config.
    
    Args:
        config: Dict chứa các tham số training
    
    Returns:
        str: Chuỗi arguments
    
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
    Lấy prefix cho script dựa trên model_type.
    
    Args:
        model_type: Loại model
    
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
    Lấy network module dựa trên model_type.
    
    Args:
        model_type: Loại model
    
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

# Max blocks_to_swap cho FLUX.2
FLUX2_BLOCKS_LIMIT = {
    "flux2_dev": 29,
    "flux2_klein_4b": 13,
    "flux2_klein_base_4b": 13,
    "flux2_klein_9b": 16,
    "flux2_klein_base_9b": 16,
}


def get_flux2_version(model_type: str) -> str:
    """Lấy FLUX.2 version string từ model_type."""
    return FLUX2_VERSION_MAP.get(model_type, "")


def get_max_blocks_to_swap(model_type: str) -> int:
    """Lấy giới hạn blocks_to_swap cho model_type."""
    return FLUX2_BLOCKS_LIMIT.get(model_type, 36)


def is_flux2_model(model_type: str) -> bool:
    """Kiểm tra có phải FLUX.2 model không."""
    return model_type.startswith("flux2_")


def is_edit_model(model_type: str) -> bool:
    """Kiểm tra có phải Edit model không."""
    return model_type in ["qwen_image_edit", "qwen_image_edit_2509"]


def get_train_script(model_type: str) -> str:
    """Lấy tên file script training."""
    if model_type.startswith("flux2_"):
        return "flux_2_train_network.py"
    elif model_type == "z_image_turbo":
        return "zimage_train_network.py"
    else:
        return "qwen_image_train_network.py"


def get_cache_script_prefix(model_type: str) -> str:
    """Lấy prefix cho cache scripts (latents & text encoder)."""
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
    Tạo file dataset_config.toml theo format musubi-tuner chính thức.
    
    Args:
        data_dir: Thư mục chứa ảnh (có thể có subfolders 10_name)
        control_dir: Thư mục control images (cho FLUX.2/Qwen Edit)
        cache_dir: Thư mục cache (mặc định = data_dir)
        config_dir: Thư mục lưu file config
        resolution: (width, height) mặc định [960, 544]
        control_resolution: Resolution cho control images
                           - Qwen Edit: đề nghị [1024, 1024]
                           - FLUX.2 1 control: [2024, 2024]
                           - FLUX.2 multi control: [1024, 1024]
        batch_size: Batch size, mặc định 1
        default_repeats: Số repeats mặc định, mặc định 1
        caption_extension: Extension caption files
        enable_bucket: Bật bucketing
        bucket_no_upscale: Không upscale khi bucket
        no_resize_control: Không resize control images (dùng cho FLUX.2)
        model_type: Loại model để auto-detect settings
    
    Returns:
        str: Đường dẫn file config đã tạo
    
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
    
    # Quét subfolders
    try:
        subfolders = sorted([f.path for f in os.scandir(data_dir) if f.is_dir()])
    except FileNotFoundError:
        subfolders = []
    
    # Thêm root dir nếu có ảnh
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
        
        # Tạo dataset entry
        dataset_item = {
            "image_directory": folder_path,
            "num_repeats": repeats
        }
        
        # Cache directory (khác cho mỗi dataset)
        if cache_dir:
            dataset_cache = os.path.join(cache_dir, f"cache_{idx}")
            dataset_item["cache_directory"] = dataset_cache
        
        # Control directory (cho Edit mode hoặc FLUX.2)
        if control_dir and (is_edit or is_flux2):
            dataset_item["control_directory"] = control_dir
            
            if control_resolution:
                dataset_item["control_resolution"] = list(control_resolution)
            
            if no_resize_control:
                dataset_item["no_resize_control"] = True
        
        print(f"   ✅ Dataset: '{folder_name}' | Repeats: {repeats}")
        datasets.append(dataset_item)
    
    if not datasets:
        raise ValueError("Không tìm thấy dữ liệu ảnh nào!")
    
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
    
    # Write TOML với format đúng
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
    print(f"✅ Đã tạo: {config_path}")
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
    Tạo file sample prompts từ dataset config.
    
    Args:
        dataset_config_path: Path đến dataset_config.toml
        output_path: Path lưu file samples.txt
        samples_per_dataset: Số samples mỗi dataset
        sample_width/height: Kích thước sample
        sample_steps: Số bước sampling
        control_dir: Thư mục control images (nếu cần)
    
    Returns:
        str: Path đến file samples.txt
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
                    
                    # Thêm control image nếu có
                    if ctrl_dir:
                        basename = os.path.splitext(os.path.basename(img))[0]
                        ctrl_candidates = glob.glob(os.path.join(ctrl_dir, f"{basename}.*"))
                        if ctrl_candidates:
                            sample_line += f" --ci {ctrl_candidates[0]}"
                    
                    lines.append(sample_line)
    
    if lines:
        with open(output_path, "w", encoding='utf-8') as f:
            f.write("\n".join(lines))
        print(f"✅ Tạo {len(lines)} sample prompts: {output_path}")
    
    return output_path


# ==============================================================================
# COLAB STORE HELPERS
# ==============================================================================

def store_variables(var_dict: Dict[str, Any]):
    """
    Lưu biến vào IPython %store (chạy trong Colab).
    
    Args:
        var_dict: Dict {tên_biến: giá_trị}
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
        
        print(f"✅ Đã lưu {len(var_dict)} biến")
    except Exception as e:
        print(f"⚠️ Không thể store biến: {e}")


def restore_variables(*var_names: str) -> Dict[str, Any]:
    """
    Khôi phục biến từ IPython %store.
    
    Args:
        var_names: Tên các biến cần khôi phục
    
    Returns:
        dict: {tên_biến: giá_trị}
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
        print(f"⚠️ Không thể restore biến: {e}")
        return {}


# ==============================================================================
# MISC HELPERS
# ==============================================================================

def count_images_in_folder(folder: str) -> int:
    """Đếm số ảnh trong folder."""
    import glob
    count = 0
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']:
        count += len(glob.glob(os.path.join(folder, ext)))
    return count


def get_all_image_paths(folder: str, recursive: bool = False) -> List[str]:
    """Lấy tất cả đường dẫn ảnh trong folder."""
    import glob
    images = []
    pattern = "**/" if recursive else ""
    for ext in ['jpg', 'jpeg', 'png', 'webp', 'bmp']:
        images.extend(glob.glob(os.path.join(folder, f"{pattern}*.{ext}"), recursive=recursive))
    return sorted(images)


def ensure_dir(path: str) -> str:
    """Tạo thư mục nếu chưa tồn tại và trả về path."""
    os.makedirs(path, exist_ok=True)
    return path


def read_caption(image_path: str, extension: str = ".txt") -> str:
    """Đọc caption file tương ứng với ảnh."""
    cap_path = os.path.splitext(image_path)[0] + extension
    if os.path.exists(cap_path):
        with open(cap_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return ""


def write_caption(image_path: str, caption: str, extension: str = ".txt"):
    """Ghi caption file tương ứng với ảnh."""
    cap_path = os.path.splitext(image_path)[0] + extension
    with open(cap_path, 'w', encoding='utf-8') as f:
        f.write(caption)

