"""File handling utilities"""
import os
import shutil
from pathlib import Path
from typing import List, Optional

def ensure_directory(path: str) -> Path:
    """Ensure directory exists, create if not"""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def get_file_extension(file_path: str) -> str:
    """Get file extension in lowercase"""
    return Path(file_path).suffix.lower()

def is_supported_file(file_path: str) -> bool:
    """Check if file type is supported"""
    supported_extensions = {'.pdf', '.docx', '.doc', '.txt', '.csv'}
    return get_file_extension(file_path) in supported_extensions

def safe_filename(filename: str) -> str:
    """Create a safe filename by removing invalid characters"""
    invalid_chars = '<>:"/\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def copy_file_safely(src: str, dst: str) -> bool:
    """Copy file with error handling"""
    try:
        shutil.copy2(src, dst)
        return True
    except Exception:
        return False