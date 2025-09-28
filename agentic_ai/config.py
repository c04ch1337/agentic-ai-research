import os
import json
from pathlib import Path
from typing import Dict, Any

class Config:
    """Configuration management for the Agentic AI system"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file and environment"""
        # Default configuration
        default_config = {
            "models": {
                "reasoning_model": "gpt-4",
                "embedding_model": "all-MiniLM-L6-v2",
                "fallback_model": "gpt-3.5-turbo"
            },
            "rag": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "max_search_results": 10,
                "similarity_threshold": 0.7
            },
            "reasoning": {
                "max_steps": 10,
                "confidence_threshold": 0.7,
                "enable_self_reflection": True
            },
            "api": {
                "openai_api_key": os.getenv("OPENAI_API_KEY"),
                "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
                "huggingface_token": os.getenv("HUGGINGFACE_TOKEN")
            },
            "storage": {
                "chroma_db_path": os.getenv("CHROMA_DB_PATH", "./chroma_db"),
                "sessions_path": os.getenv("SESSIONS_PATH", "./sessions"),
                "data_path": os.getenv("DATA_PATH", "./data")
            },
            "logging": {
                "level": os.getenv("LOG_LEVEL", "INFO"),
                "file": os.getenv("LOG_FILE", "./logs/agentic_ai.log")
            }
        }
        
        # Load from file if exists
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                    self._deep_update(default_config, file_config)
            except Exception as e:
                print(f"Warning: Could not load config file {self.config_file}: {e}")
        
        return default_config
    
    def _deep_update(self, base_dict, update_dict):
        """Deep update nested dictionaries"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key: str, default=None):
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self._config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")