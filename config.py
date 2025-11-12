# config.py
import yaml
from pathlib import Path
from typing import List, Dict, Optional

from pydantic import BaseModel, Field, FilePath, DirectoryPath

# --- Nested Models for better structure ---

class WeightsPaths(BaseModel):
    """Defines and validates paths to backbone weights."""
    pvt_v2_b2: FilePath
    pvt_v2_b4: FilePath
    pvt_v2_b5: FilePath

# --- Main Configuration Class ---

class Config(BaseModel):

    batch_size: int = Field(..., gt=0, description="Batch size for training.")
    batch_size_valid: int = Field(..., gt=0, description="Batch size for validation.")
    epochs: int = Field(..., gt=0, description="Total number of training epochs.")
    lr: float = Field(..., gt=0, description="Learning rate.")
    weight_decay: float = Field(..., ge=0, description="Weight decay for the optimizer.")
    rand_seed: int = 42
    precisionHigh: bool = True

    # --- Dataset Configuration ---
    train_dir: DirectoryPath   # Validates that the train directory exists
    test_dir: DirectoryPath    # Validates that the test directory exists
    load_all: bool = False

    # --- Model Configuration ---
    backbone: str
    img_size: int = Field(..., gt=0)
    lateral_channels: List[int]
    resume: Optional[str] = None # Allows the field to be missing or empty ""
    compile: bool = True

    # --- Multi-GPU Settings ---
    device_ids: List[int]
    multi_GPU: bool = False

    # --- Save Settings ---
    save_model_dir: str 
    name: str 
    save_last: int = Field(..., ge=0)
    save_step: int = Field(..., gt=0)

    # --- Data Augmentation ---
    preproc_methods: List[str]

    # --- Path Settings ---
    weights: WeightsPaths

    # --- System ---
    num_workers: int = Field(..., ge=0)
    
    # --- Helper Properties (for cleaner code in train.py) ---
    @property
    def is_ddp(self) -> bool:
        """Returns True if Distributed Data Parallel (DDP) should be used."""
        return self.multi_GPU and len(self.device_ids) > 1

    @property
    def save_path(self) -> Path:
        """
        Generates the full, unique path for saving models and logs for this run.
        It creates the directory if it doesn't exist.
        """
        path = Path(self.save_model_dir) / self.name
        path.mkdir(parents=True, exist_ok=True)
        return path


def load_config(config_path: str = "config.yaml") -> Config:
    """Loads a YAML configuration file into a validated Config object."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Pydantic will raise a ValidationError if the config is invalid
    return Config(**config_dict)