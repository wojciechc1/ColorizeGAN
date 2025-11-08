from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

CONFIG = {
    # Hyperparameters
    "batch_size": 32,
    "num_epochs": 2,
    "lr": 1e-5,
    "betas": (0.7, 0.999),
    "l1_weight": 30.0,
    "scheduler_step": 4,
    "scheduler_gamma": 0.9,
    "dropout": 0.5,  # only uses by discriminator
    # Device
    "device": "cuda",  # "cuda" / "cuda:0" / "cpu"
    # Dataset
    "dataset_path": r"C:\Users\wojte\Desktop\projects\Datasets\cifar10\data",
    "download": False,
    "train_samples": 100,
    "val_samples": 100,
    # Paths
    "generator_path": PROJECT_ROOT / "checkpoints" / "generator1.pth",
    "discriminator_path": PROJECT_ROOT / "checkpoints" / "discriminator1.pth",
    "generator_save_path": PROJECT_ROOT / "checkpoints" / "generator_beta.pth",
    "discriminator_save_path": PROJECT_ROOT / "checkpoints" / "discriminator_beta.pth",
    "plots_save_path": "./plots",
    "log_interval": 10,
}
