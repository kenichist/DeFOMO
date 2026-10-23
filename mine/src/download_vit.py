import os
from pathlib import Path
from transformers import ViTModel, AutoImageProcessor

# Get the current directory using pathlib (works better on Windows)
current_dir = Path(__file__).parent.absolute()

# Download and save the model
model_name = "google/vit-base-patch16-224-in21k"
save_path = current_dir / "models" / "vit-base-patch16-224-in21k"

# Create the directory if it doesn't exist
save_path.mkdir(parents=True, exist_ok=True)

print(f"Downloading {model_name}...")
print(f"Current directory: {current_dir}")
print(f"Save path: {save_path}")

model = ViTModel.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

print(f"Saving to {save_path}...")
model.save_pretrained(str(save_path))
processor.save_pretrained(str(save_path))

print("Done!")