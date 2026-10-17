import os
import shutil
import random
from pathlib import Path
from ultralytics import YOLO

def prepare_full_dataset():
    src_dir = Path(r"C:\Users\Rakesh Sharma\OneDrive\Desktop\New folder\adas-vision\imgs\train")
    dataset_dir = Path(r"C:\Users\Rakesh Sharma\OneDrive\Desktop\New folder\adas-vision\datasets\driver_behavior_full")
    
    if dataset_dir.exists():
        print(f"Dataset already split at {dataset_dir}. Skipping copy.")
        return str(dataset_dir)
        
    print("Splitting FULL 22,000-image dataset into train (80%) and val (20%)...")
    print("This might take a couple of minutes to copy files...")
    
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    
    for class_folder in src_dir.iterdir():
        if not class_folder.is_dir(): continue
        
        c_name = class_folder.name
        (train_dir / c_name).mkdir(parents=True, exist_ok=True)
        (val_dir / c_name).mkdir(parents=True, exist_ok=True)
        
        images = list(class_folder.glob("*.jpg"))
        random.shuffle(images)
        
        # 80/20 Split on the FULL dataset
        split_idx = int(len(images) * 0.8)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]
        
        print(f"Copying Class {c_name}: {len(train_imgs)} train, {len(val_imgs)} val...")
        
        for img in train_imgs:
            shutil.copy(str(img), str(train_dir / c_name / img.name))
        for img in val_imgs:
            shutil.copy(str(img), str(val_dir / c_name / img.name))
            
    print(f"Full dataset ready at {dataset_dir}!")
    return str(dataset_dir)

def train_model(data_path):
    print("Starting full YOLOv8 classification training...")
    print("NOTE: On a CPU, this will take several hours. Leave the PC running.")
    
    # Start with the Nano model (fastest)
    model = YOLO("yolov8n-cls.pt")
    
    # Train for 15 epochs (Good baseline for 22,000 images)
    # Using workers=0 to prevent multiprocessing freezing on Windows
    results = model.train(
        data=data_path,
        epochs=15, 
        imgsz=224,   # Standard classification resolution
        batch=16,
        name="driver_behavior_full",
        workers=0
    )
    print("Full training complete! Model saved in 'runs/classify/driver_behavior_full/weights/best.pt'")

if __name__ == "__main__":
    data_path = prepare_full_dataset()
    train_model(data_path)
