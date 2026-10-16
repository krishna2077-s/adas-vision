import os
import shutil
import random
from pathlib import Path
from ultralytics import YOLO

def prepare_dataset():
    src_dir = Path(r"C:\Users\Rakesh Sharma\OneDrive\Desktop\New folder\adas-vision\imgs\train")
    dataset_dir = Path(r"C:\Users\Rakesh Sharma\OneDrive\Desktop\New folder\adas-vision\datasets\driver_behavior")
    
    if dataset_dir.exists():
        print(f"Dataset already exists at {dataset_dir}")
        return str(dataset_dir)
        
    print("Splitting dataset into train (80%) and val (20%)...")
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    
    for class_folder in src_dir.iterdir():
        if not class_folder.is_dir(): continue
        
        c_name = class_folder.name
        (train_dir / c_name).mkdir(parents=True, exist_ok=True)
        (val_dir / c_name).mkdir(parents=True, exist_ok=True)
        
        images = list(class_folder.glob("*.jpg"))
        random.shuffle(images)
        
        split_idx = int(len(images) * 0.8)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]
        
        print(f"Class {c_name}: {len(train_imgs)} train, {len(val_imgs)} val")
        
        for img in train_imgs:
            shutil.copy(str(img), str(train_dir / c_name / img.name))
        for img in val_imgs:
            shutil.copy(str(img), str(val_dir / c_name / img.name))
            
    print("Dataset preparation complete!")
    return str(dataset_dir)

def train_model(data_path):
    print("Starting YOLOv8 nano classification training...")
    # Load the base classification model
    model = YOLO("yolov8n-cls.pt")
    
    # Train for 5 epochs (as a quick initial test to see if it works)
    results = model.train(
        data=data_path,
        epochs=5, 
        imgsz=128,   # Smaller image size for faster CPU training
        batch=16,
        name="driver_behavior_v1"
    )
    print("Training complete! Your model is saved in 'runs/classify/driver_behavior_v1/weights/best.pt'")

if __name__ == "__main__":
    data_path = prepare_dataset()
    train_model(data_path)
