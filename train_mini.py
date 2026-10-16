import os
import shutil
import random
from pathlib import Path
from ultralytics import YOLO

def prepare_mini_dataset():
    src_dir = Path(r"C:\Users\Rakesh Sharma\OneDrive\Desktop\New folder\adas-vision\imgs\train")
    dataset_dir = Path(r"C:\Users\Rakesh Sharma\OneDrive\Desktop\New folder\adas-vision\datasets\driver_behavior_mini")
    
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir) # Clear it for a fresh mini run
        
    print("Creating a MINI dataset (20 train / 5 val images per class) for quick testing...")
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    
    for class_folder in src_dir.iterdir():
        if not class_folder.is_dir(): continue
        
        c_name = class_folder.name
        (train_dir / c_name).mkdir(parents=True, exist_ok=True)
        (val_dir / c_name).mkdir(parents=True, exist_ok=True)
        
        images = list(class_folder.glob("*.jpg"))
        random.shuffle(images)
        
        # Take just 25 images total per class
        subset = images[:25]
        train_imgs = subset[:20]
        val_imgs = subset[20:]
        
        for img in train_imgs:
            shutil.copy(str(img), str(train_dir / c_name / img.name))
        for img in val_imgs:
            shutil.copy(str(img), str(val_dir / c_name / img.name))
            
    print(f"Mini dataset ready at {dataset_dir}!")
    return str(dataset_dir)

def train_model(data_path):
    print("Starting fast YOLOv8 classification training on MINI dataset...")
    model = YOLO("yolov8n-cls.pt")
    
    # Train for 3 epochs on the tiny dataset (should take seconds/minutes on CPU)
    results = model.train(
        data=data_path,
        epochs=3, 
        imgsz=128,
        batch=8,
        name="driver_behavior_mini",
        workers=0 # Avoid multiprocessing issues on Windows
    )
    print("Mini training complete! Model saved in 'runs/classify/driver_behavior_mini/weights/best.pt'")

if __name__ == "__main__":
    data_path = prepare_mini_dataset()
    train_model(data_path)
