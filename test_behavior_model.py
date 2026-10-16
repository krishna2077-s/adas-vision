import cv2
from pathlib import Path
from ultralytics import YOLO

def test_custom_behavior():
    img_path = r"C:\Users\Rakesh Sharma\.gemini\antigravity\brain\47d794cc-1f6d-4e10-9a52-8e8628f6ad91\.user_uploaded\media_1791095434397.png"
    model_path = r"C:\Users\Rakesh Sharma\OneDrive\Desktop\New folder\adas-vision\runs\classify\driver_behavior_full\weights\best.pt"
    out_path = r"C:\Users\Rakesh Sharma\OneDrive\Desktop\New folder\adas-vision\output_behavior_test.png"

    # Human-readable labels for the State Farm dataset
    LABELS = {
        'c0': 'Safe driving',
        'c1': 'Texting (Right Hand)',
        'c2': 'Talking on Phone (Right Hand)',
        'c3': 'Texting (Left Hand)',
        'c4': 'Talking on Phone (Left Hand)',
        'c5': 'Operating Radio',
        'c6': 'Drinking',
        'c7': 'Reaching Behind',
        'c8': 'Hair & Makeup',
        'c9': 'Talking to Passenger'
    }

    print("Loading custom AI Behavior Model...")
    model = YOLO(model_path)
    
    print("Analyzing your image...")
    results = model(img_path)
    
    # Extract top prediction
    probs = results[0].probs
    top_idx = probs.top1
    top_class_id = results[0].names[top_idx] # e.g. "c1"
    top_conf = probs.top1conf.item()
    
    behavior_name = LABELS.get(top_class_id, top_class_id)
    
    print(f"\n--- ANALYSIS COMPLETE ---")
    print(f"Predicted Behavior: {behavior_name}")
    print(f"Confidence: {top_conf * 100:.1f}%")
    
    # Draw it on the image
    img = cv2.imread(img_path)
    cv2.putText(img, f"BEHAVIOR: {behavior_name} ({top_conf*100:.0f}%)", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.imwrite(out_path, img)
    print(f"Saved annotated image to {out_path}")

if __name__ == "__main__":
    test_custom_behavior()
