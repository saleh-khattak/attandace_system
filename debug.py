# debug.py
import cv2
import numpy as np
import os
import json
from PIL import Image

def diagnose():

    print("=" * 50)
    print("FACE RECOGNITION DIAGNOSTIC")
    print("=" * 50)

    # ── 1. Check Dataset ───────────────────────────────
    print("\n[1] DATASET CHECK")
    dataset_path = "dataset"
    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        if os.path.isdir(person_path):
            images = os.listdir(person_path)
            print(f"    {person}: {len(images)} images")

            # Check sizes of first 3 images
            for img_file in images[:3]:
                img = Image.open(os.path.join(person_path, img_file)).convert('L')
                print(f"        {img_file} → size: {img.size}")

    # ── 2. Check Trained Labels ────────────────────────
    print("\n[2] LABEL MAP CHECK")
    with open("trainer/labels.json") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    print(f"    {label_map}")

    # ── 3. Test Prediction on Saved Images ────────────
    print("\n[3] PREDICTION TEST (on saved face images)")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer/trainer.yml")

    correct = 0
    total   = 0

    for label_id, person_name in label_map.items():
        person_path = os.path.join(dataset_path, person_name)
        images      = os.listdir(person_path)[:10]  # Test on first 10 images

        for img_file in images:
            img       = Image.open(os.path.join(person_path, img_file)).convert('L')
            img       = img.resize((100, 100))
            img_array = np.array(img)

            predicted_id, confidence = recognizer.predict(img_array)
            predicted_name           = label_map.get(predicted_id, "Unknown")
            is_correct               = predicted_name == person_name
            if is_correct:
                correct += 1
            total += 1

            print(f"    {person_name} → predicted: {predicted_name} | confidence: {confidence:.1f} | {'✓' if is_correct else '✗ WRONG'}")

    print(f"\n    Accuracy on saved images: {correct}/{total} ({100*correct//total}%)")

    # ── 4. Check Confidence Threshold ─────────────────
    print("\n[4] CONFIDENCE THRESHOLD CHECK")
    print("    Current threshold in code: 70")
    print("    Meaning: confidence < 70 → recognized, >= 70 → Unknown")
    print("    If all confidences above are > 70, threshold needs raising")

    print("\n" + "=" * 50)

diagnose()