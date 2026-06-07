import cv2
import numpy as np
import os
import json
from PIL import Image

def train_model():

    dataset_path = "dataset"
    recognizer   = cv2.face.LBPHFaceRecognizer_create()

    faces     = []
    labels    = []
    label_map = {}
    label_id  = 0

    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path):
            continue

        label_map[label_id] = person_name
        print(f"[INFO] Processing: {person_name} (label {label_id})")

        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)

            img = Image.open(img_path).convert('L')

            # ── FIX: Resize every face to same size before training ────────
            # LBPH compares histograms — they must be from same-sized regions
            # Without this, a face captured close vs far = different sizes
            # = model gets confused and collapses all predictions to one person
            img = img.resize((100, 100))
            # 100x100 is a good balance — big enough for detail, small enough
            # to be fast. All images MUST be the same size for LBPH to work.

            img_array = np.array(img)

            faces.append(img_array)
            labels.append(label_id)

        label_id += 1

    print(f"\n[INFO] Training model on {len(faces)} images...")
    recognizer.train(faces, np.array(labels))

    os.makedirs("trainer", exist_ok=True)
    recognizer.save("trainer/trainer.yml")
    print("[SAVED] Model saved to trainer/trainer.yml")

    with open("trainer/labels.json", "w") as f:
        json.dump(label_map, f, indent=4)
    print("[SAVED] Labels saved to trainer/labels.json")

    print(f"\n[DONE] Training complete for: {list(label_map.values())}")


if __name__ == "__main__":
    train_model()