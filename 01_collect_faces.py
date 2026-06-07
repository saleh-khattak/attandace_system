import cv2
import os

def collect_faces(name, num_samples=300):  # Increased from 100 → 300
    save_path = f"dataset/{name}"
    os.makedirs(save_path, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    cap = cv2.VideoCapture(0)
    count = 0

    # Instructions split into phases so person moves naturally
    phases = {
        0  : "Look STRAIGHT at camera",
        100: "Turn HEAD SLIGHTLY LEFT",
        200: "Turn HEAD SLIGHTLY RIGHT",
    }

    print(f"[INFO] Collecting 300 samples for '{name}'.")
    print("[INFO] Follow the on-screen instructions as count increases.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            cv2.imwrite(f"{save_path}/{count}.jpg", face_img)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Show current phase instruction on screen
        instruction = phases.get(
            max(k for k in phases if k <= count),  # closest phase below count
            "Look STRAIGHT at camera"
        )

        cv2.putText(frame, instruction,
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Sample {count}/{num_samples}",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Collecting Faces — Press Q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or count >= num_samples:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[DONE] Collected {count} samples for '{name}'")

if __name__ == "__main__":
    name = input("Enter person's name: ").strip()
    collect_faces(name)