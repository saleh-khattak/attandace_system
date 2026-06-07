import cv2  # OpenCV — for webcam access, face detection, drawing on frames
import os   # For creating folders to save images

def collect_faces(name, num_samples=100):
    """
    Captures face images from webcam and saves them to dataset/<name>/
    
    Parameters:
        name        : Name of the person (used as folder name)
        num_samples : How many face images to capture (default 100)
    """

    # ── 1. Create the save folder ──────────────────────────────────────────
    # Example: dataset/Muhammad/
    # exist_ok=True means: don't crash if folder already exists
    save_path = f"dataset/{name}"
    os.makedirs(save_path, exist_ok=True)

    # ── 2. Load the Face Detector ──────────────────────────────────────────
    # Haar Cascade is a pre-trained XML model by OpenCV
    # It detects WHERE a face is in an image (not WHO it is)
    # cv2.data.haarcascades gives the path to OpenCV's built-in cascades
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    # ── 3. Open the Webcam ─────────────────────────────────────────────────
    # 0 = default webcam. Use 1 or 2 if you have multiple cameras
    cap = cv2.VideoCapture(0)

    count = 0  # Tracks how many face images we've saved so far

    print(f"[INFO] Collecting {num_samples} samples for '{name}'.")
    print("[INFO] Look at the camera. Press Q to stop early.")

    # ── 4. Main Loop — read frames continuously ────────────────────────────
    while True:
        ret, frame = cap.read()
        # ret  → True if frame was successfully captured
        # frame → the actual image (numpy array, BGR format)

        if not ret:
            print("[ERROR] Could not read from webcam.")
            break

        # ── 5. Convert to Grayscale ────────────────────────────────────────
        # Face detection & recognition works on grayscale images
        # Grayscale = simpler (1 channel) vs color (3 channels BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── 6. Detect Faces in the Frame ───────────────────────────────────
        # detectMultiScale scans the image at multiple scales
        # Returns a list of rectangles: [(x, y, w, h), ...]
        #   x, y = top-left corner of face box
        #   w, h = width and height of face box
        # 1.3 = scaleFactor  → how much image is reduced each scan pass
        # 5   = minNeighbors → how many neighbors needed to confirm a face
        #                      (higher = fewer false detections)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=7)

        # ── 7. Process Each Detected Face ──────────────────────────────────
        for (x, y, w, h) in faces:
            count += 1

            # Crop just the face region from the grayscale frame
            # gray[y:y+h, x:x+w] → numpy array slicing (rows, cols)
            face_img = gray[y : y+h, x : x+w]

            # Save the cropped face as a .jpg file
            # Example: dataset/Muhammad/1.jpg, dataset/Muhammad/2.jpg ...
            cv2.imwrite(f"{save_path}/{count}.jpg", face_img)

            # Draw a green rectangle around the detected face on screen
            # (x,y) = top-left corner, (x+w, y+h) = bottom-right corner
            # (0,255,0) = green in BGR,  2 = line thickness
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Show sample count on screen above the face box
            cv2.putText(
                frame,
                f"Sample {count}/{num_samples}",
                (x, y - 10),                    # Position: just above face box
                cv2.FONT_HERSHEY_SIMPLEX,        # Font style
                0.6,                             # Font size
                (0, 255, 0),                     # Green color
                2                                # Thickness
            )

        # ── 8. Show the Live Frame ─────────────────────────────────────────
        cv2.imshow("Collecting Faces — Press Q to quit", frame)

        # ── 9. Stop Conditions ─────────────────────────────────────────────
        # waitKey(1) waits 1ms for a key press, returns the key code
        # 0xFF masks the result to get the last 8 bits (cross-platform fix)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("[INFO] Stopped early by user.")
            break

        if count >= num_samples:
            print(f"[DONE] Collected {count} samples!")
            break

    # ── 10. Cleanup ────────────────────────────────────────────────────────
    cap.release()          # Release the webcam so other apps can use it
    cv2.destroyAllWindows() # Close all OpenCV windows


# ── Entry Point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    name = input("Enter person's name: ").strip()
    collect_faces(name)