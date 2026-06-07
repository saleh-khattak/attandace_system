import cv2
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime

def load_existing_attendance(filepath, today):
    """
    Loads already-marked attendance for TODAY from the Excel file.
    
    Returns a dict of { name: time } for people already marked today.
    This is used at startup so we don't re-mark someone who was
    marked in a previous run of the program on the same day.
    """
    if not os.path.exists(filepath):
        return {}  # No file yet → no one marked

    existing_df = pd.read_excel(filepath, sheet_name="Attendance")

    # Filter only today's records
    today_df = existing_df[existing_df["Date"] == today]

    # Convert to dict { name: time }
    already_marked = {}
    for _, row in today_df.iterrows():
        already_marked[row["Name"]] = row["Time"]

    return already_marked


def mark_attendance(name, attendance_log, already_marked_today):
    """
    Marks attendance for a recognized person.

    3 cases:
      1. Already marked in THIS session     → skip silently (avoid spam)
      2. Already marked in a PREVIOUS run today → show 'already marked' msg
      3. New person not marked yet           → mark and record

    Returns:
        attendance_log  → updated dict
        message         → string to display on screen (or empty string)
    """

    # Case 1: Already marked in this current session → silent skip
    if name in attendance_log:
        return attendance_log, ""

    # Case 2: Was marked in a previous run today → show warning
    if name in already_marked_today:
        msg = f"Already marked: {name}"
        return attendance_log, msg

    # Case 3: Fresh mark → record the time
    attendance_log[name] = datetime.now().strftime("%H:%M:%S")
    print(f"[✓] Attendance marked: {name} at {attendance_log[name]}")
    msg = f"Marked: {name}"
    return attendance_log, msg


def save_to_excel(attendance_log, filepath):
    """
    Appends only NEW people (from this session) to the Excel file.
    Does NOT touch records from other days or previously marked people today.
    """
    os.makedirs("attendance", exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")

    # Build DataFrame from this session's new records only
    new_records = []
    for name, time in attendance_log.items():
        new_records.append({
            "Name"   : name,
            "Date"   : today,
            "Time"   : time,
            "Status" : "Present"
        })
    new_df = pd.DataFrame(new_records)

    # ── Append to existing file or create new ─────────────────────────────
    if os.path.exists(filepath):
        existing_df = pd.read_excel(filepath, sheet_name="Attendance")

        # Combine existing records + new session records
        # No overwrite — we only add new people
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        final_df = new_df

    # Sort for clean viewing
    final_df = final_df.sort_values(["Date", "Name"]).reset_index(drop=True)

    # Write back to Excel
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        final_df.to_excel(writer, sheet_name="Attendance", index=False)

        # Auto-fit column widths
        worksheet = writer.sheets["Attendance"]
        for col in worksheet.columns:
            max_length = max(len(str(cell.value)) for cell in col if cell.value)
            worksheet.column_dimensions[col[0].column_letter].width = max_length + 4

    print(f"\n[SAVED] Excel updated → {filepath}")
    print(final_df.to_string(index=False))


def run_attendance():
    """
    Main function:
      - Loads model + label map
      - Loads already-marked attendance for today (from previous runs)
      - Opens webcam and processes faces in real-time
      - Handles multiple faces simultaneously
      - Shows on-screen message for already-marked people
      - Press S → save new records to Excel
      - Press Q → quit without saving
    """

    filepath = "attendance/attendance_log.xlsx"
    today    = datetime.now().strftime("%Y-%m-%d")

    # ── 1. Load Trained Model ──────────────────────────────────────────────
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer/trainer.yml")

    # ── 2. Load Label Map ──────────────────────────────────────────────────
    with open("trainer/labels.json") as f:
        raw_map = json.load(f)
    label_map = {int(k): v for k, v in raw_map.items()}

    # ── 3. Load Face Detector ──────────────────────────────────────────────
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    # ── 4. Load Already-Marked People for Today ───────────────────────────
    # This handles the case where program was run earlier today
    # e.g. Muhammad was marked at 9am, now it's 11am second run
    already_marked_today = load_existing_attendance(filepath, today)

    if already_marked_today:
        print(f"[INFO] Already marked today: {list(already_marked_today.keys())}")

    # ── 5. Open Webcam ─────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)

    # attendance_log → only NEW people marked in THIS session
    attendance_log = {}

    # message_log → { name: (message_text, timestamp) }
    # Used to show on-screen messages with a timer so they fade after 2 seconds
    message_log = {}

    print("\n[INFO] Attendance system running...")
    print("[INFO] Press  S  to save and quit.")
    print("[INFO] Press  Q  to quit without saving.\n")

    # ── 6. Main Loop ───────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Cannot read webcam.")
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # ── 7. Process All Faces in Current Frame ──────────────────────────
        # This loop runs for EVERY face detected simultaneously
        # e.g. if 3 people are in frame → 3 iterations, each processed
        for (x, y, w, h) in faces:

            face_roi = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
            label_id, confidence = recognizer.predict(face_roi)
            
            print(f"[DEBUG] Predicted: {label_map.get(label_id)} | Confidence: {confidence:.1f}")

            if confidence < 130:
                name  = label_map.get(label_id, "Unknown")
                color = (0, 255, 0)   # Green → recognized

                # Mark attendance and get display message
                attendance_log, message = mark_attendance(
                    name, attendance_log, already_marked_today
                )

                # Store message with current time (for 2-second display timer)
                if message:
                    message_log[name] = (message, datetime.now())

                # Show appropriate label on face box
                label = f"{name} ({confidence:.0f})"

            else:
                name  = "Unknown"
                color = (0, 0, 255)   # Red → unrecognized
                label = f"Unknown ({confidence:.0f})"

            # Draw box around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            # Draw name + confidence above face box
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        # ── 8. Show Status Messages on Screen ─────────────────────────────
        # Display each person's message for 2 seconds then remove it
        y_offset = 70   # Start position for messages (below present count)
        now = datetime.now()

        expired = []
        for person, (msg, timestamp) in message_log.items():
            seconds_elapsed = (now - timestamp).total_seconds()

            if seconds_elapsed < 2:
                # Choose color: orange for already-marked, green for new mark
                msg_color = (0, 165, 255) if "Already" in msg else (0, 255, 0)
                cv2.putText(frame, msg, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, msg_color, 2)
                y_offset += 30  # Stack messages vertically if multiple people
            else:
                expired.append(person)  # Mark for removal after 2 seconds

        # Remove expired messages
        for person in expired:
            del message_log[person]

        # ── 9. Show Present Count ──────────────────────────────────────────
        total_today = len(attendance_log) + len(already_marked_today)
        cv2.putText(frame,
                    f"Present today: {total_today}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 0), 2)

        cv2.imshow("Attendance System  |  S = Save   Q = Quit", frame)

        # ── 10. Key Controls ───────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            if attendance_log:
                save_to_excel(attendance_log, filepath)
            else:
                print("[INFO] No new attendance to save.")
            break
        elif key == ord('q'):
            print("[INFO] Quit without saving.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_attendance()