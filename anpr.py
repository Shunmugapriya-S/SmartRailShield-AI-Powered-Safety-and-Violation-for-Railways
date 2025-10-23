import cv2
from ultralytics import YOLO
import easyocr
import re
import pyttsx3

# -----------------------------
# Model Initialization
# -----------------------------
print("🚀 Loading models...")
model = YOLO("license_palte_best.pt")
reader = easyocr.Reader(['en'], gpu=False)
engine = pyttsx3.init()
engine.setProperty('rate', 160)
print("✅ Models loaded successfully.")

plate_pattern = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{3}$")

# -----------------------------
# Helper Functions
# -----------------------------
def correct_plate_format(ocr_text):
    mapping_num_to_alpha = {"0": "O", "1": "I", "5": "S", "8": "B"}
    mapping_alpha_to_num = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8"}
    ocr_text = ocr_text.upper().replace(" ", "")
    if len(ocr_text) != 7:
        return ""
    corrected = []
    for i, ch in enumerate(ocr_text):
        if i < 2 or i >= 4:
            if ch.isdigit() and ch in mapping_num_to_alpha:
                corrected.append(mapping_num_to_alpha[ch])
            elif ch.isalpha():
                corrected.append(ch)
            else:
                return ""
        else:
            if ch.isalpha() and ch in mapping_alpha_to_num:
                corrected.append(mapping_alpha_to_num[ch])
            elif ch.isdigit():
                corrected.append(ch)
            else:
                return ""
    return "".join(corrected)


def recognise_plate(plate_crop):
    if plate_crop.size == 0:
        return ""
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plate_resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    try:
        ocr_result = reader.readtext(
            plate_resized, detail=0,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        )
        if len(ocr_result) > 0:
            candidate = correct_plate_format(ocr_result[0])
            if candidate and plate_pattern.match(candidate):
                return candidate
    except:
        pass
    return ""


# -----------------------------
# Video Processing with Single Alert
# -----------------------------
def process_video():
    input_path = "license_video.mp4"
    output_path = "output.mp4"

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("❌ Cannot open video")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Flag to make voice alert once
    level_crossing_alerted = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
                conf = float(box.conf.cpu().numpy())
                if conf < 0.3:
                    continue

                plate_crop = frame[y1:y2, x1:x2]
                text = recognise_plate(plate_crop)

                # Draw rectangle & text
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                if text:
                    cv2.putText(frame, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    print(f"🚗 License Plate Detected: {text}")

                    # Trigger **voice alert once**
                    if not level_crossing_alerted:
                        engine.say("⚠️ Attention! The train is near the level crossing!")
                        engine.runAndWait()
                        level_crossing_alerted = True

        out.write(frame)

    cap.release()
    out.release()
    print("✅ Video saved as 'output.mp4'")


if __name__ == "__main__":
    process_video()
