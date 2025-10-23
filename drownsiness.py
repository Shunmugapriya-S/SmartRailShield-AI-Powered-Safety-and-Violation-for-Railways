import cv2
import mediapipe as mp
import pyttsx3
import threading
import time

# ---------------------------
# Global variables
# ---------------------------
COUNTER = 0
SIREN_ALERTED = False
YAWN_ALERTED = False

engine = pyttsx3.init()
engine.setProperty('rate', 150)

def siren_alert():
    for _ in range(5):
        print("BEEP!")  # replace with winsound.Beep(1000, 500) on Windows
        time.sleep(0.2)

def voice_alert(msg):
    engine.say(msg)
    engine.runAndWait()

# ---------------------------
# Mediapipe setup
# ---------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Eye and mouth landmarks
LEFT_EYE = [33, 133]  # landmarks around left eye
RIGHT_EYE = [362, 263]  # landmarks around right eye
MOUTH = [13, 14]  # top and bottom lips

# ---------------------------
# Utility functions
# ---------------------------
def euclidean_dist(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5

def process_frame(frame):
    global COUNTER, SIREN_ALERTED, YAWN_ALERTED

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            # Get eye aspect ratio approximation
            left_eye = [(int(face_landmarks.landmark[i].x*w), int(face_landmarks.landmark[i].y*h)) for i in LEFT_EYE]
            right_eye = [(int(face_landmarks.landmark[i].x*w), int(face_landmarks.landmark[i].y*h)) for i in RIGHT_EYE]
            left_eye_ratio = euclidean_dist(left_eye[0], left_eye[1])
            right_eye_ratio = euclidean_dist(right_eye[0], right_eye[1])

            # Drowsiness detection
            if left_eye_ratio < 5 and right_eye_ratio < 5:  # adjust threshold
                COUNTER += 1
                if COUNTER >= 15 and not SIREN_ALERTED:
                    threading.Thread(target=siren_alert, daemon=True).start()
                    SIREN_ALERTED = True
            else:
                COUNTER = 0
                SIREN_ALERTED = False

            # Mouth open / Yawning detection
            mouth = [(int(face_landmarks.landmark[i].x*w), int(face_landmarks.landmark[i].y*h)) for i in MOUTH]
            mouth_open_ratio = euclidean_dist(mouth[0], mouth[1])
            if mouth_open_ratio > 25 and not YAWN_ALERTED:  # adjust threshold
                threading.Thread(target=voice_alert, args=("Attention! Driver is yawning",), daemon=True).start()
                YAWN_ALERTED = True
            elif mouth_open_ratio <= 25:
                YAWN_ALERTED = False

    return frame

# ---------------------------
# Video capture
# ---------------------------
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = process_frame(frame)
    cv2.imshow("Driver Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
