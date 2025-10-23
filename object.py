# object.py
from ultralytics import YOLO
import cv2

class ObstacleDetector:
    def __init__(self, model_path="best.pt", camera_index=0):
        # Load YOLO model
        self.model = YOLO(model_path)
        # Initialize camera (DirectShow for Windows)
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam. Check camera index or permissions.")

    def get_obstacles(self, current_location={"lat":0.0, "lon":0.0}, show_frame=True):
        """
        Detect obstacles from webcam and return list with lat/lon & class.
        Returns: obstacles list, quit_flag (True if 'q' pressed)
        """
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame from camera")
            return [], False

        obstacles = []

        # Run YOLO prediction
        results = self.model.predict(frame, conf=0.5, verbose=False)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls = int(box.cls[0])
                class_name = self.model.names.get(cls, "Unknown")

                # Draw bounding box and label
                if show_frame:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, class_name, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Approximate lat/lon near the loco
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                obs_lat = current_location["lat"] + (center_y / 100000)
                obs_lon = current_location["lon"] + (center_x / 100000)

                obstacles.append({"lat": obs_lat, "lon": obs_lon, "class": class_name})

        # Show live frame if enabled
        quit_flag = False
        if show_frame:
            try:
                cv2.imshow("Obstacle Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    quit_flag = True
            except cv2.error as e:
                print("cv2.imshow error:", e)

        return obstacles, quit_flag

    def release(self):
        """Release camera safely."""
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()


# ---------------- Standalone test -----------------
if __name__ == "__main__":
    detector = ObstacleDetector("best.pt")
    current_location = {"lat": 0.0, "lon": 0.0}

    try:
        while True:
            obstacles, quit_flag = detector.get_obstacles(current_location, show_frame=True)
            if quit_flag:
                break
            for obs in obstacles:
                print(f"Detected {obs['class']} at lat:{obs['lat']:.6f}, lon:{obs['lon']:.6f}")
    finally:
        detector.release()
