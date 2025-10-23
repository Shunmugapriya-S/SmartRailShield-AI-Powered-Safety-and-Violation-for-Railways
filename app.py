from flask import Flask, jsonify, render_template
import threading, time, math
from object import ObstacleDetector
from firebase_config import database as db
from sms_alert import send_sms_alert

app = Flask(__name__)
current_location = {"lat":13.0827, "lon":80.2707}
obstacles = []

# Haversine utility
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlambda = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# Simulate loco movement
def update_location():
    global current_location
    while True:
        current_location["lat"] += 0.00001
        current_location["lon"] += 0.00001
        try:
            db.child("loco").set(current_location)
        except:
            pass
        time.sleep(5)

# Obstacle detection thread
def detect_obstacles_thread(detector):
    global obstacles
    while True:
        new_obstacles, quit_flag = detector.get_obstacles(current_location, show_frame=True)
        obstacles = new_obstacles

        # Firebase update
        try:
            db.child("obstacles").set(obstacles)
        except:
            pass

        # SMS alert
        for obs in obstacles:
            dist = haversine_distance(current_location["lat"], current_location["lon"],
                                      obs["lat"], obs["lon"])
            if dist < 0.5:
                try:
                    send_sms_alert(f"Obstacle Alert: {obs['class']} ({dist*1000:.0f} m ahead)")
                except:
                    pass

        if quit_flag:
            print("Quit detected. Exiting thread.")
            break
        time.sleep(0.5)

# Flask routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/location")
def location_api():
    return jsonify(current_location)

@app.route("/obstacles")
def obstacles_api():
    return jsonify(obstacles)

# Main
if __name__ == "__main__":
    detector = ObstacleDetector("best.pt")

    threading.Thread(target=update_location, daemon=True).start()
    threading.Thread(target=detect_obstacles_thread, args=(detector,), daemon=True).start()

    try:
        app.run(debug=True)
    finally:
        detector.release()
        print("Camera released")