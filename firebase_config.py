# firebase_config.py
# firebase_config.py
import firebase_admin
from firebase_admin import credentials, db

# Initialize Firebase Admin SDK
cred = credentials.Certificate("")
firebase_admin.initialize_app(cred, {
    "databaseURL": ""
})

# Root database reference
database = db.reference()

# ----------------------------
# Define separate "sub-databases" (nodes)
# ----------------------------
loco_db = database.child("loco")                    # Live loco location
obstacles_db = database.child("obstacles")          # Detected obstacles
anpr_db = database.child("anpr")                    # Automatic Number Plate Recognition data
engine_fault_db = database.child("engine_fault")    # Engine status/fault info
drowsiness_db = database.child("drowsiness")        # Locopilot drowsiness/yawn alerts
object_detection_db = database.child("object_detection")  # Object detection at track
alerts_db = database.child("alerts")    