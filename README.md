# SmartRailShield: AI-Powered Safety and Violation Detection for Railways
# TEAM WORK
 -1.NITHYASRI R
 -2.SHUNMUGAPRIYA S(MYSELF)

## Overview
SmartRailShield is an **AI-powered railway safety system** designed to enhance railway operations by monitoring track safety, driver alertness, and train health in real-time. The system integrates four key modules:

1. **Object Detection at Railway Tracks**  
   - Detect obstacles and unauthorized access on railway tracks using YOLOv8.
2. **Automatic Number Plate Recognition (ANPR)**  
   - Recognize vehicles at level crossings to track violations using YOLO + EasyOCR.
3. **Driver Drowsiness & Yawning Detection**  
   - Monitor loco pilots for drowsiness and yawning using MediaPipe and OpenCV.
4. **Machine Fault Detection**  
   - Predict potential train machine faults using historical static data and ML models.

## Features
- Real-time obstacle detection alerts
- Automatic number plate recognition at level crossings
- Driver monitoring with voice alerts
- Predictive maintenance for trains
- Firebase database integration for storing alerts and records
- SMTP email notifications for critical events

## Tech Stack
  **Frontend:** HTML, CSS, JavaScript  
  **Backend:** Flask (Python)  
  **Database:** Firebase  
  **AI Models:** YOLOv8, EasyOCR, MediaPipe Face detection,  
  **Alerts:** pyttsx3 (voice) / SMTP 

## SETUP INSTRUCTION
## 1. Clone the repository
-- git clone https://github.com/Shunmugapriya-S/SmartRailShield-AI-Powered-Safety-and-Violation-for-Railways.git
-- cd SmartRailShield
## 2. Install Dependencies
-- pip install -r requirements.txt
## 3.Add firebase Credentials
## 4.Execution
    -python anpr.py
    -python drowsiness.py
    -python faultdetection.py
## 5.Run the Flask app
    python app.py
## 6.Open in browser
http://127.0.0.1:5000/
## Future Enhancements
--IoT sensor integration for real-time train data
--Map-based visualization of tracks and detected obstacles
--Control room dashboard for monitoring multiple trains simultaneously
--Mobile app integration for alerts and reports
 
