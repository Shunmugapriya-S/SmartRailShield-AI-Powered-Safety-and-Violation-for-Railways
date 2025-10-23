# SmartRailShield: AI-Powered Safety and Violation Detection for Railways

## Team Members
1. Nithyasri R  
2. Shunmugapriya S (Myself)

---

## Overview
**SmartRailShield** is an AI-powered railway safety system designed to enhance railway operations by monitoring track safety, driver alertness, and train health in real-time. The system integrates **four key modules**:

1. **Object Detection at Railway Tracks**  
   - Detect obstacles and unauthorized access on railway tracks using YOLOv8.

2. **Automatic Number Plate Recognition (ANPR)**  
   - Recognize vehicles at level crossings to track violations using YOLO + EasyOCR.

3. **Driver Drowsiness & Yawning Detection**  
   - Monitor loco pilots for drowsiness and yawning using MediaPipe and OpenCV.

4. **Machine Fault Detection**  
   - Predict potential train machine faults using historical static data and ML models.

---

## Features
- Real-time obstacle detection alerts  
- Automatic number plate recognition at level crossings  
- Driver monitoring with voice alerts  
- Predictive maintenance for trains  
- Firebase database integration for storing alerts and records  
- SMTP email notifications for critical events  

---

## Tech Stack
- **Frontend:** HTML, CSS, JavaScript  
- **Backend:** Flask (Python)  
- **Database:** Firebase  
- **AI Models:** YOLOv8, EasyOCR, MediaPipe Face Detection  
- **Alerts:** pyttsx3 (voice) / SMTP  

---

## Setup Instructions
# 1. Clone the repository
```bash
git clone https://github.com/Shunmugapriya-S/SmartRailShield-AI-Powered-Safety-and-Violation-for-Railways.git
cd SmartRailShield

2. Install dependencies
pip install -r requirements.txt


3. Add Firebase credentials

Place your firebase-key.json or configuration file in the project root.


4. Run individual modules
python anpr.py
python drowsiness.py
python faultdetection.py
python detection.py


5. Run the Flask app
python app.py

6. Open in browser
http://127.0.0.1:5000/

Future Enhancements

IoT sensor integration for real-time train data

Map-based visualization of tracks and detected obstacles

Control room dashboard for monitoring multiple trains simultaneously

Mobile app integration for alerts and reports



License

This project is licensed under the MIT License. See LICENSE
 for details.

Author

Shunmugapriya S
Nithyasri R

If any queries contact gmail:
                      sshunmugapriya49@gmail.com
                      nithyasri482006sri@gmail.com
                              
