# email_alert.py for Object Detection
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
SMTP_SERVER = ""
SMTP_PORT = 587
EMAIL_ADDRESS = ""          
EMAIL_PASSWORD = ""            
TO_EMAIL_ADDRESS = "" 
def send_email_alert(subject: str, message: str):
    """
    Send an email alert via SMTP whenever an obstacle is detected.
    """
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = TO_EMAIL_ADDRESS
        msg['Subject'] = subject
        msg.attach(MIMEText(message, 'plain'))
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()

        print(f"[{time.strftime('%H:%M:%S')}] Email sent: {subject}")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Failed to send email: {e}")

if __name__ == "__main__":
    send_email_alert(
        "Object Detection Test Alert",
        "This is a test alert from the object detection module."
    )
