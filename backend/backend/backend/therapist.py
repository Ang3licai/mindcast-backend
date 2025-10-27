from flask import Flask, request, render_template
import smtplib
from email.mime.text import MIMEText

app = Flask(__name__)

@app.route('/diary', methods=['GET', 'POST'])
def diary():
    if request.method == 'POST':
        entry = request.form['entry']
        stress_level = request.form['stress_level']
        anonymized = f"Stress Level: {stress_level}\nEntry: {entry}"
        send_to_therapist(anonymized)
        return "Entry forwarded anonymously. Thank you for sharing."
    return render_template('diary.html')

def send_to_therapist(content):
    msg = MIMEText(content)
    msg['Subject'] = 'Anonymous Diary Entry'
    msg['From'] = 'noreply@stressapp.com'
    msg['To'] = 'therapist@example.com'
    with smtplib.SMTP('smtp.example.com', 587) as server:
        server.starttls()
        server.login('your_email', 'your_password')
        server.send_message(msg)
