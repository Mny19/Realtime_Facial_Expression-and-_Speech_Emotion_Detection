# main.py
from flask import Flask, render_template, Response, request, send_file
from camera import VideoCamera, record_audio, audio_file_tone_emotion_detection
import os

app = Flask(__name__)
video_camera = None

@app.route('/')
def index():
    return render_template('index.html')

def video_stream():
    global video_camera
    if video_camera is None:
        video_camera = VideoCamera()
    while True:
        frame = video_camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/record_audio', methods=['POST'])
def start_audio_recording():
    record_audio()
    return 'Audio recording complete.'

@app.route('/analyze_emotion', methods=['POST'])
def analyze_emotion():
    audio_file_path = 'output10.wav'
    audio_file_tone_emotion_detection(audio_file_path)
    return 'Emotion analysis complete.'

@app.route('/get_waveform')
def get_waveform():
    return send_file('waveform.png', mimetype='image/png')

@app.route('/get_emotion_graph')
def get_emotion_graph():
    return send_file('emotion_graph.png', mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
