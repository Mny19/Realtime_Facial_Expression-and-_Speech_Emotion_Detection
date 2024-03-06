# camera.py
import cv2
from model import FacialExpressionModel
import numpy as np
import pyaudio
import wave
import matplotlib.pyplot as plt
import time
import librosa

# Facial Expression Model Initialization
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model_a1.json", "model_weights1.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

# Audio Constants
CHUNK = 1024 
FORMAT = pyaudio.paInt16
CHANNELS = 2 
RATE = 44100 
RECORD_SECONDS = 20
WAVE_OUTPUT_FILENAME = "output10.wav"

# Audio Tone Emotion Detection
def detect_emotion(audio_data):
    # Example simple rule-based approach for tone emotion detection
    audio_feature = np.mean(audio_data)
    if audio_feature > 0.1:
        return [0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Joy
    elif audio_feature < -0.1:
        return [0.9, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Anger
    elif audio_feature < -0.05:
        return [0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]  # Sad
    elif audio_feature > 0.05:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0]  # Surprise
    elif audio_feature < -0.15:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0]  # Fear
    elif audio_feature > 0.2:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0]  # Confident
    else:
        return [0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0]  # Neutral

# Video Camera Class
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()

# Audio Recorder
def record_audio():
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

# Implementation of audio tone emotion detection from file
def audio_file_tone_emotion_detection(file_path, chunk_size=1024):
    try:
        audio_data, rate = librosa.load(file_path, sr=None, mono=True)

        plt.ion()
        fig, ax = plt.subplots(2, 1)
        x = np.arange(0, len(audio_data) / rate, 1 / rate)
        line, = ax[0].plot(x, audio_data)
        ax[0].set_title('Audio Signal')
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('Amplitude')
        ax[0].set_xlim(0, len(audio_data) / rate)
        ax[0].grid(True)

        # Define emotions
        emotions = ['Neutral', 'Joy', 'Anger', 'Sad', 'Analytical', 'Confident', 'Surprise', 'Fear']

        # Bar plot settings
        bar_width = 0.35
        index = np.arange(len(emotions))
        rects = ax[1].bar(index, [0] * len(emotions), bar_width, label='Emotion')
        ax[1].set_title('Detected Emotion')
        ax[1].set_xlabel('Emotion')
        ax[1].set_ylabel('Probability')
        ax[1].set_xticks(index + bar_width / 2)
        ax[1].set_xticklabels(emotions)
        ax[1].set_ylim(0, 1)
        ax[1].grid(True)

        fig.tight_layout()

        print("Starting audio file tone emotion detection...")
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]

            # Perform emotion detection based on the audio data
            emotion_probabilities = detect_emotion(chunk)

            # Update emotion bar plot
            for rect, prob in zip(rects, emotion_probabilities):
                rect.set_height(prob)
            fig.canvas.draw()
            fig.canvas.flush_events()

            # Insert a small delay to simulate real-time processing
            time.sleep(0.1)

    except Exception as e:
        print("Error during audio file tone emotion detection:", e)
