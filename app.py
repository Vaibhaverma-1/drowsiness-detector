import os
import gdown
import streamlit as st

# -----------------------------
# Download Model with Spinner
# -----------------------------
MODEL_PATH = "shape_predictor_68_face_landmarks.dat"
MODEL_URL = "https://drive.google.com/uc?id=1ZLh3RZxngbB-R57x_55Uos36gW7Z6l9L"

if not os.path.exists(MODEL_PATH):
    with st.spinner("🔽 Downloading Dlib face landmark model (this happens only once)..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    st.success("✅ Model downloaded successfully!")

# -----------------------------
# Imports after model download
# -----------------------------
import av
import cv2
import dlib
import imutils
import numpy as np
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# -----------------------------
# Safe audio initialization
# -----------------------------
try:
    mixer.init()
    mixer.music.load("music.wav")
    AUDIO_ENABLED = True
except Exception as e:
    st.warning("⚠️ Audio alerts disabled (no sound device in cloud).")
    print("Audio init failed:", e)
    AUDIO_ENABLED = False

# -----------------------------
# Drowsiness Detection Config
# -----------------------------
THRESH = 0.25
FRAME_CHECK = 20

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor(MODEL_PATH)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("😴 Real-Time Drowsiness Detection System")
st.markdown("Close your eyes for a few seconds to trigger the alert 🚨")

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

class DrowsinessDetector(VideoProcessorBase):
    def __init__(self):
        self.flag = 0

    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        rects = detect(gray, 0)
        for rect in rects:
            shape = predict(gray, rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frm, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frm, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < THRESH:
                self.flag += 1
                if self.flag >= FRAME_CHECK:
                    cv2.putText(frm, "DROWSINESS ALERT!!!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if AUDIO_ENABLED and not mixer.music.get_busy():
                        mixer.music.play()
            else:
                self.flag = 0
                if AUDIO_ENABLED:
                    mixer.music.stop()

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

webrtc_streamer(
    key="drowsy",
    video_processor_factory=DrowsinessDetector,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False}
)
