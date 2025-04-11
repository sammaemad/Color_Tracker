import streamlit as st
import cv2
import numpy as np
import tempfile

st.title("Color Object Tracker")
st.markdown("Upload a video and choose a color range to track objects of that color.")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
# Color ranges
COLOR_RANGES = {
    "None": [],
    "Red": [([0, 120, 70], [10, 255, 255]), ([170, 120, 70], [180, 255, 255])],
    "Green": [([36, 25, 25], [86, 255, 255])],
    "Blue": [([94, 80, 2], [126, 255, 255])],
    "Yellow": [([15, 100, 100], [35, 255, 255])],
    "Orange": [([10, 100, 20], [25, 255, 255])],
    "Purple": [([129, 50, 70], [158, 255, 255])],
    "Pink": [([160, 50, 70], [170, 255, 255])]
}

color_choice = st.sidebar.selectbox("Select a Color to Track", list(COLOR_RANGES.keys()))

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        output = frame.copy()

        if color_choice != "None":
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            masks = []

            # Handle single or multiple HSV ranges
            for lower, upper in COLOR_RANGES[color_choice]:
                lower_np = np.array(lower)
                upper_np = np.array(upper)
                masks.append(cv2.inRange(hsv, lower_np, upper_np))

            mask = masks[0]
            for additional in masks[1:]:
                mask = cv2.bitwise_or(mask, additional)

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 800:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(output, color_choice, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        stframe.image(output, channels="BGR")

    cap.release()