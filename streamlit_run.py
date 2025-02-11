import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import threading
import random
from datetime import datetime
import os
import time
import pygame  # Import pygame for sound

class ObjectMonitoringApp:
    def __init__(self):
        self.models = {}
        self.current_model = None
        self.cap = None
        self.class_colors = {}
        self.restricted_area = None
        self.csv_file = "data/detection_log.csv"
        self.object_entry_times = {}

        # Alert system
        self.alert_active = False  # Track if an alert is currently playing
        self.alert_thread = None
        self.alert_classes = []  # Store the alert-triggering classes

        # Initialize CSV file if not exists
        if not os.path.exists(self.csv_file):
            pd.DataFrame(columns=["Timestamp", "Class", "Confidence", "Restricted Area Violation"]).to_csv(self.csv_file, index=False)

        # Initialize pygame for sound
        pygame.mixer.init()

    def load_models(self, model_paths):
        """Load multiple YOLO models."""
        for model_name, path in model_paths.items():
            self.models[model_name] = YOLO(path)
        self.current_model = self.models["Intrusion"]

    def generate_class_colors(self, model):
        """Generate a unique random color for each class in the given model."""
        colors = {model.names[class_id]: tuple(random.randint(0, 255) for _ in range(3)) for class_id in model.names}
        return colors

    def start_webcam(self):
        """Start the webcam."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            st.error("Error: Unable to access the webcam.")
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return True

    def stop_webcam(self):
        """Stop the webcam."""
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()
            self.cap = None
            self.stop_alert()  # Stop alert when webcam stops

    def play_alert_sound(self, sound_path):
        """Play alert sound in a loop while an object is inside the restricted area."""
        pygame.mixer.music.load(sound_path)
        pygame.mixer.music.play(-1)  # Loop indefinitely

        while self.alert_active:
            time.sleep(1)

        pygame.mixer.music.stop()  # Stop when no object is in the restricted area

    def start_alert(self, sound_path):
        """Start playing alert sound in a separate thread if not already playing."""
        if not self.alert_active:
            self.alert_active = True
            self.alert_thread = threading.Thread(target=self.play_alert_sound, args=(sound_path,), daemon=True)
            self.alert_thread.start()

    def stop_alert(self):
        """Stop the alert sound."""
        if self.alert_active:
            self.alert_active = False  # This stops the play_alert_sound loop

    def draw_roi(self, frame):
        """Draw an elliptical restricted area on the frame."""
        if self.current_model == self.models["Intrusion"]:
            height, width, _ = frame.shape
            center = (width // 2, height // 2)
            axes = (width // 4, height // 8)
            self.restricted_area = (center, axes)
            cv2.ellipse(frame, center, axes, 0, 0, 360, (0, 0, 255), 2)
        return frame

    def is_near_restricted_area(self, box):
        """Check if the detected object is inside the restricted area."""
        if self.restricted_area:
            center, axes = self.restricted_area
            x1, y1, x2, y2 = box
            object_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            distance = np.linalg.norm(np.array(center) - np.array(object_center))
            return distance < (min(axes) + 50)
        return False

    def save_detection_data(self, class_name, confidence, near_restricted_area):
        """Save detection data to CSV if a restricted area violation occurs."""
        if near_restricted_area:
            data = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Class": class_name,
                "Confidence": confidence,
                "Restricted Area Violation": "Yes",
            }
            df = pd.DataFrame([data])
            df.to_csv(self.csv_file, mode='a', header=False, index=False)

    def update_frame(self, model, confidence_threshold, selected_classes, alert_classes):
        """Process each frame for object detection and alerts."""
        if not self.cap:
            return None, []

        ret, frame = self.cap.read()
        if not ret:
            return None, []

        results = model(frame, conf=confidence_threshold, iou=0.3)
        detected_classes = []
        annotated_frame = frame.copy()
        object_inside_restricted_area = False  

        for result in results[0].boxes:
            class_id = int(result.cls)
            class_name = model.names[class_id]

            if class_name in selected_classes:
                detected_classes.append(class_name)
                color = self.class_colors.get(class_name, (0, 255, 0))
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                conf = result.conf[0]
                label = f"{class_name} {conf:.2f}"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if self.is_near_restricted_area([x1, y1, x2, y2]):
                    object_inside_restricted_area = True
                    cv2.putText(annotated_frame, "Object in Restricted Area!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                    if class_name not in self.object_entry_times:
                        self.object_entry_times[class_name] = time.time()

                    if time.time() - self.object_entry_times[class_name] > 2: #This is time threshold for saving data to csv
                        self.save_detection_data(class_name, float(conf), object_inside_restricted_area)
                        self.object_entry_times[class_name] = time.time()

        if object_inside_restricted_area and any(cls in alert_classes for cls in detected_classes):
            self.start_alert("alert.mp3")  # Start alert
        else:
            self.stop_alert()  # Stop alert if no relevant objects are detected

        annotated_frame = self.draw_roi(annotated_frame)

        return annotated_frame, detected_classes

    def run(self):
        st.set_page_config(page_title="Real-Time Object Monitoring System", layout="wide")
        # Title for the app
        st.markdown("<h2 style='text-align: center;'>🔍 Real-Time Intrusion Detection & Restricted Area Monitoring</h2>", unsafe_allow_html=True)

        st.sidebar.title("🔧 Settings")
        model_paths = {"Intrusion": "model/yolov8n.pt"}
        selected_model = st.sidebar.selectbox("Select Model", options=model_paths.keys())

        if self.current_model != self.models.get(selected_model):
            self.current_model = self.models[selected_model]
            self.class_colors = self.generate_class_colors(self.current_model)

        confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05)
        available_classes = list(self.current_model.names.values())
        selected_classes = st.sidebar.multiselect("Objects to Detect", available_classes, default=[])
        self.alert_classes = st.sidebar.multiselect("Objects for Alert Sound", available_classes, default=[])

        if st.sidebar.button("▶️ Start Webcam"):
            if self.start_webcam():
                st.success("Webcam started successfully!")

        if st.sidebar.button("⏹️ Stop Webcam"):
            self.stop_webcam()
            st.success("Webcam stopped.")

        if self.cap:
            frame_placeholder = st.empty()
            while self.cap.isOpened():
                result = self.update_frame(self.current_model, confidence_threshold, selected_classes, self.alert_classes)
                if result:
                    frame, _ = result
                    if frame is not None:
                        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

if __name__ == "__main__":
    app = ObjectMonitoringApp()
    app.load_models({"Intrusion": "model/yolov8n.pt"})
    app.run()