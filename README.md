
---

# 🚀 **Real-Time Restricted Area Monitoring System with Yolo**

This project integrates **FastAPI** and **Streamlit** to create a **real-time monitoring dashboard** for object detection logs stored in a CSV file.

- **FastAPI** handles real-time WebSocket communication.
- **Streamlit** provides an interactive and dynamic UI for live visualization.

---

## 🎯 **Key Features**

- ✅ **Real-time Object Detection** with Streamlit for continuous video streaming.
- ✅ **Live Data Dashboard** powered by FastAPI.
- ✅ **Automatic Data Refresh** every second.
- ✅ **Interactive Analytics & Insights** with Pandas.
- ✅ **CSV-based Logging** for easy data handling and monitoring.

---

## 🛠️ **Installation & Setup**

Follow these steps to set up the project:

### 1️⃣ **Install Dependencies**

Ensure you have Python installed, then run the following command to install required dependencies:

```bash
pip install -r requirements.txt
```

### 2️⃣ **Start FastAPI Backend**

Run the FastAPI server to serve data via WebSockets:

```bash
uvicorn fastapi_run:app --reload
```

- 🔗 **Server URL:** [http://127.0.0.1:8000](http://127.0.0.1:8000)

### 3️⃣ **Launch Streamlit Dashboard**

Start the Streamlit frontend:

```bash
streamlit run streamlit_run.py
```

- 📊 **Your real-time  Video Streaming dashboard will open in your web browser!**

---

## 🔍 **How It Works**
### 🚀 **FastAPI WebSocket Backend**
![Real Time Res (1)](https://github.com/user-attachments/assets/bf2c4ac3-7e4c-4f84-abb0-06514eaefafe)
- **Data Handling**: Reads live data from `detection_log.csv`.
- **Real-Time Data**: Uses WebSockets to push real-time detection logs to connected clients without requiring page refreshes.
- **Live Data Monitoring**: 
  - Displays updates on object detection and violations.
  - Real-time updates of detection and violation counts on the dashboard.

#### **Summary Cards**
- **Total Detections**: Displays the total number of detected objects in real time.
- **Total Violations 🚨**: Displays the number of violations detected (e.g., unauthorized access).
- **Most Frequent Class**: Shows the most frequently detected object class.

#### **Charts**
- **Object with Confidence**: Bar chart visualizing the confidence levels of detected objects.
- **Violation Counts by Time of Day**: Pie chart illustrating violation distribution by time.

#### **Data Table**
- Displays a table with recent detection data: timestamp, object class, confidence percentage, and violation status.
- Shows a message like "Waiting for live data..." when no data is available.

---
### 📊 **Streamlit Frontend**
<img width="954" alt="Capture1234" src="https://github.com/user-attachments/assets/2cdd8f8a-906d-46f7-9954-db8f203430d6" />
- **Real-Time Intrusion Detection**: Detects objects using YOLO and triggers alerts when specific objects enter a restricted area.
- **Restricted Area**: Defines a central restricted area to monitor violations.
- **Sound Alerts**: Triggers sound alerts when an object enters the restricted area.
- **Detection Logging**: Logs detection data to a CSV file when a restricted area violation occurs.

---

## 🎬 **Live Demo & Usage**

Once both servers are running, open the **Streamlit dashboard** in your browser to start monitoring real-time detections and insights! 🚀

- 📡 **Monitor, analyze, and visualize data live!**

---

## 💡 **Contribute & Connect**

- 💻 **Fork & Customize**: Improve and adapt this project for your needs.
- 📩 **Feedback & Suggestions**: Always welcome!

🚀 **Made with ❤️ by ApyCoder**

---

