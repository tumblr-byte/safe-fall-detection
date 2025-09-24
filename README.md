# Fall Detection and Alert System  

## Project Overview  
This project uses a **YOLO-based computer vision model** to detect falls from video input and trigger alerts if a person remains on the ground for more than **10 seconds**.  
It is implemented with a **Streamlit app**, where users can upload videos and get annotated results with alerts.  

Here is the link for the live demo: [Live Demo](https://safe-fall-detection.streamlit.app/)  

---

## What It Does  
- Detects human activities: **Walking, Sitting, Fall Detected**  
- Starts a **timer when a fall is detected**  
- If the person recovers **before 10 seconds** → no alert  
- If the person stays down **≥ 10 seconds** → alert is triggered  
- Outputs a processed video with **colored bounding boxes**:  
  - **Green** → Walking / Sitting  
  - **Yellow** → Fall detected, countdown running  
  - **Red** → Fall alert triggered  

---

## Who It Helps  
- **Elderly people living alone** → vulnerable to serious injury from falls  
- **Patients in hospitals or care centers** → need timely detection for quick response  
- **Industrial and construction workers** → at risk of accidents, slips, or sudden medical events (e.g., heart attacks, strokes)  
- **Caregivers and healthcare staff** → gain real-time awareness of dangerous falls, reducing response time  
- **Communities without access to expensive fall-detection hardware** → affordable AI alternative makes safety more inclusive  

---

## Why It Matters  
Falls are one of the **leading causes of injury among elderly populations**, but they also affect workers and people with medical conditions.  
Early detection can save lives by ensuring help arrives quickly.  

- WHO reports that **37.3 million falls annually** are severe enough to require medical attention.  
- Workplace injuries and sudden collapses (heart attack, stroke) often go unnoticed without monitoring.  
- Rapid alerts reduce response time, improving recovery outcomes and survival rates.  

---

## Measurable Impact  
- **Accuracy**: The YOLO model was trained on activity datasets (Walking, Sitting, Falling).  
- **Threshold**: Falls lasting more than **10 seconds** are flagged, but this can be customized depending on the use case.  
- **Practical demo**: The Streamlit app allows users to upload a video and download an annotated version with fall alerts clearly marked.  

---

## Tech Stack  
- Python  
- OpenCV  
- YOLO12 (Ultralytics)  
- Streamlit  

---

---
## Optimization: For faster processing, the system analyzes every 2nd frame.  
The fall alert countdown is based on real-time seconds, so skipping frames does **not delay or miss alerts**.

---
## Future Scope

While the current system is optimized for detecting falls of a **single person** (ideal for elderly individuals living alone), there are several opportunities for expansion:

### 1. Multi-Person Detection
- Extend the system to track and monitor multiple individuals simultaneously in crowded environments such as hospitals, care centers, or workplaces.

### 2. Real-Time Notifications
- Integrate with messaging/alert systems (SMS, WhatsApp, email) to immediately notify caregivers, family members, or emergency services when a fall is detected.

### 3. Wearable & IoT Integration
- Connect with smartwatches, health monitors, or IoT devices to gather additional health data (e.g., heart rate, oxygen levels) alongside fall detection for more reliable alerts.

### 4. Cloud & Edge Deployment
- Deploy the system on low-cost edge devices (Raspberry Pi, Jetson Nano) or cloud platforms for real-time monitoring without requiring expensive hardware.

### 5. Dataset Expansion & Accuracy
- Train on larger, more diverse datasets (different body types, lighting, camera angles) to reduce false positives and improve robustness in real-world conditions.

### 6. User-Friendly Interfaces
- Build mobile and web dashboards for caregivers to monitor status, view alerts, and analyze incident history with visualizations.

### 7. Accessibility & Inclusivity
- Localize alerts into multiple languages and support voice-based notifications for visually impaired users.

