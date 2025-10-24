# Fall Detection and Emergency Alert System

## Project Overview
This project leverages **YOLO-based computer vision** to detect falls from video input and trigger emergency alerts if a person remains on the ground for more than **10 seconds**.  

It is implemented as a **Streamlit web app**, allowing users to upload videos or (in future versions) use live webcam feeds for real-time monitoring. The system notifies **family members** and **nearby hospitals**, providing location, contact info, and fall snapshots for faster response.

> An affordable, open-source solution designed to save lives for elderly individuals, patients, and workers in high-risk environments.

---

## Current Features & Functionality
- Detects human activities: **Walking**, **Sitting**, **Fall Detected**
- Starts a **fall timer** automatically
- **Alerts triggered** if the person remains fallen for ≥10 seconds
- Outputs **processed video** with annotated bounding boxes:
  - **Green** → Walking / Sitting
  - **Yellow** → Fall detected, countdown running
  - **Red** → Fall alert triggered
- **Family Dashboard**: Notified instantly with snapshot, location, and fall details
- **Hospital Emergency Center**: Nearby hospitals (3 by default) notified with patient info, location, and fall snapshot
- Optimized for performance by analyzing **every 2nd frame** without delaying alerts
- Base64 snapshot storage allows quick display in dashboards

---

## Demo & Current Limitations
- Users can upload a video for fall detection analysis
- Processed video can be **downloaded** with annotations
- Alerts are simulated for demo purposes:
  - 3 nearby hospitals notified in demo
  - Family dashboard displays fall snapshots and alert info
- **Live webcam monitoring** is possible but currently limited:
  - Streamlit’s live webcam support is not optimized for real-time high FPS
  - Video upload is recommended for smooth demo

---

## Who It Helps
- **Elderly people living alone** → reduces risk of serious injuries
- **Patients in hospitals or care centers** → enables timely intervention
- **Industrial and construction workers** → improves workplace safety
- **Caregivers and healthcare staff** → provides real-time awareness of dangerous falls
- **Communities without access to expensive hardware** → offers a low-cost, open-source solution

---

## Why It Matters
Falls are a leading cause of injury worldwide. Early detection can save lives and reduce recovery time:
- **37.3 million severe falls annually** require medical attention (WHO)
- Rapid alerts improve **response times** for caregivers
- Provides **accessible, low-cost monitoring** for individuals and small facilities

---

## Measurable Impact
- **Accuracy:** YOLO model trained on Walking, Sitting, and Falling datasets
- **Threshold:** Alerts triggered for falls lasting ≥10 seconds (customizable)
- **Demo:** Streamlit app allows video upload and annotated download
- Real-time detection ensures **no missed alerts**, even with frame-skipping optimization

---

## Tech Stack
- **Python**
- **OpenCV**
- **Ultralytics YOLO**
- **Streamlit**
- **NumPy**
- **Base64** for snapshot storage

---

## Current Workflow
1. User uploads a video or uses a live demo feed
2. YOLO detects human activity in frames
3. If a **fall is detected** and persists >10 seconds:
   - Snapshot is captured
   - Emergency alert sent to family dashboard
   - Nearby hospitals notified (demo: 3 hospitals)
4. Dashboards display:
   - Fall snapshot
   - Patient location and contact
   - Fall duration
   - List of hospitals notified
5. Processed video with bounding boxes is available for download

---

## Future Roadmap
- **Multi-Person Detection** – monitor multiple individuals simultaneously
- **Real-Time Notifications** – integrate with SMS, WhatsApp, or email for instant alerts
- **Live Webcam Integration** – support real-time monitoring in care centers or homes
- **Wearable & IoT Integration** – combine with smart devices for health monitoring
- **Cloud & Edge Deployment** – run on Raspberry Pi, Jetson Nano, or cloud for live monitoring
- **Dataset Expansion & Accuracy** – improve detection under diverse lighting, angles, and body types
- **User-Friendly Dashboards** – mobile/web dashboards for caregivers to view live alerts and history
- **Accessibility & Inclusivity** – multilingual alerts, voice notifications for visually impaired users

---

## Impact & Social Value
- Open-source, low-cost, and scalable solution
- Empowers families and caregivers to monitor safety without expensive hardware
- Deployable in homes, hospitals, or workplaces
- Ready-to-use demo for **education, research, and community safety programs**

---

## Contributing
We welcome contributions from the community:
- Fork the repository
- Submit issues or pull requests for improvements
- Add new features, datasets, or optimizations

---

## How to Run Locally
```bash
# Clone the repo
git clone https:/tumblr-byte/github.com/safe-fall-detection.git
cd safe-fall-detection

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

