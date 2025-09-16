# Fall Detection and Alert System

##  Project Overview
This project uses a **YOLO-based computer vision model** to detect falls from video input and trigger alerts if a person remains on the ground for more than **5 seconds**.  
It is implemented with a **Streamlit app** where users can upload videos and get annotated results.

---

##  What It Does
- Detects human activities: **Walking, Sitting, Fall Detected**  
- Starts a **timer when a fall is detected**  
- If the person recovers before 5 seconds → no alert  
- If the person stays down **≥ 5 seconds** → **alert is triggered**  
- Outputs a processed video with **colored bounding boxes**:  
  - Green → Walking / Sitting  
  - Yellow → Fall detected, countdown running  
  - Red → Fall alert triggered  

---

## Who It Helps
- **Elderly people living alone**  
- **Patients in hospitals**  
- **Caregivers** who need real-time awareness of dangerous falls  
- **Underserved communities** without access to expensive fall-detection hardware  

---

##  Why It Matters
Falls are one of the **leading causes of injury among elderly populations**, especially those living independently.  
Early detection can **save lives** by ensuring help arrives quickly.  

- WHO reports that **37.3 million falls** annually are severe enough to require medical attention.  
- Rapid alerts reduce risk of long-term complications.  

---

## 📊 Measurable Impact
- **Accuracy**: Our YOLO model was trained on activity datasets (Walking, Sitting, Falling).  
- **Impact metric**: If a fall lasts **> 5 seconds**, it is detected and flagged.  
- **Practical demo**: Uploaded video is annotated and downloadable with fall alerts marked.  

---
