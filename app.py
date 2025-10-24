import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'

import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
import time
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="Fall Detection Emergency System",
    page_icon="🚨",
    layout="wide"
)

# Initialize session state
if 'processed_video_path' not in st.session_state:
    st.session_state.processed_video_path = None
if 'processed_video_name' not in st.session_state:
    st.session_state.processed_video_name = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'emergency_alerts' not in st.session_state:
    st.session_state.emergency_alerts = []
if 'fall_snapshot' not in st.session_state:
    st.session_state.fall_snapshot = None
if 'alert_sent' not in st.session_state:
    st.session_state.alert_sent = False

# Fake hospital database with coordinates
HOSPITALS = [
    {
        "name": "City General Hospital",
        "address": "123 Medical Center Dr, Downtown",
        "lat": 28.6139,
        "lng": 77.2090,
        "distance": "2.3 km",
        "phone": "+91-11-2345-6789"
    },
    {
        "name": "Emergency Care Center",
        "address": "456 Healthcare Ave, Central District",
        "lat": 28.6180,
        "lng": 77.2150,
        "distance": "3.1 km",
        "phone": "+91-11-2345-6790"
    },
    {
        "name": "Metro Medical Hospital",
        "address": "789 Wellness Blvd, Medical District",
        "lat": 28.6100,
        "lng": 77.2050,
        "distance": "4.2 km",
        "phone": "+91-11-2345-6791"
    }
]

# Fake user location (can be customized)
USER_LOCATION = {
    "address": "45 Residential Complex, Sector 12, Ghaziabad",
    "lat": 28.6139,
    "lng": 77.2090
}

def save_fall_snapshot(frame):
    """Save fall detection snapshot"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    cv2.imwrite(temp_file.name, frame)
    return temp_file.name

def create_emergency_alert(fall_duration, snapshot_path):
    """Create emergency alert when fall detected"""
    alert = {
        "id": len(st.session_state.emergency_alerts) + 1,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "location": USER_LOCATION,
        "fall_duration": fall_duration,
        "snapshot_path": snapshot_path,
        "status": "CRITICAL",
        "hospitals_notified": len(HOSPITALS)
    }
    st.session_state.emergency_alerts.insert(0, alert)
    return alert

def detect_action(frame, model):
    """Detect action in frame"""
    results = model(frame)
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            cls_id = box.cls[0]
            conf = box.conf[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            return cls_id, conf, x1, y1, x2, y2, frame
    return None, None, None, None, None, None, None

def process_video(input_path, output_path, progress_bar, status_text):
    """Process video with fall detection and emergency alerts"""
    try:
        status_text.text("Loading YOLO model...")
        model = YOLO("best.pt")
        model.fuse()
        classes = ["Fall Detected", "Walking", "Sitting"]
        
        status_text.text("Opening video file...")
        cap = cv2.VideoCapture(input_path)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        process_width = min(640, width)
        process_height = int((process_width / width) * height)
        scale_x = width / process_width
        scale_y = height / process_height
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        fall_start_time = None
        fall_frame_count = 0
        frame_count = 0
        skip_frames = 2
        alert_triggered = False
        
        status_text.text(f"Processing video frames... ({total_frames} total)")
        
        last_detection = None
        detection_confidence = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if frame_count % 10 == 0:
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_count}/{total_frames}")
            
            if frame_count % skip_frames == 0:
                small_frame = cv2.resize(frame, (process_width, process_height))
                cls_id, conf, x1, y1, x2, y2, detected_frame = detect_action(small_frame, model)
                
                if cls_id is not None:
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    
                    last_detection = (cls_id, conf, x1, y1, x2, y2, frame.copy())
                    detection_confidence = conf
                else:
                    detection_confidence *= 0.9
                    if detection_confidence < 0.5:
                        last_detection = None
            
            if last_detection is not None and detection_confidence > 0.5:
                cls_id, conf, x1, y1, x2, y2, detected_frame = last_detection
                label = classes[int(cls_id)]
                
                if label == "Fall Detected":
                    if fall_start_time is None:
                        fall_start_time = frame_count / fps
                        fall_frame_count = 0
                    else:
                        fall_frame_count += 1
                        elapsed_seconds = fall_frame_count / fps
                        
                        # Trigger alert at 10 seconds
                        if elapsed_seconds >= 10 and not alert_triggered:
                            snapshot_path = save_fall_snapshot(detected_frame)
                            st.session_state.fall_snapshot = snapshot_path
                            create_emergency_alert(elapsed_seconds, snapshot_path)
                            alert_triggered = True
                            st.session_state.alert_sent = True
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.putText(frame, "🚨 EMERGENCY ALERT SENT! 🚨",
                                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.8, (0, 0, 255), 2)
                            cv2.putText(frame, f"Hospitals notified - Fall: {int(elapsed_seconds)}s",
                                        (50, 90), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6, (0, 0, 255), 2)
                        elif elapsed_seconds >= 10:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.putText(frame, f"🚨 ALERT ACTIVE - Fall: {int(elapsed_seconds)}s",
                                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.8, (0, 0, 255), 2)
                        else:
                            remaining = int(10 - elapsed_seconds)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                            cv2.putText(frame, f"Fall detected, alert in {remaining}s",
                                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7, (0, 255, 255), 2)
                else:
                    fall_start_time = None
                    fall_frame_count = 0
                    alert_triggered = False
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label,
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)
            
            out.write(frame)
        
        cap.release()
        out.release()
        
        status_text.text("Processing completed successfully!")
        return True
        
    except Exception as e:
        status_text.text(f"Error during processing: {str(e)}")
        return False

def user_dashboard():
    """User/Caregiver Dashboard"""
    st.title("🎥 Fall Detection Emergency System - User Dashboard")
    
    # Alert status at top
    if st.session_state.alert_sent and st.session_state.emergency_alerts:
        latest_alert = st.session_state.emergency_alerts[0]
        st.error(f"🚨 **EMERGENCY ALERT ACTIVE** - Sent to {latest_alert['hospitals_notified']} hospitals at {latest_alert['timestamp']}")
    
    # Show download section if processing is complete
    if st.session_state.processing_complete and st.session_state.processed_video_path:
        if os.path.exists(st.session_state.processed_video_path):
            st.success("✅ Video processing completed!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📥 Download Processed Video")
                with open(st.session_state.processed_video_path, 'rb') as file:
                    st.download_button(
                        label="📥 Download Processed Video",
                        data=file.read(),
                        file_name=st.session_state.processed_video_name,
                        mime="video/mp4"
                    )
            
            with col2:
                st.markdown("### 🚨 Emergency Status")
                if st.session_state.alert_sent:
                    st.error("**ALERT SENT TO HOSPITALS**")
                    st.info(f"📍 Location: {USER_LOCATION['address']}")
                    st.info(f"🏥 Hospitals Notified: {len(HOSPITALS)}")
                else:
                    st.success("No emergency detected")
            
            if st.button("🔄 Process Another Video"):
                try:
                    os.unlink(st.session_state.processed_video_path)
                except:
                    pass
                st.session_state.processing_complete = False
                st.session_state.processed_video_path = None
                st.session_state.processed_video_name = None
                st.session_state.alert_sent = False
                st.rerun()
    
    # File upload section
    if not st.session_state.processing_complete:
        st.subheader("📤 Upload Video File")
        video_file = st.file_uploader("Choose video file", type=['mp4', 'avi', 'mov', 'mkv'])
        
        if video_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
                tmp_video.write(video_file.read())
                video_path = tmp_video.name
            
            st.write("✅ Video uploaded successfully!")
            
            if st.button("🚀 Process Video with Fall Detection"):
                output_filename = f"processed_{video_file.name.rsplit('.', 1)[0]}.avi"
                output_path = os.path.join(tempfile.gettempdir(), output_filename)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner('Processing video... Monitoring for falls...'):
                    success = process_video(video_path, output_path, progress_bar, status_text)
                
                if success:
                    st.session_state.processed_video_path = output_path
                    st.session_state.processed_video_name = output_filename
                    st.session_state.processing_complete = True
                    
                    progress_bar.progress(1.0)
                    status_text.success("🎉 Processing completed!")
                    
                    if st.session_state.alert_sent:
                        st.balloons()
                        st.warning("⚠️ EMERGENCY ALERT TRIGGERED! Check Hospital Dashboard.")
                    
                    st.rerun()
                else:
                    st.error("❌ Video processing failed.")
            
            try:
                os.unlink(video_path)
            except:
                pass
        else:
            st.info("📤 Please upload a video file to begin monitoring.")

def hospital_dashboard():
    """Hospital Emergency Dashboard"""
    st.title("🏥 Hospital Emergency Response Dashboard")
    
    # Stats row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Active Alerts", len([a for a in st.session_state.emergency_alerts if a['status'] == 'CRITICAL']))
    with col2:
        st.metric("Total Alerts Today", len(st.session_state.emergency_alerts))
    with col3:
        st.metric("Response Time Avg", "2.3 min")
    
    st.markdown("---")
    
    # Emergency alerts
    if st.session_state.emergency_alerts:
        st.subheader("🚨 Emergency Alerts")
        
        for alert in st.session_state.emergency_alerts:
            with st.expander(f"⚠️ ALERT #{alert['id']} - {alert['timestamp']} - {alert['status']}", expanded=True):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("### 📍 Patient Location")
                    st.write(f"**Address:** {alert['location']['address']}")
                    st.write(f"**Coordinates:** {alert['location']['lat']}, {alert['location']['lng']}")
                    st.write(f"**Fall Duration:** {alert['fall_duration']:.1f} seconds")
                    st.write(f"**Alert Time:** {alert['timestamp']}")
                    
                    # Google Maps Link
                    maps_url = f"https://www.google.com/maps/search/?api=1&query={alert['location']['lat']},{alert['location']['lng']}"
                    st.markdown(f"[🗺️ **Open in Google Maps**]({maps_url})")
                    
                    # Fall snapshot
                    if alert['snapshot_path'] and os.path.exists(alert['snapshot_path']):
                        st.markdown("### 📸 Fall Detection Image")
                        st.image(alert['snapshot_path'], caption="Fall Detection Snapshot", use_container_width=True)
                
                with col2:
                    st.markdown("### 🏥 Nearby Hospitals")
                    for hospital in HOSPITALS:
                        with st.container():
                            st.markdown(f"**{hospital['name']}**")
                            st.write(f"📍 {hospital['address']}")
                            st.write(f"📞 {hospital['phone']}")
                            st.write(f"🚗 Distance: {hospital['distance']}")
                            
                            hospital_maps_url = f"https://www.google.com/maps/dir/?api=1&origin={alert['location']['lat']},{alert['location']['lng']}&destination={hospital['lat']},{hospital['lng']}"
                            st.markdown(f"[🚑 **Get Directions**]({hospital_maps_url})")
                            st.markdown("---")
    else:
        st.info("✅ No active emergency alerts. System monitoring...")
    
    # Refresh button
    if st.button("🔄 Refresh Dashboard"):
        st.rerun()

def main():
    # Sidebar navigation
    st.sidebar.title("🚨 Navigation")
    page = st.sidebar.radio("Select View:", ["👤 User Dashboard", "🏥 Hospital Dashboard"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 System Info")
    st.sidebar.info(f"Active Alerts: {len(st.session_state.emergency_alerts)}")
    st.sidebar.info(f"Hospitals Connected: {len(HOSPITALS)}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 About")
    st.sidebar.write("Fall Detection Emergency Response System")
    st.sidebar.write("Automatic hospital notification on fall detection >10s")
    
    # Route to correct dashboard
    if page == "👤 User Dashboard":
        user_dashboard()
    else:
        hospital_dashboard()

if __name__ == "__main__":
    main()
