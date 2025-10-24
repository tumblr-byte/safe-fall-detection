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
import base64
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Fall Detection Emergency System",
    page_icon="🚨",
    layout="wide"
)


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

USER_LOCATION = {
    "address": "45 Residential Complex, Sector 12, Ghaziabad",
    "lat": 28.6139,
    "lng": 77.2090,
    "phone": "+91-98765-43210"
}


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

def save_fall_snapshot(frame):
    try:
        if frame is None or frame.size == 0:
            return None
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        success, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        if success:
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            print(f"Snapshot saved in memory ({len(jpg_as_text)} bytes)")
            return jpg_as_text
        return None
    except Exception as e:
        print(f"Error saving snapshot: {e}")
        return None

def create_emergency_alert(fall_duration, snapshot_base64):
    alert = {
        "id": len(st.session_state.emergency_alerts) + 1,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "location": USER_LOCATION,
        "fall_duration": fall_duration,
        "snapshot_data": snapshot_base64,
        "status": "CRITICAL",
        "hospitals_notified": len(HOSPITALS)
    }
    st.session_state.emergency_alerts.insert(0, alert)
    return alert

def detect_action(frame, model):
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
                        
                        if elapsed_seconds >= 10 and not alert_triggered:
                            snapshot_frame = frame.copy()
                            cv2.rectangle(snapshot_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.putText(snapshot_frame, "FALL DETECTED!", 
                                      (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.9, (0, 0, 255), 2)
                            
                            snapshot_data = save_fall_snapshot(snapshot_frame)
                            
                            if snapshot_data:
                                st.session_state.fall_snapshot = snapshot_data
                                create_emergency_alert(elapsed_seconds, snapshot_data)
                                alert_triggered = True
                                st.session_state.alert_sent = True
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.putText(frame, "EMERGENCY ALERT SENT!",
                                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.8, (0, 0, 255), 2)
                        elif elapsed_seconds >= 10:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.putText(frame, f"ALERT ACTIVE - Fall: {int(elapsed_seconds)}s",
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
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            out.write(frame)
        
        cap.release()
        out.release()
        status_text.text("Processing completed successfully!")
        return True
        
    except Exception as e:
        status_text.text(f"Error during processing: {str(e)}")
        return False

def user_upload_view():
    st.title("👤 User - Upload Video for Fall Detection")
    
    st.subheader("Upload Video File")
    video_file = st.file_uploader("Choose video file", type=['mp4', 'avi', 'mov', 'mkv'])
    
    if video_file is not None:
        st.write("Video uploaded successfully!")
        
        st.markdown("""
            <style>
            video {
                width: 200px !important;
                height: 200px !important;
                object-fit: cover;
            }
            </style>
        """, unsafe_allow_html=True)
        st.video(video_file)
        
        if st.button("Process Video with Fall Detection"):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
                video_file.seek(0)
                tmp_video.write(video_file.read())
                video_path = tmp_video.name
            
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
                status_text.success("Processing completed!")
                
                if st.session_state.alert_sent:
                    st.balloons()
                    st.error("EMERGENCY DETECTED! Check Family Dashboard to see alert details!")
                
                st.rerun()
            
            try:
                os.unlink(video_path)
            except:
                pass
    else:
        st.info("Please upload a video file to begin monitoring.")
    

    if st.session_state.processing_complete and st.session_state.processed_video_path:
        if os.path.exists(st.session_state.processed_video_path):
            st.success("Video processing completed!")
            
            with open(st.session_state.processed_video_path, 'rb') as file:
                st.download_button(
                    label="Download Processed Video",
                    data=file.read(),
                    file_name=st.session_state.processed_video_name,
                    mime="video/mp4"
                )
            
            if st.button("Process Another Video"):
                try:
                    os.unlink(st.session_state.processed_video_path)
                except:
                    pass
                st.session_state.processing_complete = False
                st.session_state.processed_video_path = None
                st.session_state.processed_video_name = None
                st.session_state.alert_sent = False
                st.rerun()

def family_dashboard():
    st.title("👨‍👩‍👧 Family Dashboard - Emergency Monitoring")
    
    if st.session_state.alert_sent and st.session_state.emergency_alerts:
        latest_alert = st.session_state.emergency_alerts[0]
        st.error(f" **EMERGENCY ALERT ACTIVE** - Alert sent at {latest_alert['timestamp']}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Patient Information")
            st.write(f"**Location:** {USER_LOCATION['address']}")
            st.write(f"**Contact:** {USER_LOCATION['phone']}")
            st.write(f"**Fall Duration:** {latest_alert['fall_duration']:.1f} seconds")
            st.write(f"**Alert Time:** {latest_alert['timestamp']}")
            
            # Show fall image
            st.markdown("### Fall Detection Image")
            snapshot_data = latest_alert.get('snapshot_data')
            if snapshot_data:
                try:
                    img_bytes = base64.b64decode(snapshot_data)
                    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if img is not None:
                        st.image(img, caption="Fall Detection Snapshot", use_column_width=True)
                except:
                    st.info(" Image loading...")
        
        with col2:
            st.markdown("### 🏥 Hospitals Notified")
            st.success(f" Alert sent to {len(HOSPITALS)} nearby hospitals")
            
            for idx, hospital in enumerate(HOSPITALS, 1):
                with st.container():
                    st.markdown(f"**{idx}. {hospital['name']}**")
                    st.write(f"📍 {hospital['address']}")
                    st.write(f"📞 {hospital['phone']}")
                    st.write(f"🚑 Distance: {hospital['distance']}")
                    st.markdown("---")
            
            st.info("Whichever hospital responds first will send help!")
    
    else:
        st.success("No emergency alerts. System monitoring...")
        st.info("When a fall is detected for more than 10 seconds, an alert will be sent to nearby hospitals automatically.")
    
    if st.button("Refresh Dashboard"):
        st.rerun()

def hospital_view():
    st.title("🏥 Hospital Emergency Response Center")
    
    # Hospital selector (simulating which hospital is logged in)
    st.sidebar.markdown("### 🏥 Logged in as:")
    selected_hospital = st.sidebar.selectbox(
        "Select Hospital",
        [h['name'] for h in HOSPITALS],
        index=0
    )
    
    hospital_info = next(h for h in HOSPITALS if h['name'] == selected_hospital)
    st.sidebar.success(f"**{hospital_info['name']}**")
    st.sidebar.write(f"📍 {hospital_info['address']}")
    st.sidebar.write(f"📞 {hospital_info['phone']}")
    
    # Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Active Alerts", len([a for a in st.session_state.emergency_alerts if a['status'] == 'CRITICAL']))
    with col2:
        st.metric("Today's Alerts", len(st.session_state.emergency_alerts))
    with col3:
        st.metric("Avg Response", "2.3 min")
    
    st.markdown("---")
    
    # Show alerts
    if st.session_state.emergency_alerts:
        st.subheader("Emergency Alerts")
        
        for alert in st.session_state.emergency_alerts:
            with st.expander(f"ALERT #{alert['id']} - {alert['timestamp']} - {alert['status']}", expanded=True):
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("### 📍 Patient Emergency Details")
                    st.write(f"**Address:** {alert['location']['address']}")
                    st.write(f"**Contact:** {alert['location']['phone']}")
                    st.write(f"**Coordinates:** {alert['location']['lat']}, {alert['location']['lng']}")
                    st.write(f"**Fall Duration:** {alert['fall_duration']:.1f} seconds")
                    st.write(f"**Alert Time:** {alert['timestamp']}")
                    
                    # Calculate distance from this hospital
                    st.write(f"**Distance from your hospital:** {hospital_info['distance']}")
                    
                    # Google Maps navigation
                    maps_url = f"https://www.google.com/maps/dir/?api=1&origin={hospital_info['lat']},{hospital_info['lng']}&destination={alert['location']['lat']},{alert['location']['lng']}"
                    
                    st.markdown(f"""
                        <a href="{maps_url}" target="_blank">
                            <button style="
                                background-color: #FF4B4B;
                                color: white;
                                padding: 15px 32px;
                                font-size: 18px;
                                border: none;
                                border-radius: 8px;
                                cursor: pointer;
                                margin: 10px 0;
                            ">
                                🚑 GET DIRECTIONS & DISPATCH AMBULANCE
                            </button>
                        </a>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### 📸 Fall Detection Image")
                    snapshot_data = alert.get('snapshot_data')
                    
                    if snapshot_data:
                        try:
                            img_bytes = base64.b64decode(snapshot_data)
                            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                            
                            if img is not None:
                                st.image(img, caption="Fall Detection Snapshot", width=400)
                            else:
                                st.warning("Image loading...")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                    else:
                        st.info("📸 No image available")
                
                st.markdown("---")
                st.info("**Note:** This alert was also sent to other nearby hospitals. Whichever hospital responds first should dispatch help.")
    
    else:
        st.success("No active emergency alerts. System monitoring...")
    
    if st.button("Refresh Alerts"):
        st.rerun()

def main():
    # Sidebar navigation - 3 VIEWS
    st.sidebar.title("Select View")
    page = st.sidebar.radio(
        "Choose Dashboard:",
        ["👤 User - Upload Video", "👨‍👩‍👧 Family Dashboard", "🏥 Hospital Emergency Center"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Status")
    st.sidebar.info(f"Active Alerts: {len(st.session_state.emergency_alerts)}")
    st.sidebar.info(f"Hospitals: {len(HOSPITALS)}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About System")
    st.sidebar.write("**Fall Detection Emergency Response**")
    st.sidebar.write("Automatic hospital alerts when fall >10s detected")
    
    # Route to correct view
    if page == "👤 User - Upload Video":
        user_upload_view()
    elif page == "👨‍👩‍👧 Family Dashboard":
        family_dashboard()
    else:
        hospital_view()

if __name__ == "__main__":
    main()

