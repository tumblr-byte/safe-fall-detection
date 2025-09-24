import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'

import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
import time

# Initialize session state
if 'processed_video_path' not in st.session_state:
    st.session_state.processed_video_path = None
if 'processed_video_name' not in st.session_state:
    st.session_state.processed_video_name = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

def detect_action(frame, model):
    """Detect action in frame"""
    results = model(frame)
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            cls_id = box.cls[0]
            conf = box.conf[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            return cls_id, conf, x1, y1, x2, y2
    return None, None, None, None, None, None

def process_video(input_path, output_path, progress_bar, status_text):
    """Process video with fall detection"""
    try:
        # Load model
        status_text.text("Loading YOLO model...")
        model = YOLO("best.pt")
        classes = ["Fall Detected", "Walking", "Sitting"]
        
        # Open video
        status_text.text("Opening video file...")
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        fall_start_time = None
        frame_count = 0
        
        status_text.text("Processing video frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            
            cls_id, conf, x1, y1, x2, y2 = detect_action(frame, model)
            
            if cls_id is None:
                out.write(frame)
                continue
            
            label = classes[int(cls_id)]
            
            if label == "Fall Detected":
                if fall_start_time is None:
                    fall_start_time = time.time()
                else:
                    elapsed = time.time() - fall_start_time
                    if elapsed >= 10:  # 10 seconds threshold
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(frame, "ALERT! Fall detected for 10s",
                                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 0, 255), 2)
                    else:
                        remaining = int(10 - elapsed)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                        cv2.putText(frame, f"Fall detected, alert in {remaining}s",
                                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 255, 255), 2)
            else:
                fall_start_time = None
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 123, 3), 3)
                cv2.putText(frame, label,
                            (x1 - 10, y1),
                            cv2.FONT_HERSHEY_COMPLEX,
                            1, (0, 255, 0), 2)
            
            out.write(frame)
        
        cap.release()
        out.release()
        
        status_text.text("Processing completed successfully!")
        return True
        
    except Exception as e:
        status_text.text(f"Error during processing: {str(e)}")
        return False

def main():
    st.title("🎥 Fall Detection Video Processor")
    st.write("Upload a video file to detect falls and generate alerts.")
    
    # Show download section if processing is complete
    if st.session_state.processing_complete and st.session_state.processed_video_path:
        if os.path.exists(st.session_state.processed_video_path):
            st.success("✅ Video processing completed!")
            st.markdown("### 📥 Download Your Processed Video")
            
            # Simple download button using Streamlit's built-in download
            with open(st.session_state.processed_video_path, 'rb') as file:
                st.download_button(
                    label="📥 Download Processed Video",
                    data=file.read(),
                    file_name=st.session_state.processed_video_name,
                    mime="video/mp4"
                )
            
            # Reset button
            if st.button("🔄 Process Another Video"):
                try:
                    os.unlink(st.session_state.processed_video_path)
                except:
                    pass
                st.session_state.processing_complete = False
                st.session_state.processed_video_path = None
                st.session_state.processed_video_name = None
        else:
            st.error("Processed video file not found. Please process the video again.")
            st.session_state.processing_complete = False
    
    # File upload section
    st.subheader("📤 Upload Video File")
    video_file = st.file_uploader("Choose video file", type=['mp4', 'avi', 'mov', 'mkv'])
    
    if video_file is not None:
        # Create temporary file for video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
            tmp_video.write(video_file.read())
            video_path = tmp_video.name
        
        st.write("✅ Video uploaded successfully!")
        
        # Process button
        if st.button("🚀 Process Video"):
            # Create output file path
            output_dir = tempfile.mkdtemp()
            output_filename = f"processed_{video_file.name}"
            output_path = os.path.join(output_dir, output_filename)
            
            # Progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process video
            success = process_video(video_path, output_path, progress_bar, status_text)
            
            if success:
                # Store in session state
                st.session_state.processed_video_path = output_path
                st.session_state.processed_video_name = output_filename
                st.session_state.processing_complete = True
                # No st.rerun() - let the user see the result naturally
                st.balloons()  # Celebration effect!
            else:
                st.error("❌ Video processing failed. Please check your file and try again.")
        
        # Cleanup temporary upload file
        try:
            os.unlink(video_path)
        except:
            pass
    
    else:
        st.info("📤 Please upload a video file to begin processing.")
    
    # Instructions
    with st.expander("📋 Instructions", expanded=False):
        st.write("1. Upload your video file (supported formats: MP4, AVI, MOV, MKV)")
        st.write("2. Click '🚀 Process Video' to start fall detection")
        st.write("3. Download the processed video with fall detection annotations")
    
    with st.expander("🎯 Fall Detection Logic", expanded=False):
        st.write("- 🟢 Green box: Walking/Sitting detected")
        st.write("- 🟡 Yellow box: Fall detected (counting down to alert)")
        st.write("- 🔴 Red box: Fall alert triggered (person down for 10+ seconds)")

if __name__ == "__main__":
    main()
