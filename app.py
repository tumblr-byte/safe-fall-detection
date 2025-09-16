import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
import time
import base64

def get_download_link(file_path, filename):
    """Generate download link for processed video"""
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:video/mp4;base64,{b64}" download="{filename}">Download Processed Video</a>'
    return href

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

def process_video(input_path, model_path, output_path, progress_bar, status_text):
    """Process video with fall detection"""
    try:
        # Load model
        status_text.text("Loading YOLO model...")
        model = YOLO(model_path)
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
                    if elapsed >= 5:  # 5 seconds threshold
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(frame, "ALERT! Fall detected for 5s",
                                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 0, 255), 2)
                    else:
                        remaining = int(5 - elapsed)
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
    st.title("Fall Detection Video Processor")
    st.write("Upload a video file and YOLO model to detect falls and generate alerts.")
    
    # File upload sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Video File")
        video_file = st.file_uploader("Choose video file", type=['mp4', 'avi', 'mov', 'mkv'])
    
    with col2:
        st.subheader("Upload YOLO Model")
        model_file = st.file_uploader("Choose YOLO model file (.pt)", type=['pt'])
    
    if video_file is not None and model_file is not None:
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
            tmp_video.write(video_file.read())
            video_path = tmp_video.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_model:
            tmp_model.write(model_file.read())
            model_path = tmp_model.name
        
        # Output file path
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        
        st.write("Files uploaded successfully!")
        
        # Process button
        if st.button("Process Video"):
            # Progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process video
            success = process_video(video_path, model_path, output_path, progress_bar, status_text)
            
            if success:
                st.success("Video processing completed!")
                
                # Download link
                st.markdown("### Download Processed Video")
                download_filename = f"processed_{video_file.name}"
                download_link = get_download_link(output_path, download_filename)
                st.markdown(download_link, unsafe_allow_html=True)
            else:
                st.error("Video processing failed. Please check your files and try again.")
        
        # Cleanup temporary files on app restart
        try:
            os.unlink(video_path)
            os.unlink(model_path)
        except:
            pass
    
    elif video_file is None and model_file is None:
        st.info("Please upload both a video file and a YOLO model file to begin processing.")
    elif video_file is None:
        st.warning("Please upload a video file.")
    else:
        st.warning("Please upload a YOLO model file (.pt).")
    
    # Instructions
    st.markdown("---")
    st.subheader("Instructions")
    st.write("1. Upload your video file (supported formats: MP4, AVI, MOV, MKV)")
    st.write("2. Upload your trained YOLO model file (.pt format)")
    st.write("3. Click 'Process Video' to start fall detection")
    st.write("4. Download the processed video with fall detection annotations")
    
    st.markdown("---")
    st.subheader("Fall Detection Logic")
    st.write("- Green box: Walking/Sitting detected")
    st.write("- Yellow box: Fall detected (counting down to alert)")
    st.write("- Red box: Fall alert triggered (person down for 10+ seconds)")

if __name__ == "__main__":

    main()
