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
        # Load model once with optimizations
        status_text.text("Loading YOLO model...")
        model = YOLO("best.pt")
        model.fuse()  # Optimize model for inference
        classes = ["Fall Detected", "Walking", "Sitting"]
        
        # Open video
        status_text.text("Opening video file...")
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Resize for faster processing
        process_width = min(640, width)  # Limit processing width
        process_height = int((process_width / width) * height)
        scale_x = width / process_width
        scale_y = height / process_height
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        fall_start_time = None
        fall_frame_count = 0
        frame_count = 0
        
        # Process every 2nd frame (as you requested - not too much skipping)
        skip_frames = 2
        
        status_text.text(f"Processing video frames... ({total_frames} total)")
        
        # Batch processing variables
        last_detection = None
        detection_confidence = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Update progress less frequently (every 10 frames, not 20)
            if frame_count % 10 == 0:
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_count}/{total_frames}")
            
            # Only run detection on selected frames
            if frame_count % skip_frames == 0:
                # Resize frame for detection
                small_frame = cv2.resize(frame, (process_width, process_height))
                
                # Run detection on smaller frame
                cls_id, conf, x1, y1, x2, y2 = detect_action(small_frame, model)
                
                if cls_id is not None:
                    # Scale coordinates back to original size
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    
                    last_detection = (cls_id, conf, x1, y1, x2, y2)
                    detection_confidence = conf
                else:
                    # Decay confidence if no detection
                    detection_confidence *= 0.9  # Less aggressive decay
                    if detection_confidence < 0.5:
                        last_detection = None
            
            # Use last detection for annotation
            if last_detection is not None and detection_confidence > 0.5:
                cls_id, conf, x1, y1, x2, y2 = last_detection
                label = classes[int(cls_id)]
                
                if label == "Fall Detected":
                    if fall_start_time is None:
                        fall_start_time = frame_count / fps
                        fall_frame_count = 0
                    else:
                        fall_frame_count += 1
                        elapsed_seconds = fall_frame_count / fps
                        
                        if elapsed_seconds >= 10:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.putText(frame, "ALERT! Fall detected for 10s",
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
                st.rerun()
        else:
            st.error("Processed video file not found. Please process the video again.")
            st.session_state.processing_complete = False
    
    # File upload section - ONLY show if not processing complete
    if not st.session_state.processing_complete:
        st.subheader("📤 Upload Video File")
        video_file = st.file_uploader("Choose video file", type=['mp4', 'avi', 'mov', 'mkv'])
        
        if video_file is not None:
            # IMPORTANT: Reset processing state when new video is uploaded
            if st.session_state.processing_complete:
                # Clean up old file
                try:
                    if st.session_state.processed_video_path:
                        os.unlink(st.session_state.processed_video_path)
                except:
                    pass
                # Reset state
                st.session_state.processing_complete = False
                st.session_state.processed_video_path = None
                st.session_state.processed_video_name = None
            
            # Create temporary file for video
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
                tmp_video.write(video_file.read())
                video_path = tmp_video.name
            
            st.write("✅ Video uploaded successfully!")
            
            # Process button
            if st.button("🚀 Process Video"):
                # Create output file path with better naming
                output_filename = f"processed_{video_file.name.rsplit('.', 1)[0]}.avi"
                output_path = os.path.join(tempfile.gettempdir(), output_filename)
                
                # Progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Show processing info
                with st.spinner('Processing video... Please wait...'):
                    # Process video
                    success = process_video(video_path, output_path, progress_bar, status_text)
                
                # IMMEDIATELY update session state and show download
                if success:
                    # Store in session state
                    st.session_state.processed_video_path = output_path
                    st.session_state.processed_video_name = output_filename
                    st.session_state.processing_complete = True
                    
                    # Force immediate UI update
                    progress_bar.progress(1.0)
                    status_text.success("🎉 Processing completed!")
                    
                    st.balloons()  # Celebration effect!
                    
                    # Force page refresh to show download at top
                    st.rerun()
                else:
                    st.error("❌ Video processing failed. Please check your file and try again.")
            
            # Cleanup temporary upload file
            try:
                os.unlink(video_path)
            except:
                pass
        
        else:
            st.info("📤 Please upload a video file to begin processing.")
                output_filename = f"processed_{video_file.name.rsplit('.', 1)[0]}.avi"
                output_path = os.path.join(tempfile.gettempdir(), output_filename)
                
                # Progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Show processing info
                with st.spinner('Processing video... Please wait...'):
                    # Process video
                    success = process_video(video_path, output_path, progress_bar, status_text)
                
                # IMMEDIATELY update session state and show download
                if success:
                    # Store in session state
                    st.session_state.processed_video_path = output_path
                    st.session_state.processed_video_name = output_filename
                    st.session_state.processing_complete = True
                    
                    # Force immediate UI update
                    progress_bar.progress(1.0)
                    status_text.success("🎉 Processing completed!")
                    
                    st.balloons()  # Celebration effect!
                    
                    # Force page refresh to show download at top
                    st.rerun()
                else:
                    st.error("❌ Video processing failed. Please check your file and try again.")
            
            # Cleanup temporary upload file
            try:
                os.unlink(video_path)
            except:
                pass
        
        else:
            st.info("📤 Please upload a video file to begin processing.")
    
    else:
        # Show message that processing is complete and user can download above
        st.info("🎉 Your video has been processed! Use the download button above.")
        
        # Only show the reset button here
        if st.button("🔄 Process Another Video", key="main_reset"):
            try:
                os.unlink(st.session_state.processed_video_path)
            except:
                pass
            st.session_state.processing_complete = False
            st.session_state.processed_video_path = None
            st.session_state.processed_video_name = None
            st.rerun()
    
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
