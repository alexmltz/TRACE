import cv2
import os
import numpy as np
from BallDetection import BallDetector

def draw_ball_detection(frame, x, y, radius=8, color=(0, 255, 0), thickness=2):
    """
    Draw a circle around the detected ball position
    """
    if x is not None and y is not None:
        cv2.circle(frame, (int(x), int(y)), radius, color, thickness)
        # Add a small cross at the center
        cv2.line(frame, (int(x-5), int(y)), (int(x+5), int(y)), color, thickness)
        cv2.line(frame, (int(x), int(y-5)), (int(x), int(y+5)), color, thickness)
    return frame

def process_video(input_path, output_path, ball_detector):
    """
    Process a single video file and detect tennis balls in each frame
    """
    print(f"Processing video: {input_path}")
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    detections_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Detect ball in current frame
        ball_detector.detect_ball(frame)
        
        # Get the latest ball coordinates
        if len(ball_detector.xy_coordinates) > 0:
            x, y = ball_detector.xy_coordinates[-1]
            if x is not None and y is not None:
                detections_count += 1
                # Draw ball detection on frame
                frame = draw_ball_detection(frame, x, y)
        
        # Write frame to output video
        out.write(frame)
        
        # Print progress every 100 frames
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames, detections: {detections_count}")
    
    # Release everything
    cap.release()
    out.release()
    
    print(f"Finished processing {input_path}")
    print(f"Total frames: {frame_count}, Ball detections: {detections_count}")
    print(f"Output saved to: {output_path}")
    
    return True

def main():
    """
    Main function to process all videos in the input_videos folder
    """
    input_folder = "input_videos"
    output_folder = "output_videos"
    weights_path = "TrackNet/Weights.pth"
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder {input_folder} does not exist")
        return
    
    # Check if weights file exists
    if not os.path.exists(weights_path):
        print(f"Error: Weights file {weights_path} does not exist")
        return
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get list of video files in input folder
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = [f for f in os.listdir(input_folder) 
                   if any(f.lower().endswith(ext) for ext in video_extensions)]
    
    if not video_files:
        print(f"No video files found in {input_folder}")
        return
    
    print(f"Found {len(video_files)} video(s) to process: {video_files}")
    
    # Initialize ball detector
    print("Initializing ball detector...")
    ball_detector = BallDetector(weights_path, out_channels=2)
    print("Ball detector initialized successfully!")
    
    # Process each video
    for video_file in video_files:
        input_path = os.path.join(input_folder, video_file)
        
        # Create output filename with _detected suffix
        name, ext = os.path.splitext(video_file)
        output_filename = f"{name}_detected{ext}"
        output_path = os.path.join(output_folder, output_filename)
        
        # Reset detector for each video
        ball_detector.current_frame = None
        ball_detector.last_frame = None
        ball_detector.before_last_frame = None
        ball_detector.video_width = None
        ball_detector.video_height = None
        ball_detector.xy_coordinates = np.array([[None, None], [None, None]])
        
        # Process the video
        success = process_video(input_path, output_path, ball_detector)
        
        if success:
            print(f"✓ Successfully processed {video_file}")
        else:
            print(f"✗ Failed to process {video_file}")
        
        print("-" * 50)

if __name__ == "__main__":
    main() 