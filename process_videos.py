import cv2
import os
import numpy as np
from BallDetection import BallDetector
from BodyTracking import bodyMap
from TraceHeader import calculatePixels
from ScoreboardDetection import ScoreboardDetector, ScoreboardLayout, draw_scoreboard_detection
from mediapipe import solutions

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

def draw_body_tracking(frame, feet_points, hand_points, nose_points):
    """
    Draw body tracking points on the frame
    """
    if feet_points and not any(item is None for sublist in feet_points for item in sublist):
        # Draw feet points
        for i, foot in enumerate(feet_points):
            color = (0, 0, 255) if i < 2 else (0, 255, 255)  # Red for player 1, yellow for player 2
            cv2.circle(frame, foot, 5, color, -1)
    
    if hand_points and not any(item is None for sublist in hand_points for item in sublist):
        # Draw hand points
        for i, hand in enumerate(hand_points):
            color = (255, 0, 0) if i < 2 else (255, 255, 0)  # Blue for player 1, cyan for player 2
            cv2.circle(frame, hand, 5, color, -1)
    
    if nose_points and not any(item is None for sublist in nose_points for item in sublist):
        # Draw nose points
        for i, nose in enumerate(nose_points):
            color = (0, 255, 0) if i == 0 else (255, 0, 255)  # Green for player 1, magenta for player 2
            cv2.circle(frame, nose, 7, color, -1)
    
    return frame

class CropSettings:
    def __init__(self, x, y, xoffset=0, yoffset=0, xcenter=1, ycenter=0):
        self.x = x
        self.y = y
        self.xoffset = xoffset
        self.yoffset = yoffset
        self.xcenter = xcenter
        self.ycenter = ycenter

class BodyTracker:
    def __init__(self):
        mp_pose = solutions.pose
        self.pose = mp_pose.Pose(model_complexity=2, min_detection_confidence=0.25, min_tracking_confidence=0.25)
        self.x = 0
        self.y = 0
        self.x_avg = 0.0
        self.y_avg = 0.0

def process_video(input_path, output_path, ball_detector, enable_scoreboard=True):
    """
    Process a single video file with ball detection, body tracking, and scoreboard detection
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
    
    # Define the codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize crop settings for body tracking
    crop1 = CropSettings(x=50/100, y=33/100, xcenter=1, ycenter=0)
    crop2 = CropSettings(x=83/100, y=60/100, yoffset=40/100, xcenter=1, ycenter=0)
    
    # Calculate pixel values
    crop1 = calculatePixels(crop1, width, height)
    crop2 = calculatePixels(crop2, width, height)
    
    # Initialize body trackers
    body1 = BodyTracker()
    body2 = BodyTracker()
    
    # Initialize scoreboard detector
    scoreboard_detector = None
    if enable_scoreboard:
        try:
            scoreboard_detector = ScoreboardDetector()
            print("  ✓ Scoreboard Detection initialized")
        except Exception as e:
            print(f"  ✗ Scoreboard Detection failed to initialize: {e}")
            enable_scoreboard = False
    
    frame_count = 0
    detections_count = 0
    body_detections = 0
    scoreboard_detections = 0
    counter = 0
    n = 3  # smoothing frames
    
    print("Processing with enabled functionalities:")
    print("  ✓ Ball Detection & Tracking")
    print("  ✓ Body Tracking (2 players)")
    print("  ✗ Court Detection (disabled)")
    if enable_scoreboard:
        print("  ✓ Scoreboard Detection & Score Tracking")
    else:
        print("  ✗ Scoreboard Detection (disabled)")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            counter += 1
            
            # Detect ball in current frame
            ball_detector.detect_ball(frame)
            ball_coords = None
            
            # Get the latest ball coordinates
            if len(ball_detector.xy_coordinates) > 0:
                x, y = ball_detector.xy_coordinates[-1]
                if x is not None and y is not None:
                    detections_count += 1
                    ball_coords = (x, y)
                    # Draw ball detection on frame
                    frame = draw_ball_detection(frame, x, y)
            
            # Body tracking
            feet_points, hand_points, nose_points = bodyMap(frame, body1.pose, body2.pose, crop1, crop2)
            
            # Check if we have valid body tracking data
            if (not any(item is None for sublist in feet_points for item in sublist) or 
                not any(item is None for sublist in hand_points for item in sublist) or 
                not any(item is None for sublist in nose_points for item in sublist)):
                
                body_detections += 1
                
                # Draw body tracking
                frame = draw_body_tracking(frame, feet_points, hand_points, nose_points)
                
                # Calculate body positions if we have feet data
                if not any(item is None for sublist in feet_points for item in sublist):
                    # Prioritizing lower foot y in body average y position
                    if feet_points[0][1] > feet_points[1][1]:
                        lower_foot1 = feet_points[0][1]
                        higher_foot1 = feet_points[1][1]
                    else:
                        lower_foot1 = feet_points[1][1]
                        higher_foot1 = feet_points[0][1]
                        
                    if feet_points[2][1] > feet_points[3][1]:
                        lower_foot2 = feet_points[2][1]
                        higher_foot2 = feet_points[3][1]
                    else:
                        lower_foot2 = feet_points[3][1]
                        higher_foot2 = feet_points[2][1]
                    
                    # Calculate body centers
                    body1.x = (feet_points[0][0] + feet_points[1][0]) / 2
                    body1.y = lower_foot1 * 0.8 + higher_foot1 * 0.2
                    
                    body2.x = (feet_points[2][0] + feet_points[3][0]) / 2
                    body2.y = lower_foot2 * 0.8 + higher_foot2 * 0.2
                    
                    # Body coordinate smoothing
                    coeff = 1.0 / min(counter, n)
                    body1.x_avg = coeff * body1.x + (1.0 - coeff) * body1.x_avg
                    body1.y_avg = coeff * body1.y + (1.0 - coeff) * body1.y_avg
                    body2.x_avg = coeff * body2.x + (1.0 - coeff) * body2.x_avg
                    body2.y_avg = coeff * body2.y + (1.0 - coeff) * body2.y_avg
                    
                    # Draw smoothed body positions
                    cv2.circle(frame, (int(body1.x_avg), int(body1.y_avg)), 10, (0, 255, 255), -1)
                    cv2.circle(frame, (int(body2.x_avg), int(body2.y_avg)), 10, (255, 255, 0), -1)
            
            # Scoreboard detection
            current_score = None
            scoreboard_regions = []
            if enable_scoreboard and scoreboard_detector:
                try:
                    current_score = scoreboard_detector.detect_score(frame)
                    scoreboard_regions = scoreboard_detector.get_scoreboard_regions(frame)
                    
                    if current_score and current_score.confidence > 0.3:
                        scoreboard_detections += 1
                        
                        # Draw scoreboard detection visualization
                        frame = draw_scoreboard_detection(frame, current_score, scoreboard_regions)
                        
                except Exception as e:
                    if frame_count % 100 == 0:  # Only print occasionally to avoid spam
                        print(f"Scoreboard detection error: {e}")
            
            # Write frame to output video
            out.write(frame)
            
            # Print progress every 100 frames
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
                print(f"  Ball detections: {detections_count}")
                print(f"  Body detections: {body_detections}")
                if enable_scoreboard:
                    print(f"  Scoreboard detections: {scoreboard_detections}")
    
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Release everything
        cap.release()
        out.release()
    
    print(f"Finished processing {input_path}")
    print(f"Total frames: {frame_count}")
    print(f"Ball detections: {detections_count}")
    print(f"Body detections: {body_detections}")
    if enable_scoreboard:
        print(f"Scoreboard detections: {scoreboard_detections}")
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
    print("Enabled functionalities:")
    print("  ✓ Ball Detection & Tracking")
    print("  ✓ Body Tracking (2 players)")
    print("  ✗ Court Detection (disabled)")
    print("  ✓ Scoreboard Detection & Score Tracking")
    
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