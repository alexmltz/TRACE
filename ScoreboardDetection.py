import cv2
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import pytesseract
import os

class ScoreboardLayout(Enum):
    """Enumeration of different tennis broadcast layouts"""
    ESPN = "espn"
    EUROSPORT = "eurosport"
    TENNIS_CHANNEL = "tennis_channel"
    GENERIC_TOP = "generic_top"
    GENERIC_BOTTOM = "generic_bottom"
    GENERIC_CORNER = "generic_corner"
    UNKNOWN = "unknown"

@dataclass
class ScoreData:
    """Data structure to hold tennis score information"""
    player1_name: str = ""
    player2_name: str = ""
    player1_sets: List[int] = None
    player2_sets: List[int] = None
    current_set: int = 1
    player1_games: int = 0
    player2_games: int = 0
    player1_points: str = "0"  # "0", "15", "30", "40", "AD"
    player2_points: str = "0"
    serving_player: int = 0  # 1 or 2, 0 if unknown
    match_status: str = "in_progress"  # "in_progress", "finished"
    confidence: float = 0.0
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.player1_sets is None:
            self.player1_sets = []
        if self.player2_sets is None:
            self.player2_sets = []

@dataclass
class ScoreboardRegion:
    """Data structure to define a scoreboard region in the frame"""
    x: int
    y: int
    width: int
    height: int
    layout: ScoreboardLayout
    confidence: float = 0.0
    
    def get_roi(self, frame: np.ndarray) -> np.ndarray:
        """Extract region of interest from frame"""
        return frame[self.y:self.y + self.height, self.x:self.x + self.width]
    
    def get_bounds(self) -> Tuple[int, int, int, int]:
        """Get bounding box as (x, y, x2, y2)"""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

class LayoutManager:
    """Manages different scoreboard layout templates and detection"""
    
    def __init__(self):
        self.layout_templates = {}
        self.layout_regions = {}
        self._initialize_layouts()
    
    def _initialize_layouts(self):
        """Initialize common broadcast layout configurations"""
        # ESPN layout - typically top center
        self.layout_regions[ScoreboardLayout.ESPN] = {
            'relative_position': (0.25, 0.02, 0.5, 0.15),  # (x, y, width, height) as ratios
            'text_regions': {
                'player1_name': (0.0, 0.1, 0.3, 0.3),
                'player2_name': (0.0, 0.6, 0.3, 0.3),
                'player1_score': (0.35, 0.1, 0.6, 0.3),
                'player2_score': (0.35, 0.6, 0.6, 0.3),
            }
        }
        
        # Eurosport layout - typically top left
        self.layout_regions[ScoreboardLayout.EUROSPORT] = {
            'relative_position': (0.02, 0.02, 0.4, 0.12),
            'text_regions': {
                'player1_name': (0.0, 0.1, 0.4, 0.4),
                'player2_name': (0.0, 0.5, 0.4, 0.4),
                'player1_score': (0.5, 0.1, 0.5, 0.4),
                'player2_score': (0.5, 0.5, 0.5, 0.4),
            }
        }
        
        # Generic top layout
        self.layout_regions[ScoreboardLayout.GENERIC_TOP] = {
            'relative_position': (0.1, 0.02, 0.8, 0.12),
            'text_regions': {
                'player1_name': (0.0, 0.1, 0.25, 0.4),
                'player2_name': (0.0, 0.5, 0.25, 0.4),
                'player1_score': (0.3, 0.1, 0.7, 0.4),
                'player2_score': (0.3, 0.5, 0.7, 0.4),
            }
        }
        
        # Generic bottom layout
        self.layout_regions[ScoreboardLayout.GENERIC_BOTTOM] = {
            'relative_position': (0.1, 0.85, 0.8, 0.12),
            'text_regions': {
                'player1_name': (0.0, 0.1, 0.25, 0.4),
                'player2_name': (0.0, 0.5, 0.25, 0.4),
                'player1_score': (0.3, 0.1, 0.7, 0.4),
                'player2_score': (0.3, 0.5, 0.7, 0.4),
            }
        }
    
    def get_layout_region(self, layout: ScoreboardLayout, frame_width: int, frame_height: int) -> ScoreboardRegion:
        """Get scoreboard region for a specific layout"""
        if layout not in self.layout_regions:
            # Default to generic top layout
            layout = ScoreboardLayout.GENERIC_TOP
        
        region_config = self.layout_regions[layout]
        rel_x, rel_y, rel_w, rel_h = region_config['relative_position']
        
        x = int(rel_x * frame_width)
        y = int(rel_y * frame_height)
        width = int(rel_w * frame_width)
        height = int(rel_h * frame_height)
        
        return ScoreboardRegion(x, y, width, height, layout)
    
    def detect_layout(self, frame: np.ndarray) -> ScoreboardLayout:
        """Detect the most likely scoreboard layout from the frame"""
        # This is a simplified layout detection
        # In a full implementation, this would use template matching
        # or other computer vision techniques
        
        # For now, try to detect based on common regions
        frame_height, frame_width = frame.shape[:2]
        
        # Check top region for scoreboards
        top_region = frame[0:int(frame_height * 0.2), :]
        
        # Simple heuristic: look for text-like regions in different positions
        gray = cv2.cvtColor(top_region, cv2.COLOR_BGR2GRAY)
        
        # Look for high contrast regions that might contain text
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 20 and h > 10:  # Minimum text size
                text_regions.append((x, y, w, h))
        
        if len(text_regions) > 2:
            # Check if text regions are concentrated in center (ESPN style)
            center_regions = [r for r in text_regions if r[0] > frame_width * 0.25 and r[0] < frame_width * 0.75]
            if len(center_regions) >= len(text_regions) * 0.6:
                return ScoreboardLayout.ESPN
            
            # Check if text regions are concentrated on left (Eurosport style)
            left_regions = [r for r in text_regions if r[0] < frame_width * 0.4]
            if len(left_regions) >= len(text_regions) * 0.6:
                return ScoreboardLayout.EUROSPORT
        
        # Default to generic top
        return ScoreboardLayout.GENERIC_TOP

class ScoreboardRegionDetector:
    """Detects scoreboard regions in tennis broadcast frames"""
    
    def __init__(self, layout_manager: LayoutManager):
        self.layout_manager = layout_manager
        self.detected_layout = None
        self.region_cache = {}
    
    def detect_regions(self, frame: np.ndarray, force_layout: Optional[ScoreboardLayout] = None) -> List[ScoreboardRegion]:
        """Detect potential scoreboard regions in the frame"""
        frame_height, frame_width = frame.shape[:2]
        frame_key = f"{frame_width}x{frame_height}"
        
        if force_layout:
            layout = force_layout
        elif self.detected_layout:
            layout = self.detected_layout
        else:
            layout = self.layout_manager.detect_layout(frame)
            self.detected_layout = layout
        
        # Use cached region if available
        if frame_key in self.region_cache and self.region_cache[frame_key].layout == layout:
            return [self.region_cache[frame_key]]
        
        # Get region for the detected layout
        region = self.layout_manager.get_layout_region(layout, frame_width, frame_height)
        
        # Validate the region contains potential scoreboard content
        roi = region.get_roi(frame)
        if self._validate_scoreboard_region(roi):
            region.confidence = 0.8
            self.region_cache[frame_key] = region
            return [region]
        
        # If primary region doesn't validate, try other layouts including bottom-left for ATP Tennis TV
        fallback_layouts = [
            ScoreboardLayout.GENERIC_TOP, 
            ScoreboardLayout.GENERIC_BOTTOM,
            ScoreboardLayout.TENNIS_CHANNEL  # Try bottom-left region for ATP Tennis TV
        ]
        
        for alt_layout in fallback_layouts:
            if alt_layout == layout:
                continue
            
            alt_region = self.layout_manager.get_layout_region(alt_layout, frame_width, frame_height)
            alt_roi = alt_region.get_roi(frame)
            
            if self._validate_scoreboard_region(alt_roi):
                alt_region.confidence = 0.6
                self.region_cache[frame_key] = alt_region
                return [alt_region]
        
        # Try manual bottom-left region for ATP Tennis TV format
        bottom_left_region = ScoreboardRegion(
            x=int(0.02 * frame_width),
            y=int(0.75 * frame_height), 
            width=int(0.25 * frame_width),
            height=int(0.2 * frame_height),
            layout=ScoreboardLayout.TENNIS_CHANNEL,
            confidence=0.5
        )
        
        bottom_roi = bottom_left_region.get_roi(frame)
        if self._validate_scoreboard_region(bottom_roi):
            bottom_left_region.confidence = 0.7
            self.region_cache[frame_key] = bottom_left_region
            return [bottom_left_region]
        
        # Return the original region with low confidence
        region.confidence = 0.2
        return [region]
    
    def _validate_scoreboard_region(self, roi: np.ndarray) -> bool:
        """Validate if a region likely contains a scoreboard"""
        if roi.size == 0:
            return False
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Look for text-like patterns
        # High contrast regions that might contain text
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours that might be text
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_like_contours = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Text-like aspect ratio and size
            if 5 < w < roi.shape[1] * 0.8 and 8 < h < roi.shape[0] * 0.6:
                aspect_ratio = w / h
                if 0.2 < aspect_ratio < 8:  # Reasonable text aspect ratio
                    text_like_contours += 1
        
        # If we found some text-like regions, this might be a scoreboard
        return text_like_contours >= 2

class ScoreTextRecognizer:
    """Recognizes and extracts text from scoreboard regions"""
    
    def __init__(self):
        self.ocr_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
    
    def recognize_text(self, roi: np.ndarray) -> str:
        """Extract text from a region of interest"""
        if roi.size == 0:
            return ""
        
        # Preprocess the image for better OCR
        processed = self._preprocess_for_ocr(roi)
        
        try:
            # Use pytesseract to extract text
            text = pytesseract.image_to_string(processed, config=self.ocr_config)
            return text.strip()
        except Exception as e:
            print(f"OCR error: {e}")
            return ""
    
    def _preprocess_for_ocr(self, roi: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing for better OCR accuracy, especially for dark scoreboards"""
        # Convert to grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        # Resize for better OCR (OCR works better on larger images)
        height, width = gray.shape
        scale = 200 / height if height < 200 else 3.0
        new_width = int(width * scale)
        new_height = int(height * scale)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # For dark backgrounds with white text, invert if needed
        mean_val = np.mean(gray)
        if mean_val < 127:  # Dark background
            gray = cv2.bitwise_not(gray)
        
        # Apply Gaussian blur to smooth text
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply threshold to get clear black text on white background
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return thresh

class ScoreParser:
    """Parses and interprets tennis scoring from extracted text"""
    
    def __init__(self):
        self.tennis_points = ["0", "15", "30", "40", "AD"]
        self.score_patterns = {
            # Common score patterns
            'standard': r'(\d+)\s*[-–]\s*(\d+)',  # "6-4" or "6 - 4"
            'points': r'(0|15|30|40|AD)\s*[-–]\s*(0|15|30|40|AD)',  # "30-15"
            'sets': r'(\d+)\s*[-–]\s*(\d+)\s*,?\s*(\d+)\s*[-–]\s*(\d+)',  # "6-4, 3-6"
        }
    
    def parse_score_text(self, text: str, layout: ScoreboardLayout) -> Optional[ScoreData]:
        """Parse extracted text into structured score data"""
        if not text or len(text.strip()) < 2:
            return None
        
        # Clean the text
        text = text.strip()
        
        # Try ATP Tennis TV format first if it's a tennis channel layout
        if layout == ScoreboardLayout.TENNIS_CHANNEL:
            atp_result = self._parse_atp_tennis_tv_format(text)
            if atp_result:
                return atp_result
        
        # Fall back to generic parsing
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        # Initialize score data
        score_data = ScoreData()
        
        # Try to extract player names and scores
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if len(lines) >= 2:
            # Assume first two lines are player information
            player1_line = lines[0]
            player2_line = lines[1]
            
            # Parse player 1
            p1_name, p1_scores = self._parse_player_line(player1_line)
            score_data.player1_name = p1_name
            
            # Parse player 2
            p2_name, p2_scores = self._parse_player_line(player2_line)
            score_data.player2_name = p2_name
            
            # Parse scores
            if p1_scores and p2_scores:
                self._parse_scores(score_data, p1_scores, p2_scores)
        
        # Set confidence based on how much we were able to parse
        confidence = 0.0
        if score_data.player1_name and score_data.player2_name:
            confidence += 0.3
        if score_data.player1_points != "0" or score_data.player2_points != "0":
            confidence += 0.4
        if score_data.player1_games > 0 or score_data.player2_games > 0:
            confidence += 0.3
        
        score_data.confidence = confidence
        
        return score_data if confidence > 0.2 else None
    
    def _parse_atp_tennis_tv_format(self, text: str) -> Optional[ScoreData]:
        """Parse ATP Tennis TV scoreboard format specifically"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if len(lines) < 2:
            return None
        
        score_data = ScoreData()
        
        import re
        
        # Parse each line for player info
        for i, line in enumerate(lines[:2]):
            # Look for pattern: "Name Number [Number] [Text]"
            match = re.search(r'([A-Za-z]+)\s*(\d+)\s*(\d*)\s*([A-Za-z]*)', line)
            
            if match:
                name, num1, num2, extra = match.groups()
                
                if i == 0:  # First player
                    score_data.player1_name = name
                    score_data.player1_games = int(num1) if num1 else 0
                    if num2:
                        try:
                            score_data.player1_sets.append(int(num2))
                        except ValueError:
                            pass
                    if extra and extra.lower() in ['ad', 'advantage']:
                        score_data.player1_points = 'AD'
                    elif num1 in ['0', '15', '30', '40']:
                        score_data.player1_points = num1
                else:  # Second player  
                    score_data.player2_name = name
                    score_data.player2_games = int(num1) if num1 else 0
                    if num2:
                        try:
                            score_data.player2_sets.append(int(num2))
                        except ValueError:
                            pass
                    if extra and extra.lower() in ['ad', 'advantage']:
                        score_data.player2_points = 'AD'
                    elif num1 in ['0', '15', '30', '40']:
                        score_data.player2_points = num1
        
        # Set confidence based on what we found
        confidence = 0.0
        if score_data.player1_name and score_data.player2_name:
            confidence += 0.4
        if score_data.player1_games > 0 or score_data.player2_games > 0:
            confidence += 0.3
        if score_data.player1_points != "0" or score_data.player2_points != "0":
            confidence += 0.3
        
        score_data.confidence = confidence
        
        return score_data if confidence > 0.3 else None
    
    def _parse_player_line(self, line: str) -> Tuple[str, str]:
        """Parse a line containing player name and scores"""
        # Split by common delimiters
        parts = line.split()
        
        if not parts:
            return "", ""
        
        # Look for numeric patterns that indicate scores
        name_parts = []
        score_parts = []
        
        for part in parts:
            if any(char.isdigit() for char in part) or part in self.tennis_points:
                score_parts.append(part)
            else:
                name_parts.append(part)
        
        name = " ".join(name_parts)
        scores = " ".join(score_parts)
        
        return name, scores
    
    def _parse_scores(self, score_data: ScoreData, p1_scores: str, p2_scores: str):
        """Parse score strings into structured data"""
        import re
        
        # Try to find current points (0, 15, 30, 40, AD)
        for point in self.tennis_points:
            if point in p1_scores:
                score_data.player1_points = point
                break
        
        for point in self.tennis_points:
            if point in p2_scores:
                score_data.player2_points = point
                break
        
        # Try to find games in current set
        game_pattern = r'(\d+)'
        p1_games = re.findall(game_pattern, p1_scores)
        p2_games = re.findall(game_pattern, p2_scores)
        
        if p1_games and p2_games:
            # Take the last number as current games
            try:
                score_data.player1_games = int(p1_games[-1])
                score_data.player2_games = int(p2_games[-1])
            except (ValueError, IndexError):
                pass
            
            # Previous numbers might be set scores
            if len(p1_games) > 1 and len(p2_games) > 1:
                try:
                    score_data.player1_sets = [int(g) for g in p1_games[:-1]]
                    score_data.player2_sets = [int(g) for g in p2_games[:-1]]
                    score_data.current_set = len(score_data.player1_sets) + 1
                except (ValueError, IndexError):
                    pass

class ScoreboardDetector:
    """Main scoreboard detection class that coordinates all components"""
    
    def __init__(self, config_file: str = "templates/layout_config.json"):
        self.layout_manager = LayoutManager()
        self.region_detector = ScoreboardRegionDetector(self.layout_manager)
        self.text_recognizer = ScoreTextRecognizer()
        self.score_parser = ScoreParser()
        
        # Load configuration if available
        self.config = self._load_config(config_file)
        
        # Detection state
        self.current_layout = None
        self.score_history = []
        self.frame_count = 0
        self.detection_frequency = 5  # Process every 5 frames
        
    def _load_config(self, config_file: str) -> dict:
        """Load configuration from JSON file"""
        try:
            import json
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file {config_file} not found, using defaults")
            return {}
        except json.JSONDecodeError:
            print(f"Error parsing config file {config_file}, using defaults")
            return {}
    
    def detect_score(self, frame: np.ndarray, force_layout: Optional[ScoreboardLayout] = None) -> Optional[ScoreData]:
        """Detect and parse tennis score from a frame"""
        self.frame_count += 1
        
        # Skip frames based on detection frequency
        if self.frame_count % self.detection_frequency != 0:
            if self.score_history:
                return self.score_history[-1]
            return None
        
        # Detect scoreboard regions
        regions = self.region_detector.detect_regions(frame, force_layout)
        
        if not regions:
            return None
        
        # Process the highest confidence region
        best_region = max(regions, key=lambda r: r.confidence)
        
        # Extract text from the region
        roi = best_region.get_roi(frame)
        text = self.text_recognizer.recognize_text(roi)
        
        if not text:
            return None
        
        # Parse the extracted text
        score_data = self.score_parser.parse_score_text(text, best_region.layout)
        
        if score_data:
            score_data.timestamp = self.frame_count
            
            # Apply temporal filtering for better accuracy
            score_data = self._apply_temporal_filter(score_data)
            
            # Store in history
            self.score_history.append(score_data)
            
            # Keep only recent history
            if len(self.score_history) > 10:
                self.score_history = self.score_history[-10:]
            
            return score_data
        
        return None
    
    def _apply_temporal_filter(self, new_score: ScoreData) -> ScoreData:
        """Apply temporal filtering to improve score accuracy"""
        if not self.score_history:
            return new_score
        
        # Get the most recent score
        last_score = self.score_history[-1]
        
        # If the new score is very different from recent scores, be more conservative
        if self._scores_similar(new_score, last_score):
            # Boost confidence for consistent scores
            new_score.confidence = min(1.0, new_score.confidence + 0.2)
        else:
            # Reduce confidence for inconsistent scores
            new_score.confidence = max(0.0, new_score.confidence - 0.1)
        
        # Use previous score's player names if current detection failed
        if not new_score.player1_name and last_score.player1_name:
            new_score.player1_name = last_score.player1_name
        if not new_score.player2_name and last_score.player2_name:
            new_score.player2_name = last_score.player2_name
        
        return new_score
    
    def _scores_similar(self, score1: ScoreData, score2: ScoreData) -> bool:
        """Check if two scores are similar (for temporal filtering)"""
        # Check if games are the same or only differ by 1
        games_diff = abs(score1.player1_games - score2.player1_games) + abs(score1.player2_games - score2.player2_games)
        if games_diff > 2:
            return False
        
        # Check if points are reasonable transitions
        p1_points_similar = self._points_similar(score1.player1_points, score2.player1_points)
        p2_points_similar = self._points_similar(score1.player2_points, score2.player2_points)
        
        return p1_points_similar and p2_points_similar
    
    def _points_similar(self, points1: str, points2: str) -> bool:
        """Check if two point values are similar/reasonable transition"""
        point_order = ["0", "15", "30", "40", "AD"]
        
        if points1 == points2:
            return True
        
        # Allow transitions between adjacent points
        try:
            idx1 = point_order.index(points1)
            idx2 = point_order.index(points2)
            return abs(idx1 - idx2) <= 1
        except ValueError:
            # If point not in standard list, be permissive
            return True
    
    def get_scoreboard_regions(self, frame: np.ndarray) -> List[ScoreboardRegion]:
        """Get detected scoreboard regions for visualization"""
        return self.region_detector.detect_regions(frame)
    
    def set_detection_frequency(self, frequency: int):
        """Set how often to process frames (every N frames)"""
        self.detection_frequency = max(1, frequency)
    
    def reset_detection(self):
        """Reset detection state"""
        self.current_layout = None
        self.score_history = []
        self.frame_count = 0
        self.region_detector.detected_layout = None
        self.region_detector.region_cache = {}

def draw_scoreboard_detection(frame: np.ndarray, score_data: Optional[ScoreData], 
                            regions: List[ScoreboardRegion]) -> np.ndarray:
    """Draw scoreboard detection visualization on frame"""
    if not regions:
        return frame
    
    # Draw detected regions
    for region in regions:
        x, y, x2, y2 = region.get_bounds()
        
        # Color based on confidence
        if region.confidence > 0.7:
            color = (0, 255, 0)  # Green for high confidence
        elif region.confidence > 0.4:
            color = (0, 255, 255)  # Yellow for medium confidence
        else:
            color = (0, 0, 255)  # Red for low confidence
        
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
        
        # Draw confidence text
        conf_text = f"{region.layout.value}: {region.confidence:.2f}"
        cv2.putText(frame, conf_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Draw score information if available
    if score_data and score_data.confidence > 0.3:
        y_offset = 30
        
        # Draw player names and current score
        if score_data.player1_name or score_data.player2_name:
            score_text = f"{score_data.player1_name} {score_data.player1_points} - {score_data.player2_points} {score_data.player2_name}"
            cv2.putText(frame, score_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 25
        
        # Draw games
        if score_data.player1_games > 0 or score_data.player2_games > 0:
            games_text = f"Games: {score_data.player1_games} - {score_data.player2_games}"
            cv2.putText(frame, games_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 20
        
        # Draw sets if available
        if score_data.player1_sets or score_data.player2_sets:
            sets_text = f"Sets: {score_data.player1_sets} - {score_data.player2_sets}"
            cv2.putText(frame, sets_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 20
        
        # Draw confidence
        conf_text = f"Confidence: {score_data.confidence:.2f}"
        cv2.putText(frame, conf_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return frame