# TRACE

**Collected data visualized**

Court lines, ball, and player landmarks identified.

![Video](https://github.com/hgupt3/TRACE/assets/112455192/627e8ca6-86c1-4409-938d-2b45e875bbfa)


**Top-down view**

This view is extrapolated from data collected.

![Final](https://github.com/hgupt3/TRACE/assets/112455192/916287fb-e507-40a1-8bb8-7ab9f3dafbc3)


**Features**

✓ Ball Detection & Tracking using TrackNet neural network
✓ Player Body Tracking with MediaPipe pose estimation  
✓ Court Detection using line detection and perspective transformation
✓ **Scoreboard Detection & Score Tracking** for tennis matches
✓ Multi-layout support for various broadcast formats (ESPN, Eurosport, ATP Tennis TV, etc.)

**Required Libraries**

cv2
torch
mediapipe
numpy
pytesseract
opencv-contrib-python

**Installation**

```bash
pip install -r requirements.txt
```

**Usage**

```bash
python process_videos.py
```

Videos in `input_videos/` will be processed with all detection features enabled, including scoreboard detection and score tracking.

**Scoreboard Detection**

The system automatically detects and tracks tennis scores from various broadcast layouts:
- ATP Tennis TV (bottom-left scoreboards)
- ESPN (centered scoreboards)  
- Eurosport (left-aligned scoreboards)
- Generic top/bottom positioned scoreboards

Score information is overlaid on the output video with confidence indicators.

**Ball detection from:**

Yu-Chuan Huang, "TrackNet: Tennis Ball Tracking from Broadcast Video by Deep Learning Networks," Master Thesis, advised by Tsì-Uí İk and Guan-Hua Huang, National Chiao Tung University, Taiwan, April 2018.
