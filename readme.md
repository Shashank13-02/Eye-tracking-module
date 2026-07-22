**Eye Tracking and Monitor Control**

====================================

This is a 3D eye tracker that works with the webcam.


Usage
-----

Connect a webcam. By default, camera index 0 is used. Change in code if needed.

Run the tracker:
python MonitorTracking.py

Windows will open showing:
- Integrated Eye Tracking: live video with eye landmarks, gaze rays, and calibration overlays.
- Head/Eye Debug: a 3D orbit-view with the head, gaze vectors, and the calibrated virtual monitor.


Interactive controls:
-----
- c = calibrate (screen center)
- F7 = toggle mouse control (disabled by default)
- j/l = orbit yaw left/right
- i/k = orbit pitch up/down
- [ / ] = zoom orbit view out/in
- r = reset orbit view
- x = stamp a green marker on the monitor where your gaze hits
- q = quit

Notes
-----
Make sure you look at screen center when pressing c. The debug view won't render until you do this. 
Markers (x key) allow quick tests of where the system thinks you are looking.

Troubleshooting
-----
- If gaze appears jittery, increase filter_length.
- If the wrong camera opens, change cv2.VideoCapture(0) to another index.
