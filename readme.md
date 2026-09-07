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
- o = toggle window always-on-top (keep tracker in front of other apps)
- m = run guided 9-point screen calibration (after `c`)
- Esc = cancel guided screen calibration
- t = start/stop a derived-metrics research session
- v = run the standardized visual-attention target task (after `c`)
- F7 = toggle mouse control (disabled by default)
- j/l = orbit yaw left/right
- i/k = orbit pitch up/down
- [ / ] = zoom orbit view out/in
- r = reset orbit view
- x = stamp a green marker on the monitor where your gaze hits
- q = quit

Efficiency & HUD:
-----
The tracking window opens directly in the foreground in Always-On-Top mode so you can see tracking while interacting with other windows. A real-time HUD displays current FPS, processing latency (ms), face & eye lock status, gaze coordinates, and mouse toggle status. Press `o` at any time to toggle Always-On-Top on or off.

Notes
-----
Make sure you look at screen center when pressing c. The debug view won't render until you do this. Press `m` afterwards to refine screen mapping: a full-screen dot visits the center, corners, and edge midpoints. Keep your head still and look at each dot until it advances. The fitted profile is saved to `gaze_calibration.json` and is automatically reused when the display resolution matches.
Markers (x key) allow quick tests of where the system thinks you are looking.

Screening research metrics
-----

This prototype can record derived visual-engagement metrics during a session started and stopped with `t`. It stores no video, facial landmarks, name, age, or other identifier. The JSON report includes tracking availability, gaze speed, fixation ratio, gaze dispersion, spatial entropy, region transitions, and the longest period without tracking.

Press `v` after eye calibration to run a nine-target visual-attention task. The task presents a known target every three seconds and saves only per-target engagement measures: valid-gaze time, time on target, on-target ratio, and first-target latency. The task report is saved in `screening_reports/`.

Do not use this prototype to diagnose ASD, ADHD, developmental delay, or any medical condition. It has no clinically validated predictive model and therefore deliberately produces no automated risk score or referral decision. Use it only with appropriate guardian consent, privacy safeguards, and research or clinical governance. Any clinical deployment requires validation against representative, ethically collected ground-truth data and qualified clinician oversight.

Troubleshooting
-----
- If gaze appears jittery, increase filter_length.
- If the wrong camera opens, change cv2.VideoCapture(0) to another index.
