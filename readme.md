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
Make sure you look at screen center when pressing c. The debug view won't render until you do this. Press `m` afterwards to run full screen calibration and validation:
1. **9-Point Training**: A full-screen dot visits the center, corners, and edge midpoints to fit a second-order polynomial gaze mapping model.
2. **5-Point Independent Validation**: A separate, disjoint set of 5 target positions (screen quadrants and upper-center) evaluates the model on independent samples.
3. **Calibration Quality Report**: Displays the primary **Median Gaze Error (px)**, engineering quality status (`EXCELLENT` <=60px, `GOOD` <=110px, `FAIR` <=180px, `POOR` >180px), RMSE, MAE, 95th percentile error, horizontal/vertical errors, normalized screen diagonal error (%), visual angle error (degrees), and an on-screen spatial vector map.
4. **Version 2 Profile**: Saved to `gaze_calibration.json` with comprehensive training and validation metrics, automatically reloaded on startup.
5. **Training-only model selection**: Chooses a standardized ridge-regularized quadratic mapping using leave-one-training-target-out error. The five independent validation targets are never used to select the model.
Markers (x key) allow quick tests of where the system thinks you are looking.

Screening research metrics
-----

This prototype can record derived visual-engagement metrics during a session started and stopped with `t`. It stores no video, facial landmarks, name, age, or other identifier. The JSON report includes tracking availability, gaze speed, fixation ratio, gaze dispersion, spatial entropy, region transitions, and the longest period without tracking.

Frame-quality gating
--------------------

Before a frame can affect calibration, mouse movement, task scoring, or session gaze metrics, the tracker now checks face size/distance proxy, per-eye visibility, blink proxy, image illumination, landmark jitter, head yaw/pitch/roll, and calibration age. Invalid frames are marked as paused and excluded rather than being interpreted as gaze. MediaPipe FaceMesh does not expose a reliable face-detector confidence score, so reports record that this value is unavailable rather than fabricating one. Head angles use camera-calibrated `solvePnP` when an intrinsics profile exists; otherwise they are explicitly approximate image-geometry estimates, not clinical measurements.

Geometry setup
--------------

The tracker can derive pose-aware eye centres and radii from eye corners, eyelids, irises, and a `solvePnP` head-pose estimate. It begins in explicitly labelled approximate-camera mode until you measure the camera. For calibrated geometry, keep the webcam resolution and mount fixed, then:

1. Print a checkerboard and run `py -3 camera_setup.py intrinsics --cols 9 --rows 6 --square-mm 25`, replacing these values with your board's inner-corner dimensions and measured square edge.
2. Capture at least 12 varied board views, press `c`, and inspect the saved `camera_intrinsics.json` reprojection error.
3. Run `py -3 camera_setup.py screen` and enter visible screen dimensions, lens-to-screen distance, and camera offset from screen centre in millimetres.

These profiles are specific to one camera, resolution, monitor, and physical mount. Moving any of them requires a new setup. The measured screen plane improves geometry/debug rendering; the independently validated polynomial model remains the practical screen-coordinate mapping.

Ground truth
------------

See [GROUND_TRUTH_DATA_PLAN.md](GROUND_TRUTH_DATA_PLAN.md) for controlled-access and public research resources, the required independent-assessment fields, and the participant-level train/validation/test split contract. The tracked `ground_truth_manifest_template.csv` contains no participant data; keep any completed manifest outside version control as `ground_truth_manifest.csv`.

Use `dataset_normalizer.py` only to convert consented NeuroGaze research streams to a provenance-preserving normalized CSV. Do not merge public eye-tracking datasets into NeuroGaze as though the hardware, stimuli, calibration, or labels were interchangeable.

Consented research gaze-event streams
--------------------------------------

The `T` session control now requires explicit local consent configuration before it stores timestamped normalized gaze coordinates. Set a non-identifying study code and consent acknowledgement before launching the tracker:

```powershell
$env:NEUROGAZE_CONSENT = "YES"
$env:NEUROGAZE_PARTICIPANT_PSEUDONYM = "study-001"
py -3 MonitorTracking.py
```

Each session saves an ignored local JSON file under `research_gaze_streams/` containing timestamped gaze samples, validity, quality flags, active task event, and derived fixation, saccade, dwell, revisit, response-latency, and missing-data events. Do not use a child name, date of birth, hospital ID, or other direct identifier as the pseudonym. These files are research data and need an approved retention, access, and deletion policy.

Age-banded research tasks
--------------------------

During a consented `T` session, an in-app screen asks for the child's age in whole months (0–72). It shows the selected band and only enables its permitted research tasks: `1` face-versus-non-social preference, `2` moving-object tracking, and `3` social-cue gaze following. The protocol bands are 0–8 months (no self-directed screen task), 9–17 months (face preference and motion only), 18–24 months, 25–30 months, and 31–72 months. A biomarker JSON report is created when the session ends.

The age anchors reflect AAP developmental-screening time points (9, 18, 24, and 30 months) and WHO’s early-learning focus in the first three years; they do **not** provide gaze norms or validate these stimuli. The current graphics are deliberately simple research placeholders and must be co-designed and validated with pediatric/developmental clinicians before use beyond engineering research. See [AAP developmental surveillance and screening](https://www.aap.org/en/patient-care/developmental-surveillance-and-screening-patient-care/) and the [WHO early childhood development guideline](https://www.who.int/publications/i/item/97892400020986).

Press `v` after eye calibration to run a nine-target visual-attention task. The task presents a known target every three seconds and saves only per-target engagement measures: valid-gaze time, time on target, on-target ratio, and first-target latency. The task report is saved in `screening_reports/`.

Do not use this prototype to diagnose ASD, ADHD, developmental delay, or any medical condition. It has no clinically validated predictive model and therefore deliberately produces no automated risk score or referral decision. Use it only with appropriate guardian consent, privacy safeguards, and research or clinical governance. Any clinical deployment requires validation against representative, ethically collected ground-truth data and qualified clinician oversight.

Troubleshooting
-----
- If gaze appears jittery, increase filter_length.
- If the wrong camera opens, change cv2.VideoCapture(0) to another index.
