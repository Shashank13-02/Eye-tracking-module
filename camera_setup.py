"""Guided setup for camera intrinsics and measured screen geometry.

Usage examples:
  py -3 camera_setup.py intrinsics --cols 9 --rows 6 --square-mm 25
  py -3 camera_setup.py screen
"""
from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np

from geometry_calibration import calibrate_camera_from_chessboards, save_camera_intrinsics, save_screen_geometry


ROOT = os.path.dirname(os.path.abspath(__file__))


def capture_intrinsics(cols: int, rows: int, square_mm: float) -> None:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam 0.")
    observations = []
    view_signatures: list[tuple[float, float, float]] = []
    pattern = (cols, rows)
    print("Show a printed checkerboard at varied distances, angles, and screen locations.")
    print("Press SPACE to save a detected view; C to calibrate after 12+ views; Q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # The SB detector is substantially more robust on ordinary printed
        # boards, while the classic detector remains available for older OpenCV.
        if hasattr(cv2, "findChessboardCornersSB"):
            found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
        else:
            found, corners = cv2.findChessboardCorners(gray, pattern, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        preview = frame.copy()
        if found:
            cv2.drawChessboardCorners(preview, pattern, corners, found)
        cv2.putText(preview, f"Views: {len(observations)}  {'BOARD FOUND' if found else 'find board'}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if found else (0, 180, 255), 2)
        cv2.putText(preview, "SPACE save | C calibrate (12+ varied views) | Q quit", (15, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        cv2.putText(preview, "Move board: centre, edges, nearer/farther, then tilt it", (15, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
        cv2.imshow("Camera Intrinsics Setup", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and found:
            refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            flat = refined.reshape(-1, 2)
            centre = np.mean(flat, axis=0)
            span = float(np.linalg.norm(np.max(flat, axis=0) - np.min(flat, axis=0)))
            signature = (centre[0] / frame.shape[1], centre[1] / frame.shape[0], span / max(frame.shape[:2]))
            duplicate = any(
                abs(signature[0] - old[0]) < 0.04 and abs(signature[1] - old[1]) < 0.04 and abs(signature[2] - old[2]) < 0.05
                for old in view_signatures
            )
            if duplicate:
                print("Skipped near-duplicate view. Move, tilt, or change the checkerboard distance.")
            else:
                observations.append(refined)
                view_signatures.append(signature)
                print(f"Saved varied view {len(observations)}")
        elif key == ord("c"):
            if len(observations) < 12:
                print(f"Need at least 12 varied views; currently have {len(observations)}.")
                continue
            try:
                width, height = frame.shape[1], frame.shape[0]
                matrix, distortion, error = calibrate_camera_from_chessboards(observations, cols, rows, square_mm, width, height)
                destination = os.path.join(ROOT, "camera_intrinsics.json")
                save_camera_intrinsics(destination, matrix, distortion, width, height, error)
                quality = "good" if error <= 0.5 else "usable" if error <= 1.0 else "repeat recommended"
                print(f"Saved {destination}; calibration RMS reprojection error: {error:.3f}px ({quality}).")
                break
            except ValueError as error:
                print(f"Calibration could not be completed: {error}")
        elif key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def capture_screen_geometry() -> None:
    print("Measure the physical setup in millimetres. Use the same camera mount and monitor position used for tracking.")
    width = float(input("Visible screen width (mm): "))
    height = float(input("Visible screen height (mm): "))
    distance = float(input("Camera lens to screen plane, perpendicular distance (mm): "))
    offset_x = float(input("Camera lens horizontal offset from screen centre (mm; right positive): "))
    offset_y = float(input("Camera lens vertical offset from screen centre (mm; down positive): "))
    destination = os.path.join(ROOT, "screen_geometry.json")
    save_screen_geometry(destination, width, height, distance, offset_x, offset_y)
    print(f"Saved {destination}")


parser = argparse.ArgumentParser(description="NeuroGaze camera and screen setup")
sub = parser.add_subparsers(dest="command", required=True)
intrinsics = sub.add_parser("intrinsics")
intrinsics.add_argument("--cols", type=int, required=True, help="inner checkerboard corners across")
intrinsics.add_argument("--rows", type=int, required=True, help="inner checkerboard corners down")
intrinsics.add_argument("--square-mm", type=float, required=True, help="printed square edge length")
sub.add_parser("screen")
args = parser.parse_args()
if args.command == "intrinsics":
    capture_intrinsics(args.cols, args.rows, args.square_mm)
else:
    capture_screen_geometry()
