import os
import cv2
import numpy as np
import mediapipe as mp
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Clean Iris, Eye, and Hand Tracking")
    parser.add_argument("--image", type=str, help="Path to input image file (runs static mode)")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save output files")
    parser.add_argument("--margin-iris", type=float, default=1.3, help="Margin around iris crop")
    parser.add_argument("--margin-eye", type=float, default=0.25, help="Margin around whole eye crop")
    return parser.parse_args()

# Landmark Index Definitions
LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]

LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

def extract_iris_region(image, center_pt, boundary_pts, margin=1.3):
    """Extracts iris crop, masked iris, and transparent RGBA crop."""
    distances = [np.linalg.norm(np.array(center_pt) - np.array(pt)) for pt in boundary_pts]
    radius = int(np.mean(distances))
    box_radius = int(radius * margin)
    cx, cy = int(center_pt[0]), int(center_pt[1])
    h, w, _ = image.shape
    
    x1 = max(0, cx - box_radius)
    y1 = max(0, cy - box_radius)
    x2 = min(w, cx + box_radius)
    y2 = min(h, cy + box_radius)
    
    crop = image[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return None, None, None, (cx, cy), radius
        
    crop_h, crop_w, _ = crop.shape
    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    crop_cx = cx - x1
    crop_cy = cy - y1
    cv2.circle(mask, (crop_cx, crop_cy), radius, 255, -1)
    
    masked_crop = cv2.bitwise_and(crop, crop, mask=mask)
    bgra_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    bgra_crop[:, :, 3] = mask
    
    return crop, masked_crop, bgra_crop, (cx, cy), radius

def extract_eye_region(image, contour_pts, margin=0.25):
    """Extracts whole eye crop, polygon masked eye, and transparent RGBA eye crop."""
    pts = np.array(contour_pts, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    
    dw = int(w * margin)
    dh = int(h * margin)
    
    img_h, img_w, _ = image.shape
    x1 = max(0, x - dw)
    y1 = max(0, y - dh)
    x2 = min(img_w, x + w + dw)
    y2 = min(img_h, y + h + dh)
    
    crop = image[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return None, None, None
        
    crop_h, crop_w, _ = crop.shape
    shifted_pts = pts - np.array([x1, y1])
    
    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    cv2.fillPoly(mask, [shifted_pts], 255)
    
    masked_crop = cv2.bitwise_and(crop, crop, mask=mask)
    bgra_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    bgra_crop[:, :, 3] = mask
    
    return crop, masked_crop, bgra_crop

def save_all_results(base_name, output_dir, left_iris, right_iris, left_eye, right_eye):
    os.makedirs(output_dir, exist_ok=True)
    paths = {}
    
    # Save Left Iris
    if left_iris[0] is not None:
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_left_iris_crop.png"), left_iris[0])
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_left_iris_masked.png"), left_iris[1])
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_left_iris_transparent.png"), left_iris[2])
        paths['left_iris'] = os.path.join(output_dir, f"{base_name}_left_iris_transparent.png")
        
    # Save Right Iris
    if right_iris[0] is not None:
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_right_iris_crop.png"), right_iris[0])
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_right_iris_masked.png"), right_iris[1])
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_right_iris_transparent.png"), right_iris[2])
        paths['right_iris'] = os.path.join(output_dir, f"{base_name}_right_iris_transparent.png")
        
    # Save Left Eye
    if left_eye[0] is not None:
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_left_eye_crop.png"), left_eye[0])
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_left_eye_masked.png"), left_eye[1])
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_left_eye_transparent.png"), left_eye[2])
        paths['left_eye'] = os.path.join(output_dir, f"{base_name}_left_eye_transparent.png")
        
    # Save Right Eye
    if right_eye[0] is not None:
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_right_eye_crop.png"), right_eye[0])
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_right_eye_masked.png"), right_eye[1])
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_right_eye_transparent.png"), right_eye[2])
        paths['right_eye'] = os.path.join(output_dir, f"{base_name}_right_eye_transparent.png")
        
    return paths

def run_webcam_mode(margin_iris, margin_eye):
    print("\nInitializing Webcam Mode (Clean Tracking)...")
    print("Controls:")
    print("  - Press 's' to Capture & Save crops to the 'output/' folder.")
    print("  - Press 'q' to Quit.")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    # Solutions setup
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    window_title = "Clean Face & Hand Tracker"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    
    save_counter = 0
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh, \
    mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
                
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = face_mesh.process(rgb_frame)
            hand_results = hands.process(rgb_frame)
            
            vis_frame = frame.copy()
            
            # 1. Draw Hand Tracking
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        vis_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
            
            left_iris_data, right_iris_data = None, None
            left_eye_data, right_eye_data = None, None
            
            # 2. Draw Face & Eye/Iris Tracking
            if face_results.multi_face_landmarks:
                landmarks = face_results.multi_face_landmarks[0].landmark
                to_px = lambda lm: (int(lm.x * w), int(lm.y * h))
                
                # Fetch Points
                left_iris_center = to_px(landmarks[468])
                left_iris_bounds = [to_px(landmarks[idx]) for idx in [469, 470, 471, 472]]
                right_iris_center = to_px(landmarks[473])
                right_iris_bounds = [to_px(landmarks[idx]) for idx in [474, 475, 476, 477]]
                
                left_eye_pts = [to_px(landmarks[idx]) for idx in LEFT_EYE_CONTOUR]
                right_eye_pts = [to_px(landmarks[idx]) for idx in RIGHT_EYE_CONTOUR]
                
                # Extract regions
                li_crop, li_masked, li_bgra, lic, lir = extract_iris_region(frame, left_iris_center, left_iris_bounds, margin_iris)
                ri_crop, ri_masked, ri_bgra, ric, rir = extract_iris_region(frame, right_iris_center, right_iris_bounds, margin_iris)
                
                le_crop, le_masked, le_bgra = extract_eye_region(frame, left_eye_pts, margin_eye)
                re_crop, re_masked, re_bgra = extract_eye_region(frame, right_eye_pts, margin_eye)
                
                left_iris_data = (li_crop, li_masked, li_bgra)
                right_iris_data = (ri_crop, ri_masked, ri_bgra)
                left_eye_data = (le_crop, le_masked, le_bgra)
                right_eye_data = (re_crop, re_masked, re_bgra)
                
                # Draw Face Oval
                for connection in mp_face_mesh.FACEMESH_FACE_OVAL:
                    p1 = landmarks[connection[0]]
                    p2 = landmarks[connection[1]]
                    cv2.line(vis_frame, to_px(p1), to_px(p2), (255, 120, 0), 1, cv2.LINE_AA) # Blue outline
                
                # Draw Eye contours
                cv2.polylines(vis_frame, [np.array(left_eye_pts)], True, (0, 255, 255), 1, cv2.LINE_AA)  # Yellow
                cv2.polylines(vis_frame, [np.array(right_eye_pts)], True, (0, 255, 255), 1, cv2.LINE_AA)
                
                # Draw Iris Outlines
                cv2.circle(vis_frame, lic, lir, (0, 255, 0), 1, cv2.LINE_AA) # Green circle
                cv2.circle(vis_frame, lic, 1, (0, 0, 255), -1) # Red center dot
                cv2.circle(vis_frame, ric, rir, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.circle(vis_frame, ric, 1, (0, 0, 255), -1)

            cv2.imshow(window_title, vis_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if left_iris_data is not None and right_iris_data is not None and left_iris_data[0] is not None:
                    save_counter += 1
                    saved_paths = save_all_results(
                        f"capture_{save_counter}", 
                        "output", 
                        left_iris_data, 
                        right_iris_data, 
                        left_eye_data, 
                        right_eye_data
                    )
                    print(f"\nSaved capture #{save_counter}:")
                    for k, v in saved_paths.items():
                        print(f"  - {k}: {v}")
                else:
                    print("Could not save: No face/eye tracking lock.")
                    
    cap.release()
    cv2.destroyAllWindows()
    print("Session closed.")

def process_static_image(image_path, output_dir, margin_iris, margin_eye):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
        
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return
        
    h, w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    print("Processing image with MediaPipe Face Mesh & Hands...")
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh, \
    mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    ) as hands:
        
        results_face = face_mesh.process(rgb_image)
        results_hands = hands.process(rgb_image)
        
        vis_image = image.copy()
        
        # Draw Hands
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    vis_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        
        if not results_face.multi_face_landmarks:
            print("No face detected.")
            # Save visualizer with hand if any
            cv2.imwrite(os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_tracked.png"), vis_image)
            return
            
        landmarks = results_face.multi_face_landmarks[0].landmark
        to_px = lambda lm: (int(lm.x * w), int(lm.y * h))
        
        # Coordinates
        left_iris_center = to_px(landmarks[468])
        left_iris_bounds = [to_px(landmarks[idx]) for idx in [469, 470, 471, 472]]
        right_iris_center = to_px(landmarks[473])
        right_iris_bounds = [to_px(landmarks[idx]) for idx in [474, 475, 476, 477]]
        
        left_eye_pts = [to_px(landmarks[idx]) for idx in LEFT_EYE_CONTOUR]
        right_eye_pts = [to_px(landmarks[idx]) for idx in RIGHT_EYE_CONTOUR]
        
        # Extraction
        li_crop, li_masked, li_bgra, lic, lir = extract_iris_region(image, left_iris_center, left_iris_bounds, margin_iris)
        ri_crop, ri_masked, ri_bgra, ric, rir = extract_iris_region(image, right_iris_center, right_iris_bounds, margin_iris)
        
        le_crop, le_masked, le_bgra = extract_eye_region(image, left_eye_pts, margin_eye)
        re_crop, re_masked, re_bgra = extract_eye_region(image, right_eye_pts, margin_eye)
        
        # Save crops
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        saved_paths = save_all_results(
            base_name, 
            output_dir, 
            (li_crop, li_masked, li_bgra), 
            (ri_crop, ri_masked, ri_bgra),
            (le_crop, le_masked, le_bgra),
            (re_crop, re_masked, re_bgra)
        )
        
        # Draw Face annotations
        for connection in mp_face_mesh.FACEMESH_FACE_OVAL:
            p1 = landmarks[connection[0]]
            p2 = landmarks[connection[1]]
            cv2.line(vis_image, to_px(p1), to_px(p2), (255, 120, 0), 1, cv2.LINE_AA)
            
        cv2.polylines(vis_image, [np.array(left_eye_pts)], True, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.polylines(vis_image, [np.array(right_eye_pts)], True, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(vis_image, lic, lir, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(vis_image, lic, 1, (0, 0, 255), -1)
        cv2.circle(vis_image, ric, rir, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(vis_image, ric, 1, (0, 0, 255), -1)
        
        vis_path = os.path.join(output_dir, f"{base_name}_tracked.png")
        cv2.imwrite(vis_path, vis_image)
        
        print("\nEye, Iris, and Hand Tracking Completed Successfully!")
        print(f"Tracking visualization saved to: {vis_path}")
        print("Crops saved:")
        for k, v in saved_paths.items():
            print(f"  - {k}: {v}")

if __name__ == "__main__":
    args = get_args()
    if args.image:
        process_static_image(args.image, args.output_dir, args.margin_iris, args.margin_eye)
    else:
        run_webcam_mode(args.margin_iris, args.margin_eye)
