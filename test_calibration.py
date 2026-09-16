"""Unit tests for calibration_validator.py."""

import math
import os
import tempfile
import unittest
import numpy as np

from calibration_validator import (
    CALIBRATION_TARGETS,
    VALIDATION_TARGETS,
    QUALITY_THRESHOLDS,
    vector_to_yaw_pitch_deg,
    compute_calibration_features,
    fit_gaze_model,
    select_regularized_gaze_model,
    predict_screen_coordinates,
    classify_calibration_quality,
    evaluate_calibration,
    robust_target_estimate,
    save_calibration_profile_v2,
    load_calibration_profile_v2,
)


class TestCalibrationValidator(unittest.TestCase):
    def test_target_independence(self):
        """Verify validation targets are strictly disjoint from calibration targets."""
        calib_set = set(CALIBRATION_TARGETS)
        val_set = set(VALIDATION_TARGETS)
        overlap = calib_set.intersection(val_set)
        self.assertEqual(len(overlap), 0, f"Found overlapping targets between training and validation: {overlap}")
        self.assertEqual(len(CALIBRATION_TARGETS), 13)
        self.assertEqual(len(VALIDATION_TARGETS), 6)

    def test_vector_to_yaw_pitch(self):
        """Test continuous signed yaw and pitch conversion across cardinal directions."""
        # Straight ahead: -Z
        yaw, pitch = vector_to_yaw_pitch_deg(np.array([0, 0, -1]))
        self.assertAlmostEqual(yaw, 0.0, places=3)
        self.assertAlmostEqual(pitch, 0.0, places=3)

        # Looking right: +X, -Z
        yaw, pitch = vector_to_yaw_pitch_deg(np.array([1, 0, -1]))
        self.assertAlmostEqual(yaw, 45.0, places=3)
        self.assertAlmostEqual(pitch, 0.0, places=3)

        # Looking left: -X, -Z
        yaw, pitch = vector_to_yaw_pitch_deg(np.array([-1, 0, -1]))
        self.assertAlmostEqual(yaw, -45.0, places=3)
        self.assertAlmostEqual(pitch, 0.0, places=3)

        # Looking up: -Y, -Z
        yaw, pitch = vector_to_yaw_pitch_deg(np.array([0, -1, -1]))
        self.assertAlmostEqual(yaw, 0.0, places=3)
        self.assertAlmostEqual(pitch, 45.0, places=3)

        # Looking down: +Y, -Z
        yaw, pitch = vector_to_yaw_pitch_deg(np.array([0, 1, -1]))
        self.assertAlmostEqual(yaw, 0.0, places=3)
        self.assertAlmostEqual(pitch, -45.0, places=3)

    def test_features_and_fitting(self):
        """Test polynomial feature construction and least squares model fitting."""
        feats = compute_calibration_features(10.0, -5.0)
        self.assertEqual(len(feats), 6)
        expected = np.array([10.0, -5.0, 100.0, -50.0, 25.0, 1.0])
        np.testing.assert_allclose(feats, expected)

        # Create synthetic linear gaze observations
        # x_norm = 0.5 + 0.02 * yaw
        # y_norm = 0.5 - 0.02 * pitch
        training_obs = []
        for tx, ty in CALIBRATION_TARGETS:
            yaw = (tx - 0.5) / 0.02
            pitch = -(ty - 0.5) / 0.02
            training_obs.append((yaw, pitch, (tx, ty)))

        model, train_rmse = fit_gaze_model(training_obs)
        self.assertEqual(model.shape, (6, 2))
        self.assertLess(train_rmse, 1e-4)

    def test_quality_classification(self):
        """Test prototype engineering quality gates."""
        self.assertEqual(classify_calibration_quality(45.0), "EXCELLENT")
        self.assertEqual(classify_calibration_quality(60.0), "EXCELLENT")
        self.assertEqual(classify_calibration_quality(82.0), "GOOD")
        self.assertEqual(classify_calibration_quality(110.0), "GOOD")
        self.assertEqual(classify_calibration_quality(135.0), "FAIR")
        self.assertEqual(classify_calibration_quality(180.0), "FAIR")
        self.assertEqual(classify_calibration_quality(220.0), "POOR")

    def test_regularized_model_selection_uses_training_targets_only(self):
        observations = []
        for tx, ty in CALIBRATION_TARGETS:
            observations.append(((tx - 0.5) / 0.02, -(ty - 0.5) / 0.02, (tx, ty)))
        model, rmse, selection = select_regularized_gaze_model(observations)
        self.assertEqual(model.shape, (10, 2))
        self.assertLess(rmse, 1e-3)
        self.assertIn("ridge_0", selection.candidate_median_errors_norm)
        self.assertGreaterEqual(selection.selected_ridge_alpha, 0.0)

    def test_cubic_model_prediction_and_legacy_quadratic_prediction(self):
        observations = []
        for tx, ty in CALIBRATION_TARGETS:
            yaw = (tx - 0.5) / 0.02
            pitch = -(ty - 0.5) / 0.02
            observations.append((yaw, pitch, (tx, ty)))
        cubic, _, _ = select_regularized_gaze_model(observations)
        self.assertEqual(cubic.shape, (10, 2))
        x, y, _, _ = predict_screen_coordinates(0.0, 0.0, cubic, 1920, 1080)
        self.assertAlmostEqual(x, 960, delta=2)
        self.assertAlmostEqual(y, 540, delta=2)

        legacy, _ = fit_gaze_model(observations)
        self.assertEqual(legacy.shape, (6, 2))
        x, y, _, _ = predict_screen_coordinates(0.0, 0.0, legacy, 1920, 1080)
        self.assertAlmostEqual(x, 960, delta=2)
        self.assertAlmostEqual(y, 540, delta=2)

    def test_evaluation_metrics(self):
        """Test comprehensive evaluation metrics calculation on independent validation targets."""
        W, H = 1920, 1080
        # Synthetic model mapping
        # x_norm = 0.5 + 0.02 * yaw
        # y_norm = 0.5 - 0.02 * pitch
        training_obs = []
        for tx, ty in CALIBRATION_TARGETS:
            yaw = (tx - 0.5) / 0.02
            pitch = -(ty - 0.5) / 0.02
            training_obs.append((yaw, pitch, (tx, ty)))
        model, _ = fit_gaze_model(training_obs)

        # Synthetic validation observations with known injected error: +40px horizontal, +30px vertical
        # Error = sqrt(40^2 + 30^2) = 50px
        dx_norm = 40.0 / (W - 1)
        dy_norm = 30.0 / (H - 1)

        val_obs = []
        for tx, ty in VALIDATION_TARGETS:
            # Distort yaw and pitch so predicted screen position has the offset
            true_yaw = (tx - 0.5) / 0.02
            true_pitch = -(ty - 0.5) / 0.02

            distorted_yaw = (tx + dx_norm - 0.5) / 0.02
            distorted_pitch = -(ty + dy_norm - 0.5) / 0.02

            # Simulate 10 samples per target around the mean
            samples = [(distorted_yaw, distorted_pitch) for _ in range(10)]
            val_obs.append((distorted_yaw, distorted_pitch, (tx, ty), samples))

        metrics = evaluate_calibration(model, val_obs, W, H, viewing_distance_mm=600.0)

        self.assertEqual(metrics.num_targets, 6)
        self.assertEqual(metrics.num_samples, 60)
        self.assertAlmostEqual(metrics.median_px, 50.0, delta=1.0)
        self.assertAlmostEqual(metrics.mae_px, 50.0, delta=1.0)
        self.assertAlmostEqual(metrics.rmse_px, 50.0, delta=1.0)
        self.assertAlmostEqual(metrics.p95_px, 50.0, delta=1.0)
        self.assertAlmostEqual(metrics.horizontal_mae_px, 40.0, delta=1.0)
        self.assertAlmostEqual(metrics.vertical_mae_px, 30.0, delta=1.0)
        self.assertEqual(metrics.status, "EXCELLENT")
        self.assertGreater(metrics.normalized_median_error, 0.0)
        self.assertGreater(metrics.angular_median_error_deg, 0.0)
        self.assertEqual(len(metrics.spatial_errors), 6)

    def test_profile_v2_roundtrip(self):
        """Test serialization and deserialization of Version 2 profile."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            temp_path = tf.name

        try:
            W, H = 1920, 1080
            model = np.ones((6, 2), dtype=float)
            training_obs = [(0.0, 0.0, (0.5, 0.5))] * 6
            dummy_model, _ = fit_gaze_model([(tx/0.02, ty/0.02, (tx, ty)) for tx, ty in CALIBRATION_TARGETS[:6]])
            val_obs = [(tx/0.02, ty/0.02, (tx, ty), [(tx/0.02, ty/0.02)] * 5) for tx, ty in VALIDATION_TARGETS]
            metrics = evaluate_calibration(dummy_model, val_obs, W, H)

            saved = save_calibration_profile_v2(temp_path, W, H, dummy_model, 15.2, metrics)
            self.assertTrue(saved)

            loaded = load_calibration_profile_v2(temp_path, W, H)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["version"], 2)
            self.assertTrue(loaded["is_valid"])
            self.assertIn("validation", loaded)
            self.assertIn("Status: EXCELLENT", loaded["status_text"])
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_robust_target_estimate_rejects_single_bad_frame(self):
        samples = [(10.0 + 0.01 * index, -4.0 - 0.01 * index) for index in range(30)]
        samples.append((80.0, 60.0))
        estimate = robust_target_estimate(samples, minimum_inliers=25)
        self.assertIsNotNone(estimate)
        yaw, pitch, inliers = estimate
        self.assertAlmostEqual(yaw, 10.14, delta=0.03)
        self.assertAlmostEqual(pitch, -4.14, delta=0.03)
        self.assertEqual(len(inliers), 30)

    def test_robust_target_estimate_rejects_unstable_target(self):
        unstable = [(float(index), float(-index)) for index in range(30)]
        self.assertIsNone(robust_target_estimate(unstable, minimum_inliers=25))


if __name__ == "__main__":
    unittest.main()
