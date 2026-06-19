"""
Risk Engine — Multi-Dimensional Neurodevelopmental Risk Assessment
==================================================================
Computes risk scores across four dimensions:
  1. ASD Risk — social attention, face preference, joint attention
  2. ADHD Risk — sustained attention, inhibition, gaze hyperactivity
  3. Cognitive Delay Risk — pursuit accuracy, visual search, pattern recognition
  4. Social Communication Risk — gaze following, joint attention, eye contact

Produces an overall risk level: Low / Moderate / Elevated
"""

import numpy as np
from collections import OrderedDict
from config import RISK_WEIGHTS, RISK_THRESHOLDS, RISK_COLORS, AGE_GROUP_NORMS


class RiskEngine:
    """
    Multi-dimensional risk scoring engine.

    Usage:
        engine = RiskEngine(age_group="5-7")
        scores = engine.compute_risk_scores(all_features)
        level = engine.get_overall_risk_level(scores)
    """

    def __init__(self, age_group="5-7"):
        self.age_group = age_group
        self.norms = AGE_GROUP_NORMS.get(age_group, AGE_GROUP_NORMS["5-7"])

    def compute_risk_scores(self, task_features):
        """
        Compute risk scores from aggregated task features.

        Parameters
        ----------
        task_features : dict
            Flat dict of {task_id__feature_name: value} from TaskManager

        Returns
        -------
        dict with sub-scores and overall risk
        """
        scores = OrderedDict()

        # ── ASD Risk ──────────────────────────────────────────────────
        asd_indicators = self._compute_asd_indicators(task_features)
        scores["asd"] = self._weighted_score(asd_indicators, "asd")

        # ── ADHD Risk ─────────────────────────────────────────────────
        adhd_indicators = self._compute_adhd_indicators(task_features)
        scores["adhd"] = self._weighted_score(adhd_indicators, "adhd")

        # ── Cognitive Delay Risk ──────────────────────────────────────
        cognitive_indicators = self._compute_cognitive_indicators(task_features)
        scores["cognitive"] = self._weighted_score(cognitive_indicators, "cognitive")

        # ── Social Communication Risk ─────────────────────────────────
        social_indicators = self._compute_social_indicators(task_features)
        scores["social_communication"] = self._weighted_score(
            social_indicators, "social_communication"
        )

        # ── Overall Risk ──────────────────────────────────────────────
        all_scores = [s for s in scores.values() if s is not None]
        scores["overall"] = float(np.mean(all_scores)) if all_scores else 0.0

        # ── Risk Levels ───────────────────────────────────────────────
        scores["levels"] = {
            k: self._score_to_level(v)
            for k, v in scores.items() if k != "levels"
        }

        # ── Detailed indicators (for report) ──────────────────────────
        scores["indicators"] = {
            "asd": asd_indicators,
            "adhd": adhd_indicators,
            "cognitive": cognitive_indicators,
            "social_communication": social_indicators,
        }

        return scores

    def get_overall_risk_level(self, scores):
        """Return the overall risk level string."""
        return scores.get("levels", {}).get("overall", "low")

    def get_risk_color(self, level):
        """Return the color for a risk level."""
        return RISK_COLORS.get(level, "#888888")

    def get_recommendations(self, scores):
        """Generate clinical recommendations based on risk scores."""
        recommendations = []
        levels = scores.get("levels", {})

        if levels.get("overall") == "elevated":
            recommendations.append({
                "priority": "HIGH",
                "text": "Comprehensive neurodevelopmental evaluation recommended. "
                        "Multiple screening indicators suggest elevated risk.",
                "icon": "🔴",
            })

        if levels.get("asd") in ("moderate", "elevated"):
            recommendations.append({
                "priority": "HIGH" if levels["asd"] == "elevated" else "MEDIUM",
                "text": "ASD-specific evaluation recommended. Atypical patterns observed in "
                        "social attention and face preference tasks.",
                "icon": "🟡" if levels["asd"] == "moderate" else "🔴",
            })

        if levels.get("adhd") in ("moderate", "elevated"):
            recommendations.append({
                "priority": "HIGH" if levels["adhd"] == "elevated" else "MEDIUM",
                "text": "ADHD screening recommended. Indicators of attention regulation "
                        "difficulties observed in sustained attention and inhibition tasks.",
                "icon": "🟡" if levels["adhd"] == "moderate" else "🔴",
            })

        if levels.get("cognitive") in ("moderate", "elevated"):
            recommendations.append({
                "priority": "HIGH" if levels["cognitive"] == "elevated" else "MEDIUM",
                "text": "Cognitive assessment recommended. Visual tracking and pattern "
                        "recognition performance below expected range.",
                "icon": "🟡" if levels["cognitive"] == "moderate" else "🔴",
            })

        if levels.get("social_communication") in ("moderate", "elevated"):
            recommendations.append({
                "priority": "MEDIUM",
                "text": "Social communication evaluation recommended. Joint attention and "
                        "gaze following metrics warrant further assessment.",
                "icon": "🟡",
            })

        if levels.get("overall") == "low":
            recommendations.append({
                "priority": "LOW",
                "text": "All screening indicators within expected ranges. Routine "
                        "developmental monitoring recommended.",
                "icon": "🟢",
            })

        return recommendations

    # ─── Private: Indicator Computation ───────────────────────────────────

    def _compute_asd_indicators(self, features):
        """Extract ASD-relevant indicators from task features."""
        indicators = OrderedDict()

        # Face preference (from face_preference task)
        fp_ratio = features.get("face_preference__face_preference_ratio", 0.5)
        # Lower face preference → higher risk
        norm_low, norm_high = self.norms.get("face_preference_ratio", (0.55, 0.80))
        indicators["face_preference_ratio"] = self._normalize_inverse(
            fp_ratio, norm_low, norm_high
        )

        # Eye region fixation (from social_scene task)
        eye_ratio = features.get("social_scene__eye_region_fixation_ratio", 0.3)
        # Lower eye fixation → higher risk
        indicators["eye_region_fixation_ratio"] = self._normalize_inverse(
            eye_ratio, 0.15, 0.40
        )

        # Joint attention latency (from joint_attention task)
        ja_latency = features.get(
            "joint_attention__roi_target_first_fix_latency_ms", 500
        )
        # Higher latency → higher risk
        indicators["joint_attention_latency"] = self._normalize_direct(
            ja_latency, 200, 2000
        )

        # Social scene face dwell (from social_scene task)
        face_dwell = features.get("social_scene__face_dwell_ratio", 0.3)
        indicators["social_scene_face_dwell"] = self._normalize_inverse(
            face_dwell, 0.2, 0.5
        )

        # Repetitive scan pattern
        rep_idx = features.get("social_scene__repetitive_scan_index", 0.1)
        indicators["repetitive_scan_index"] = self._normalize_direct(
            rep_idx, 0.05, 0.5
        )

        # Gaze range restriction
        gaze_range = features.get("face_preference__gaze_range_ratio", 0.3)
        indicators["gaze_range_restriction"] = self._normalize_inverse(
            gaze_range, 0.1, 0.5
        )

        return indicators

    def _compute_adhd_indicators(self, features):
        """Extract ADHD-relevant indicators."""
        indicators = OrderedDict()

        # Sustained attention score (from sustained_attention task)
        omission = features.get("sustained_attention__sustained_omission_rate", 0.2)
        indicators["sustained_attention_score"] = self._normalize_direct(
            omission, 0.0, 0.8
        )

        # Anti-saccade errors
        as_error = features.get("anti_saccade__anti_saccade_error", 0)
        indicators["anti_saccade_error_rate"] = float(as_error) * 80  # scale to 0-80

        # Fixation duration variance (high variance → ADHD indicator)
        fix_var = features.get("sustained_attention__fix_duration_std_ms", 100)
        indicators["fixation_duration_variance"] = self._normalize_direct(
            fix_var, 30, 300
        )

        # Omission rate
        indicators["omission_error_rate"] = self._normalize_direct(
            omission, 0.0, 0.6
        )

        # Commission rate
        commission = features.get("sustained_attention__sustained_commission_rate", 0.1)
        indicators["commission_error_rate"] = self._normalize_direct(
            commission, 0.0, 0.5
        )

        # Hyperkinetic gaze (excessive saccades)
        sacc_rate = features.get("sustained_attention__fixation_rate_per_sec", 2.0)
        indicators["hyperkinetic_gaze_index"] = self._normalize_direct(
            sacc_rate, 1.0, 6.0
        )

        return indicators

    def _compute_cognitive_indicators(self, features):
        """Extract cognitive delay indicators."""
        indicators = OrderedDict()

        # Smooth pursuit accuracy
        pursuit_acc = features.get("smooth_pursuit__pursuit_accuracy_mean_px", 100)
        indicators["smooth_pursuit_accuracy"] = self._normalize_direct(
            pursuit_acc, 20, 200
        )

        # Visual search efficiency (pattern recognition)
        fix_count = features.get("pattern_recognition__fix_count", 10)
        # More fixations needed → less efficient → higher risk
        indicators["visual_search_efficiency"] = self._normalize_direct(
            fix_count, 3, 30
        )

        # Pattern recognition (time to find target)
        target_latency = features.get(
            "pattern_recognition__roi_target_first_fix_latency_ms", 2000
        )
        indicators["pattern_recognition_score"] = self._normalize_direct(
            target_latency if target_latency > 0 else 2000, 200, 5000
        )

        # Fixation count deviation from norm
        norm_low, norm_high = self.norms.get("fixation_count_per_task", (10, 30))
        avg_fix = np.mean([
            features.get(f"{t}__fix_count", 15)
            for t in ["smooth_pursuit", "pattern_recognition"]
        ])
        deviation = abs(avg_fix - (norm_low + norm_high) / 2)
        indicators["fixation_count_deviation"] = self._normalize_direct(
            deviation, 0, 20
        )

        # Processing speed
        pursuit_gain = features.get("smooth_pursuit__pursuit_gain", 0.8)
        indicators["processing_speed_index"] = self._normalize_inverse(
            pursuit_gain, 0.5, 1.2
        )

        return indicators

    def _compute_social_indicators(self, features):
        """Extract social communication indicators."""
        indicators = OrderedDict()

        # Joint attention
        ja_correct = features.get(
            "joint_attention__roi_target_dwell_ratio", 0.3
        )
        indicators["joint_attention_score"] = self._normalize_inverse(
            ja_correct, 0.1, 0.6
        )

        # Gaze following accuracy
        ja_latency = features.get(
            "joint_attention__roi_target_first_fix_latency_ms", 500
        )
        indicators["gaze_following_accuracy"] = self._normalize_direct(
            ja_latency if ja_latency > 0 else 1000, 100, 2000
        )

        # Social scene scanning pattern
        face_fix = features.get("social_scene__roi_face_fix_count", 3)
        total_fix = features.get("social_scene__fix_count", 10)
        ratio = face_fix / total_fix if total_fix > 0 else 0.3
        indicators["social_scene_scanning_pattern"] = self._normalize_inverse(
            ratio, 0.2, 0.6
        )

        # Eye contact duration (from face preference)
        eye_dwell = features.get("face_preference__eye_region_fixation_ratio", 0.2)
        indicators["eye_contact_duration"] = self._normalize_inverse(
            eye_dwell, 0.05, 0.30
        )

        # Face preference ratio
        fp = features.get("face_preference__face_preference_ratio", 0.5)
        indicators["face_preference_ratio"] = self._normalize_inverse(
            fp, 0.4, 0.75
        )

        return indicators

    # ─── Private: Scoring Utilities ───────────────────────────────────────

    def _weighted_score(self, indicators, category):
        """Compute weighted risk score from indicators."""
        weights = RISK_WEIGHTS.get(category, {})
        total_weight = 0.0
        weighted_sum = 0.0

        for ind_name, ind_value in indicators.items():
            w = weights.get(ind_name, 0.1)
            weighted_sum += ind_value * w
            total_weight += w

        if total_weight == 0:
            return 0.0

        return min(100.0, max(0.0, weighted_sum / total_weight))

    @staticmethod
    def _normalize_direct(value, low, high):
        """
        Normalize so that higher values → higher risk (0-100 scale).
        Values at/below 'low' → 0, at/above 'high' → 100.
        """
        if high <= low:
            return 50.0
        normalized = (value - low) / (high - low)
        return min(100.0, max(0.0, normalized * 100.0))

    @staticmethod
    def _normalize_inverse(value, low, high):
        """
        Normalize so that lower values → higher risk (0-100 scale).
        Values at/above 'high' → 0 (low risk), at/below 'low' → 100 (high risk).
        """
        if high <= low:
            return 50.0
        normalized = 1.0 - (value - low) / (high - low)
        return min(100.0, max(0.0, normalized * 100.0))

    @staticmethod
    def _score_to_level(score):
        """Convert a numeric score to a risk level string."""
        if score is None:
            return "unknown"
        for level, (lo, hi) in RISK_THRESHOLDS.items():
            if lo <= score < hi:
                return level
        return "elevated" if score >= 60 else "low"
