# Neurodevelopmental Screening System Foundation

## Purpose and boundary

This project is a privacy-aware research prototype for collecting camera-derived visual-engagement measures. It is **not** a diagnostic device and it does not classify ASD, ADHD, developmental delay, or other conditions. No automated clinical risk or referral decision is enabled.

The current implementation provides:

- Local camera gaze estimation and a 9-point gaze calibration.
- A raw webcam view, annotated gaze view, and 3D debug view.
- An opt-in, local `T`-key session that retains derived gaze metrics only.
- A `V`-key standardized nine-target visual-attention task with per-target latency and on-target engagement summaries.
- Atomic JSON reports with no captured video, face landmarks, child name, age, or identifier.
- Data-quality gates for duration, tracked-frame ratio, and usable gaze samples.

## Derived metrics

The session report measures tracking availability, gaze speed, fixation ratio, gaze dispersion, spatial entropy, region transitions, and the longest interval without a detected face. These are descriptive research features, not validated biomarkers.

## Intended research workflow

1. Obtain documented guardian consent and institutional/privacy approval before starting a session.
2. Use a standardized, age-appropriate visual task with a stable phone or display position.
3. Complete eye and 9-point calibration; repeat if calibration quality is poor.
4. Start the local metrics session with `T`, administer the task, then press `T` again to save the derived report.
5. Review data quality before analysis; do not interpret low- or insufficient-quality recordings.
6. Store reports under an approved retention and access-control policy.

## Requirements before clinical use

Clinical screening or referral requires all of the following, none of which is implemented here:

- A pre-registered study and ethics/IRB approval where applicable.
- Representative, consented training and validation data with clinician-established ground truth.
- Separate training, validation, and external test cohorts; performance stratified by age, language, sex, skin tone, device, vision correction, and accessibility needs.
- Calibration, sensitivity/specificity, false-positive/false-negative, fairness, and uncertainty thresholds agreed with clinicians.
- A human-in-the-loop referral workflow, clear caregiver communication, and an adverse-event/escalation process.
- Security review, encryption at rest/in transit, deletion controls, access logging, and applicable privacy-law compliance.
- Medical-device/regulatory assessment for every intended deployment region.

## Smartphone path

The present implementation is a desktop OpenCV prototype. A smartphone product should move camera processing on-device, use the front camera, guide device distance/lighting/pose, store no frames by default, and send only consented, encrypted derived metrics. Its task UI should be age-appropriate and co-designed with clinicians, caregivers, and accessibility experts.
