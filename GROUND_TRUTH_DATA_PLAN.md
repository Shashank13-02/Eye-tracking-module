# NeuroGaze ground-truth plan

## Boundary

Do not train an ASD, cognitive-delay, or “risk” classifier from `asd.csv`,
`Autism-Child-Data.csv`, calibration results, or task performance alone. Those
files do not link the same child’s NeuroGaze protocol to an independent clinical or
developmental assessment.

This project should first use public data only for feature-pipeline experiments,
schema mapping, and benchmark replication. It must not claim that a model trained on
another device, country, stimulus battery, age range, or label process is valid for
this webcam protocol.

## Recommended data sources

1. **NIMH Data Archive collection 3127 — Discovering Eye Tracking Biomarkers of ASD with Diagnostic and Prognostic Power.** This is the closest age/task match: its description covers toddlers aged 12–36 months and includes eye-tracking experiments plus ADOS-2 Toddler Module, Mullen, and Vineland data. Access is governed by NDA permissions.
   - https://nda.nih.gov/edit_collection.html?id=3127

2. **NIMH Data Archive collection 2077 — Gaze Modification Strategies for Toddlers with ASD.** It includes eye-tracking experiments, ADOS/ADOS-2, Mullen, and Vineland data and targets social attention in toddlers. Access is governed by NDA permissions.
   - https://nda.nih.gov/edit_collection.html?id=2077

3. **NIMH Data Archive collection 2288 — Autism Biomarkers Consortium for Clinical Trials (ABC-CT).** It provides a stronger multi-site reference for preschool children (3–5 years) and older children, with longitudinal eye-tracking/clinical measures and detailed methods documents for qualified researchers.
   - https://nda.nih.gov/edit_collection.html?id=2288

4. **Figshare Eye-Tracking Dataset to Support Research on Autism Spectrum Disorder.** Public secondary-analysis data with 25 CSV experiment files and participant metadata. Use it only for parser/feature-pipeline benchmarks: the study’s device, stimuli, ages, and labels do not match NeuroGaze.
   - https://doi.org/10.6084/m9.figshare.20113592

5. **Saliency4ASD on Zenodo.** An open dataset of fixation maps and scanpaths from 14 autistic and 14 control children viewing natural scenes. Use it for visual-attention algorithm experiments, not clinical modelling or the present task battery.
   - https://doi.org/10.5281/zenodo.13960426

## Minimum local collection contract

Before recruitment, have a qualified clinician and ethics/IRB process approve:

- Protocol and stimulus version per age band, including stop/comfort criteria.
- Guardian consent and an opaque study pseudonym—not name, date of birth, or hospital ID.
- Independent assessment instrument, assessment date, and assessor role.
- Data retention, encryption, access, deletion, and adverse-event procedures.
- Participant-level split assignment **before** model fitting.

Record one row per session in `ground_truth_manifest_template.csv`. Copy it outside
version control, fill it with approved study records, then run the manifest validator
before model development. A child must appear in exactly one of `train`,
`validation`, or `test`, even if they have many sessions.

## Required evidence before modelling

Use only assessments completed independently from the gaze pipeline as labels. Keep
an untouched external test cohort. Report calibration, missing-data rates,
sensitivity/specificity, confidence intervals, and performance stratified by age
band, sex, language, vision correction, device, and site. A failed or insufficient
gaze session must remain “insufficient data,” never be converted into a risk label.
