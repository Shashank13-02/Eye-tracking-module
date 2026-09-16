"""Local, non-diagnostic M-CHAT-R/F screening-support interface.

Run with: py -3 asd_screening_interface.py
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from asd_screening import MCHAT_QUESTIONS, MChatResult, score_follow_up, score_mchat, storage_token
from qchat_research import QCHAT_ITEMS, QChatResearchResult, score_qchat_research
from research_observations import calibration_summary, format_observation, load_research_observation
from screening_pdf_report import create_qchat_research_pdf, create_screening_pdf
from workflow_export import build_final_workflow_export, compact_webcam_parameters


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "asd.csv"
REPORT_DIRECTORY = ROOT / "screening_reports"
SOURCE_PDF = ROOT / "ASD or Autism Screening" / "Modified Checklist for Autism in Toddlers, Revised, with Follow-Up (M-CHAT-R-F).pdf"
CALIBRATION_PROFILE = ROOT / "gaze_calibration.json"


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0, background="#f6f7fb")
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.body = ttk.Frame(self.canvas, padding=18)
        self.body.bind("<Configure>", lambda _: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.canvas.bind("<Configure>", lambda event: self.canvas.itemconfigure(1, width=event.width))
        # A wheel event is delivered to the widget under the pointer, often a
        # label/radio button inside ``body`` rather than the canvas itself.  The
        # application-level bindings below route it back to this tab only when
        # the pointer is over one of this frame's descendants.
        self.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self.bind_all("<Button-4>", self._on_mousewheel, add="+")
        self.bind_all("<Button-5>", self._on_mousewheel, add="+")

    def _contains_pointer_target(self, widget: tk.Misc | None) -> bool:
        while widget is not None:
            if widget is self.body or widget is self.canvas:
                return True
            widget = widget.master
        return False

    def _on_mousewheel(self, event: tk.Event) -> str | None:
        target = self.winfo_containing(event.x_root, event.y_root)
        if not self._contains_pointer_target(target):
            return None
        # Preserve normal scrolling inside multi-line note/report fields.
        if isinstance(target, tk.Text):
            return None
        if getattr(event, "num", None) == 4:
            steps = -1
        elif getattr(event, "num", None) == 5:
            steps = 1
        else:
            delta = getattr(event, "delta", 0)
            steps = -max(1, abs(delta) // 120) if delta > 0 else max(1, abs(delta) // 120)
        self.canvas.yview_scroll(steps, "units")
        return "break"


class ScreeningApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("NeuroGaze - Local Toddler Screening Support")
        self.geometry("1040x760")
        self.minsize(860, 620)
        self.configure(background="#f6f7fb")
        self.age_var = tk.StringVar()
        self.scored_age_months: int | None = None
        self.child_id_var = tk.StringVar()
        self.session_id_var = tk.StringVar()
        self.consent_var = tk.BooleanVar(value=False)
        self.answer_vars = {item: tk.StringVar(value="") for item in range(1, 21)}
        self.result: MChatResult | None = None
        self.follow_up_vars: dict[int, tk.StringVar] = {}
        self.follow_up_record: dict | None = None
        self.review_record: dict | None = None
        self.clinician_name_var = tk.StringVar()
        self.clinician_credentials_var = tk.StringVar()
        self.review_plan_var = tk.StringVar(value="Select a local follow-up plan")
        self.review_attested_var = tk.BooleanVar(value=False)
        self.research_observation: dict | None = None
        self.research_calibration: dict | None = None
        self.attached_webcam_reports: list[dict] = []
        self.qchat_vars = {item: tk.StringVar(value="") for item in range(1, 11)}
        self.qchat_result: QChatResearchResult | None = None
        self._build_style()
        self._build_layout()
        self.age_var.trace_add("write", self._update_age_guidance)
        self.bind_all("<Control-s>", lambda _: self._save_report())
        self.bind_all("<Control-r>", lambda _: self._resume_local_session())

    def _build_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Segoe UI", 21, "bold"), foreground="#102a43", background="#f6f7fb")
        style.configure("Section.TLabel", font=("Segoe UI", 13, "bold"), foreground="#102a43", background="#f6f7fb")
        style.configure("Body.TLabel", font=("Segoe UI", 12), foreground="#243b53", background="#f6f7fb")
        style.configure("Primary.TButton", font=("Segoe UI", 12, "bold"), padding=(16, 10))
        style.configure("TButton", font=("Segoe UI", 11), padding=(10, 7))
        style.configure("TRadiobutton", font=("Segoe UI", 12))
        style.configure("TCheckbutton", font=("Segoe UI", 12))
        style.configure("TEntry", font=("Segoe UI", 12))
        style.configure("TCombobox", font=("Segoe UI", 12))

    def _build_layout(self) -> None:
        header = ttk.Frame(self, padding=(22, 18, 22, 10))
        header.pack(fill="x")
        ttk.Label(header, text="Toddler Developmental Screening Support", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Local M-CHAT-R/F scoring support. This is a screening aid, not a diagnosis, risk prediction, or replacement for clinical assessment.",
            style="Body.TLabel", wraplength=960,
        ).pack(anchor="w", pady=(5, 0))

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=18, pady=(0, 18))
        self.questionnaire_tab = ScrollableFrame(notebook)
        self.follow_up_tab = ScrollableFrame(notebook)
        self.research_tab = ScrollableFrame(notebook)
        self.qchat_tab = ScrollableFrame(notebook)
        self.review_tab = ScrollableFrame(notebook)
        self.dataset_tab = ScrollableFrame(notebook)
        notebook.add(self.questionnaire_tab, text="  Questionnaire  ")
        notebook.add(self.follow_up_tab, text="  Follow-Up  ")
        notebook.add(self.research_tab, text="  Research observations  ")
        notebook.add(self.qchat_tab, text="  Q-CHAT research form  ")
        notebook.add(self.review_tab, text="  Clinician review  ")
        notebook.add(self.dataset_tab, text="  Dataset context  ")
        self._build_questionnaire(self.questionnaire_tab.body)
        self._build_follow_up(self.follow_up_tab.body)
        self._build_research_observations(self.research_tab.body)
        self._build_qchat_research(self.qchat_tab.body)
        self._build_review(self.review_tab.body)
        self._build_dataset_context(self.dataset_tab.body)

    def _build_questionnaire(self, body: ttk.Frame) -> None:
        ttk.Label(body, text="Before you begin", style="Section.TLabel").pack(anchor="w")
        ttk.Label(
            body,
            text="Answer for the child's usual behavior. The supplied English instrument is the only language available here; no machine translation is used. The instrument is intended for toddlers aged 16-30 months. Do not enter names or other direct identifiers.",
            style="Body.TLabel", wraplength=900,
        ).pack(anchor="w", pady=(4, 12))
        fields = ttk.Frame(body)
        fields.pack(fill="x", pady=(0, 12))
        ttk.Label(fields, text="Age in months:", style="Body.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Entry(fields, width=8, textvariable=self.age_var).grid(row=0, column=1, padx=(8, 24), sticky="w")
        ttk.Label(fields, text="Child local ID:", style="Body.TLabel").grid(row=0, column=2, sticky="w")
        ttk.Entry(fields, width=20, textvariable=self.child_id_var).grid(row=0, column=3, padx=(8, 0), sticky="w")
        ttk.Label(fields, text="Session ID:", style="Body.TLabel").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(fields, width=20, textvariable=self.session_id_var).grid(row=1, column=1, padx=(8, 24), pady=(8, 0), sticky="w")
        ttk.Button(fields, text="Resume local session (Ctrl+R)", command=self._resume_local_session).grid(row=1, column=2, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Label(
            body,
            text="Use pseudonymous local IDs only: 3-64 letters, numbers, hyphens, or underscores. Do not use a child name, date of birth, hospital ID, or other direct identifier.",
            style="Body.TLabel", wraplength=900,
        ).pack(anchor="w", pady=(8, 10))
        self.age_guidance = ttk.Label(
            body,
            text="Enter age in whole months. M-CHAT-R/F scoring in this interface is limited to 16-30 months.",
            style="Body.TLabel", wraplength=900,
        )
        self.age_guidance.pack(anchor="w", pady=(0, 10))
        ttk.Checkbutton(
            body,
            text="I confirm I have appropriate permission to enter this sensitive developmental screening information locally.",
            variable=self.consent_var,
        ).pack(anchor="w", pady=(0, 18))
        ttk.Label(body, text="Caregiver concerns (optional; avoid names and direct identifiers)", style="Body.TLabel").pack(anchor="w")
        self.caregiver_concerns = tk.Text(body, height=3, wrap="word", font=("Segoe UI", 12))
        self.caregiver_concerns.pack(fill="x", pady=(4, 18))

        for item, question in enumerate(MCHAT_QUESTIONS, start=1):
            card = ttk.LabelFrame(body, text=f"Question {item}", padding=10)
            card.pack(fill="x", pady=5)
            ttk.Label(card, text=question, style="Body.TLabel", wraplength=840, justify="left").pack(anchor="w")
            controls = ttk.Frame(card)
            controls.pack(anchor="w", pady=(7, 0))
            ttk.Radiobutton(controls, text="Yes", variable=self.answer_vars[item], value="yes").pack(side="left", padx=(0, 18))
            ttk.Radiobutton(controls, text="No", variable=self.answer_vars[item], value="no").pack(side="left")

        actions = ttk.Frame(body)
        actions.pack(fill="x", pady=18)
        ttk.Button(actions, text="Score first stage", style="Primary.TButton", command=self._score_first_stage).pack(side="left")
        ttk.Button(actions, text="Save local report (Ctrl+S)", command=self._save_report).pack(side="left", padx=(10, 0))
        ttk.Button(actions, text="Generate PDF report", command=self._generate_pdf_report).pack(side="left", padx=(10, 0))
        ttk.Button(actions, text="Export final workflow JSON", command=self._export_final_workflow_json).pack(side="left", padx=(10, 0))
        self.result_label = ttk.Label(actions, text="Complete all answers to score the screen.", style="Body.TLabel", wraplength=660)
        self.result_label.pack(side="left", padx=16)

    def _build_follow_up(self, body: ttk.Frame) -> None:
        ttk.Label(body, text="Official Follow-Up outcome entry", style="Section.TLabel").pack(anchor="w")
        ttk.Label(
            body,
            text=("For a medium first-stage score, administer the official M-CHAT-R/F interview from the supplied PDF with a qualified clinician. "
                  "Record only each final Pass/Fail item outcome here; this app does not recreate or modify the copyrighted interview flowcharts."),
            style="Body.TLabel", wraplength=900,
        ).pack(anchor="w", pady=(4, 14))
        self.follow_up_items = ttk.Frame(body)
        self.follow_up_items.pack(fill="x")
        self.follow_up_result = ttk.Label(body, text="Score the first stage to identify Follow-Up items.", style="Body.TLabel", wraplength=900)
        self.follow_up_result.pack(anchor="w", pady=(14, 8))
        ttk.Button(body, text="Score Follow-Up", style="Primary.TButton", command=self._score_follow_up).pack(anchor="w")

    def _build_review(self, body: ttk.Frame) -> None:
        ttk.Label(body, text="Clinician review and local attestation", style="Section.TLabel").pack(anchor="w")
        ttk.Label(
            body,
            text=("Review the caregiver's first-stage answers, concerns, and any official Follow-Up outcomes. "
                  "This records a local workflow attestation only; it is not a legal or cryptographic signature and does not establish a diagnosis."),
            style="Body.TLabel", wraplength=900,
        ).pack(anchor="w", pady=(4, 14))
        self.review_summary = ttk.Label(body, text="Score the first stage before recording clinician review.", style="Body.TLabel", justify="left", wraplength=900)
        self.review_summary.pack(anchor="w", pady=(0, 14))

        fields = ttk.Frame(body)
        fields.pack(fill="x")
        ttk.Label(fields, text="Clinician name or local reviewer ID:", style="Body.TLabel").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(fields, textvariable=self.clinician_name_var, width=35).grid(row=0, column=1, sticky="w", padx=(12, 28), pady=5)
        ttk.Label(fields, text="Credentials / role:", style="Body.TLabel").grid(row=0, column=2, sticky="w", pady=5)
        ttk.Entry(fields, textvariable=self.clinician_credentials_var, width=24).grid(row=0, column=3, sticky="w", padx=(12, 0), pady=5)
        ttk.Label(fields, text="Local follow-up plan:", style="Body.TLabel").grid(row=1, column=0, sticky="w", pady=5)
        plan = ttk.Combobox(
            fields, textvariable=self.review_plan_var, state="readonly", width=50,
            values=(
                "Select a local follow-up plan",
                "Continue developmental surveillance",
                "Complete official Follow-Up interview",
                "Discuss developmental evaluation with caregiver",
                "Document caregiver concern and arrange follow-up",
            ),
        )
        plan.grid(row=1, column=1, columnspan=3, sticky="w", padx=(12, 0), pady=5)
        ttk.Label(body, text="Clinician review notes (optional; avoid unnecessary direct identifiers)", style="Body.TLabel").pack(anchor="w", pady=(16, 0))
        self.clinician_notes = tk.Text(body, height=6, wrap="word", font=("Segoe UI", 12))
        self.clinician_notes.pack(fill="x", pady=(4, 12))
        ttk.Checkbutton(
            body,
            text="I attest that I reviewed the recorded first-stage screen, caregiver concerns, and any Follow-Up outcomes shown above.",
            variable=self.review_attested_var,
        ).pack(anchor="w", pady=(0, 12))
        ttk.Button(body, text="Record clinician review", style="Primary.TButton", command=self._record_clinician_review).pack(anchor="w")
        self.review_status = ttk.Label(body, text="No clinician review has been recorded.", style="Body.TLabel", wraplength=900)
        self.review_status.pack(anchor="w", pady=(14, 0))

    def _build_research_observations(self, body: ttk.Frame) -> None:
        ttk.Label(body, text="Webcam research observations", style="Section.TLabel").pack(anchor="w")
        ttk.Label(
            body,
            text=("Attach one or more local webcam reports. The dashboard shows measurement quality and derived observations beside - but never scores them with - the questionnaire. "
                  "A poor or missing calibration makes gaze data unsuitable for interpretation."),
            style="Body.TLabel", wraplength=900,
        ).pack(anchor="w", pady=(4, 14))
        actions = ttk.Frame(body)
        actions.pack(fill="x")
        ttk.Button(actions, text="Attach local research report(s)", style="Primary.TButton", command=self._select_research_report).pack(side="left")
        self.research_attachment_label = ttk.Label(actions, text="No report attached.", style="Body.TLabel")
        self.research_attachment_label.pack(side="left", padx=12)
        self.research_text = tk.Text(body, height=23, wrap="word", font=("Segoe UI", 12), state="disabled", background="#ffffff", foreground="#243b53")
        self.research_text.pack(fill="both", expand=True, pady=(14, 0))
        self._show_research_text(
            "No webcam research report is attached.\n\n"
            "Questionnaire scoring and clinician review remain separate from gaze research observations."
        )

    def _build_qchat_research(self, body: ttk.Frame) -> None:
        ttk.Label(body, text="Q-CHAT-10-style dataset research form", style="Section.TLabel").pack(anchor="w")
        ttk.Label(
            body,
            text=("This local A-E form follows the scoring encoding described in the bundled Saudi toddler dataset documentation. "
                  "It is a separate research observation only: it does not alter the M-CHAT-R/F score and does not create an ASD risk, diagnosis, referral, or treatment decision."),
            style="Body.TLabel", wraplength=900,
        ).pack(anchor="w", pady=(4, 14))
        ttk.Label(
            body,
            text="Dataset reference age: 12-36 months. Use the same age entered on the Questionnaire tab.",
            style="Body.TLabel", wraplength=900,
        ).pack(anchor="w", pady=(0, 12))
        for item, (question, options) in enumerate(QCHAT_ITEMS, start=1):
            card = ttk.LabelFrame(body, text=f"Research question {item}", padding=10)
            card.pack(fill="x", pady=5)
            ttk.Label(card, text=question, style="Body.TLabel", wraplength=850, justify="left").pack(anchor="w")
            options_frame = ttk.Frame(card)
            options_frame.pack(anchor="w", pady=(7, 0))
            for code, label in zip("ABCDE", options):
                ttk.Radiobutton(options_frame, text=f"{code}. {label}", variable=self.qchat_vars[item], value=code).pack(side="left", padx=(0, 14))
        actions = ttk.Frame(body)
        actions.pack(fill="x", pady=18)
        ttk.Button(actions, text="Record research score", style="Primary.TButton", command=self._score_qchat_research).pack(side="left")
        ttk.Button(actions, text="Generate research PDF", command=self._generate_qchat_pdf_report).pack(side="left", padx=(10, 0))
        self.qchat_label = ttk.Label(actions, text="Complete all A-E responses to record the separate research score.", style="Body.TLabel", wraplength=680)
        self.qchat_label.pack(side="left", padx=16)

    def _show_research_text(self, text: str) -> None:
        self.research_text.configure(state="normal")
        self.research_text.delete("1.0", "end")
        self.research_text.insert("1.0", text)
        self.research_text.configure(state="disabled")

    def _select_research_report(self) -> None:
        filenames = filedialog.askopenfilenames(
            title="Attach local webcam research report(s)",
            initialdir=REPORT_DIRECTORY,
            filetypes=(("JSON reports", "*.json"),),
        )
        if not filenames:
            return
        attached: list[dict] = []
        presentation: list[str] = []
        try:
            for filename in filenames:
                path = Path(filename)
                raw_report = json.loads(path.read_text(encoding="utf-8"))
                attached.append(compact_webcam_parameters(raw_report, path.name))
                try:
                    observation = load_research_observation(path)
                    presentation.append(format_observation(observation, calibration_summary(CALIBRATION_PROFILE)))
                    self.research_observation = observation
                except ValueError:
                    presentation.append(f"Attached research report: {path.name}\nParameters will be preserved in the final workflow JSON.")
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            messagebox.showerror("Cannot load research report", str(error))
            return
        self.attached_webcam_reports = attached
        self.research_calibration = calibration_summary(CALIBRATION_PROFILE)
        self.research_attachment_label.configure(text=f"{len(attached)} report(s) attached.")
        self._show_research_text("\n\n".join(presentation))

    def _score_qchat_research(self) -> None:
        try:
            age = int(self.age_var.get())
            answers = {
                item: self.qchat_vars[item].get()
                for item in range(1, 11)
                if self.qchat_vars[item].get()
            }
            self.qchat_result = score_qchat_research(answers, age)
        except ValueError as error:
            messagebox.showerror("Cannot record research score", str(error))
            return
        self.qchat_label.configure(text=(
            f"Dataset-derived research score: {self.qchat_result.score}/10. {self.qchat_result.interpretation}"
        ))
        self._refresh_review_summary()

    def _build_dataset_context(self, body: ttk.Frame) -> None:
        ttk.Label(body, text="Bundled toddler dataset context", style="Section.TLabel").pack(anchor="w")
        ttk.Label(
            body,
            text=("This panel describes the bundled CSV only. It is not used to classify a child, generate a probability, or override a clinician. "
                  "Dataset labels are not interchangeable with this questionnaire or webcam measures."),
            style="Body.TLabel", wraplength=900,
        ).pack(anchor="w", pady=(4, 14))
        summary = self._dataset_summary()
        ttk.Label(body, text=summary, style="Body.TLabel", justify="left", wraplength=900).pack(anchor="w")
        ttk.Label(body, text="Source files: asd.csv and the local M-CHAT-R/F PDFs. Keep completed reports under an approved access and retention policy.", style="Body.TLabel", wraplength=900).pack(anchor="w", pady=(18, 0))

    def _dataset_summary(self) -> str:
        try:
            with DATASET_PATH.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            labelled = [row for row in rows if row.get("Diagnosed_ASD") in {"Yes", "No"}]
            positives = sum(row["Diagnosed_ASD"] == "Yes" for row in labelled)
            ages = [int(row["Age"]) for row in labelled if str(row.get("Age", "")).isdigit()]
            return (
                f"Records with an ASD label: {len(labelled)}\n"
                f"Label distribution in this file: Yes {positives}; No {len(labelled) - positives}\n"
                f"Recorded age range: {min(ages)}-{max(ages)} months\n\n"
                "These are dataset characteristics, not a clinical reference range or a prediction for the current questionnaire."
            )
        except (OSError, ValueError, KeyError):
            return "The bundled toddler CSV could not be read. Questionnaire scoring remains available without it."

    def _score_first_stage(self) -> None:
        if not self.consent_var.get():
            messagebox.showwarning("Permission required", "Confirm appropriate permission before scoring or saving sensitive screening information.")
            return
        try:
            age = int(self.age_var.get())
            answers = {
                item: self.answer_vars[item].get() == "yes"
                for item in range(1, 21)
                if self.answer_vars[item].get() in {"yes", "no"}
            }
            self.result = score_mchat(answers, age)
        except ValueError as error:
            messagebox.showerror("Cannot score questionnaire", str(error))
            return
        result = self.result
        self.scored_age_months = age
        self.follow_up_record = None
        self.review_record = None
        self.result_label.configure(text=(f"Score: {result.score}/20 | Category: {result.category.upper()} | Failed items: {', '.join(map(str, result.failed_items)) or 'none'}\n{result.next_step}"))
        self._render_follow_up_items()
        self._refresh_review_summary()
        if result.category == "medium":
            messagebox.showinfo("Follow-Up indicated", "Use the supplied official M-CHAT-R/F Follow-Up interview, then record the item outcomes in the Follow-Up tab.")

    def _update_age_guidance(self, *_: object) -> None:
        entered = self.age_var.get().strip()
        if not entered:
            self.age_guidance.configure(text="Enter age in whole months. M-CHAT-R/F scoring in this interface is limited to 16-30 months.")
            return
        try:
            age = int(entered)
        except ValueError:
            self.age_guidance.configure(text="Enter a whole number of months before scoring.")
            return
        if 16 <= age <= 30:
            self.age_guidance.configure(text=f"Age {age} months is within this interface's supported M-CHAT-R/F scoring range (16-30 months).")
        else:
            self.age_guidance.configure(text=(f"Age {age} months is outside this interface's supported M-CHAT-R/F scoring range (16-30 months). "
                                             "Scoring and report saving are blocked; use age-appropriate clinical guidance."))

    def _render_follow_up_items(self) -> None:
        for child in self.follow_up_items.winfo_children():
            child.destroy()
        self.follow_up_vars.clear()
        if self.result is None or self.result.category != "medium":
            message = "Follow-Up is shown only for a medium first-stage score (3-7)."
            ttk.Label(self.follow_up_items, text=message, style="Body.TLabel").pack(anchor="w")
            return
        for item in self.result.failed_items:
            row = ttk.Frame(self.follow_up_items)
            row.pack(fill="x", pady=4)
            ttk.Label(row, text=f"Item {item}: {MCHAT_QUESTIONS[item - 1]}", style="Body.TLabel", wraplength=700, justify="left").pack(side="left", fill="x", expand=True)
            variable = tk.StringVar(value="")
            self.follow_up_vars[item] = variable
            ttk.Radiobutton(row, text="Pass", variable=variable, value="pass").pack(side="left", padx=8)
            ttk.Radiobutton(row, text="Fail", variable=variable, value="fail").pack(side="left")

    def _score_follow_up(self) -> None:
        if self.result is None or self.result.category != "medium":
            self.follow_up_result.configure(text="A medium first-stage score is required before Follow-Up can be recorded.")
            return
        try:
            outcomes = {
                item: self.follow_up_vars[item].get() == "pass"
                for item in self.result.failed_items
                if self.follow_up_vars[item].get() in {"pass", "fail"}
            }
            failed_count, outcome = score_follow_up(outcomes, self.result.failed_items)
        except ValueError as error:
            self.follow_up_result.configure(text=str(error))
            return
        self.follow_up_record = {
            "failed_items": failed_count,
            "outcome": outcome,
            "item_outcomes": {str(item): self.follow_up_vars[item].get() for item in self.result.failed_items},
        }
        self.follow_up_result.configure(text=f"Follow-Up failed items: {failed_count}. {outcome}")
        self._refresh_review_summary()

    def _refresh_review_summary(self) -> None:
        if self.result is None:
            self.review_summary.configure(text="Score the first stage before recording clinician review.")
            return
        concerns = self.caregiver_concerns.get("1.0", "end").strip() or "None recorded"
        summary = (
            f"First-stage result: {self.result.category.upper()} ({self.result.score}/20).\n"
            f"Failed item numbers: {', '.join(map(str, self.result.failed_items)) or 'None'}.\n"
            f"Caregiver concerns: {concerns}\n"
        )
        if self.follow_up_record is not None:
            summary += f"Official Follow-Up outcome recorded: {self.follow_up_record['outcome']}"
        elif self.result.category == "medium":
            summary += "Official Follow-Up outcome: not yet recorded."
        else:
            summary += "Official Follow-Up outcome: not indicated by the first-stage category."
        if self.qchat_result is not None:
            summary += f"\nSeparate Q-CHAT-style dataset research score: {self.qchat_result.score}/10 (non-diagnostic; not combined with M-CHAT-R/F)."
        self.review_summary.configure(text=summary)

    def _record_clinician_review(self) -> None:
        if self.result is None:
            messagebox.showerror("Cannot record review", "Score the first-stage questionnaire first.")
            return
        if not self.clinician_name_var.get().strip() or not self.clinician_credentials_var.get().strip():
            messagebox.showerror("Reviewer details required", "Enter a local reviewer ID/name and credentials or role.")
            return
        if self.review_plan_var.get() == "Select a local follow-up plan":
            messagebox.showerror("Follow-up plan required", "Select a local follow-up plan before recording review.")
            return
        if not self.review_attested_var.get():
            messagebox.showerror("Attestation required", "Confirm the local review attestation before recording review.")
            return
        self._refresh_review_summary()
        self.review_record = {
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "reviewer": self.clinician_name_var.get().strip(),
            "credentials_or_role": self.clinician_credentials_var.get().strip(),
            "local_follow_up_plan": self.review_plan_var.get(),
            "notes": self.clinician_notes.get("1.0", "end").strip() or None,
            "local_attestation": True,
        }
        self.review_status.configure(text=(
            f"Local clinician review recorded at {self.review_record['reviewed_at']}. "
            "Use Save local report to preserve this reviewed summary on this device."
        ))

    def _build_report_payload(self) -> tuple[dict, str, str]:
        """Return the scored, pseudonymous local record used for JSON and PDF output."""
        if self.result is None:
            raise RuntimeError("Score the questionnaire before saving a report.")
        if self.scored_age_months != int(self.age_var.get()):
            raise RuntimeError("The age changed after scoring. Score the questionnaire again before saving.")
        child_id = storage_token(self.child_id_var.get(), "Child local ID")
        session_id = storage_token(self.session_id_var.get(), "Session ID")
        payload = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "instrument": "M-CHAT-R/F local scoring support",
            "child_local_id": child_id,
            "session_id": session_id,
            "age_months": int(self.age_var.get()),
            "answers": {str(item): self.answer_vars[item].get() for item in range(1, 21)},
            "result": asdict(self.result),
            "caregiver_concerns": self.caregiver_concerns.get("1.0", "end").strip() or None,
            "follow_up": self.follow_up_record,
            "clinician_review": self.review_record,
            "research_observation": self.research_observation,
            "calibration_context": self.research_calibration,
            "webcam_research_parameters": self.attached_webcam_reports,
            "qchat_research": None if self.qchat_result is None else {
                "answers": {str(item): self.qchat_vars[item].get() for item in range(1, 11)},
                "score": self.qchat_result.score,
                "item_codes": list(self.qchat_result.item_codes),
                "interpretation": self.qchat_result.interpretation,
            },
            "record_status": "reviewed" if self.review_record is not None else "draft_unreviewed",
            "clinical_interpretation": "Screening support only; not a diagnosis or automated risk prediction.",
        }
        return payload, child_id, session_id

    def save_local_report(self) -> Path:
        """Save the local, pseudonymous JSON record used to resume the form."""
        payload, child_id, session_id = self._build_report_payload()
        REPORT_DIRECTORY.mkdir(exist_ok=True)
        destination = REPORT_DIRECTORY / f"mchat_screen_{child_id}_{session_id}.json"
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, destination)
        return destination

    def generate_pdf_report(self) -> Path:
        """Render an explicit local PDF of the form and published scoring record."""
        payload, child_id, session_id = self._build_report_payload()
        REPORT_DIRECTORY.mkdir(exist_ok=True)
        destination = REPORT_DIRECTORY / f"mchat_report_{child_id}_{session_id}.pdf"
        return create_screening_pdf(payload, destination)

    def _build_qchat_report_payload(self) -> tuple[dict, str, str]:
        if self.qchat_result is None:
            raise RuntimeError("Record the Q-CHAT-style research score before generating its PDF.")
        try:
            age = int(self.age_var.get())
        except ValueError as error:
            raise RuntimeError("Enter a whole-number age before generating the Q-CHAT research PDF.") from error
        if not 12 <= age <= 36:
            raise RuntimeError("The bundled Q-CHAT-style dataset documentation applies to ages 12 through 36 months.")
        child_id = storage_token(self.child_id_var.get(), "Child local ID")
        session_id = storage_token(self.session_id_var.get(), "Session ID")
        answers = {str(item): self.qchat_vars[item].get() for item in range(1, 11)}
        if any(answer not in {"A", "B", "C", "D", "E"} for answer in answers.values()):
            raise RuntimeError("Complete all ten A-E responses before generating the Q-CHAT research PDF.")
        return {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "instrument": "Q-CHAT-10-style dataset research observation",
            "child_local_id": child_id,
            "session_id": session_id,
            "age_months": age,
            "answers": answers,
            "score": self.qchat_result.score,
            "item_codes": list(self.qchat_result.item_codes),
            "interpretation": self.qchat_result.interpretation,
            "clinical_interpretation": "Research observation only; not a diagnosis, risk score, or automated referral.",
        }, child_id, session_id

    def generate_qchat_pdf_report(self) -> Path:
        payload, child_id, session_id = self._build_qchat_report_payload()
        REPORT_DIRECTORY.mkdir(exist_ok=True)
        destination = REPORT_DIRECTORY / f"qchat_research_{child_id}_{session_id}.pdf"
        return create_qchat_research_pdf(payload, destination)

    def export_final_workflow_json(self) -> Path:
        """Export separate source records without constructing a combined risk result."""
        payload, child_id, session_id = self._build_report_payload()
        final_export = build_final_workflow_export(
            payload, self.attached_webcam_reports, self.research_calibration
        )
        REPORT_DIRECTORY.mkdir(exist_ok=True)
        destination = REPORT_DIRECTORY / f"workflow_summary_{child_id}_{session_id}.json"
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(final_export, indent=2), encoding="utf-8")
        os.replace(temporary, destination)
        return destination

    def _resume_local_session(self) -> None:
        if not self.consent_var.get():
            messagebox.showwarning("Permission required", "Confirm appropriate permission before opening sensitive local screening information.")
            return
        try:
            child_id = storage_token(self.child_id_var.get(), "Child local ID")
            session_id = storage_token(self.session_id_var.get(), "Session ID")
        except ValueError as error:
            messagebox.showerror("Cannot resume session", str(error))
            return
        source = REPORT_DIRECTORY / f"mchat_screen_{child_id}_{session_id}.json"
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
            if payload.get("child_local_id") != child_id or payload.get("session_id") != session_id:
                raise ValueError("The selected report does not match the supplied local IDs.")
            age = int(payload["age_months"])
            raw_answers = payload["answers"]
            answers = {item: raw_answers[str(item)] == "yes" for item in range(1, 21)}
            restored_result = score_mchat(answers, age)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            messagebox.showerror("Cannot resume session", f"No valid local session was loaded: {error}")
            return
        self.age_var.set(str(age))
        self.scored_age_months = age
        for item in range(1, 21):
            self.answer_vars[item].set(raw_answers[str(item)])
        self.caregiver_concerns.delete("1.0", "end")
        self.caregiver_concerns.insert("1.0", payload.get("caregiver_concerns") or "")
        self.result = restored_result
        self.result_label.configure(text=(
            f"Resumed score: {restored_result.score}/20 | Category: {restored_result.category.upper()} | "
            f"Failed items: {', '.join(map(str, restored_result.failed_items)) or 'none'}\n{restored_result.next_step}"
        ))
        self._render_follow_up_items()
        self.follow_up_record = payload.get("follow_up")
        if self.follow_up_record is not None:
            for item, outcome in self.follow_up_record.get("item_outcomes", {}).items():
                if int(item) in self.follow_up_vars:
                    self.follow_up_vars[int(item)].set(outcome)
            self.follow_up_result.configure(text=(
                f"Follow-Up failed items: {self.follow_up_record.get('failed_items')}. "
                f"{self.follow_up_record.get('outcome', '')}"
            ))
        self.review_record = payload.get("clinician_review")
        self.research_observation = payload.get("research_observation")
        self.research_calibration = payload.get("calibration_context")
        self.attached_webcam_reports = payload.get("webcam_research_parameters", [])
        self.research_attachment_label.configure(text=f"{len(self.attached_webcam_reports)} report(s) attached.")
        if self.research_observation is not None and self.research_calibration is not None:
            self._show_research_text(format_observation(self.research_observation, self.research_calibration))
        qchat = payload.get("qchat_research")
        if isinstance(qchat, dict):
            for item, response in qchat.get("answers", {}).items():
                if int(item) in self.qchat_vars:
                    self.qchat_vars[int(item)].set(response)
            try:
                self.qchat_result = score_qchat_research(
                    {item: self.qchat_vars[item].get() for item in range(1, 11)}, age
                )
                self.qchat_label.configure(text=(
                    f"Resumed dataset-derived research score: {self.qchat_result.score}/10. {self.qchat_result.interpretation}"
                ))
            except ValueError:
                self.qchat_result = None
        if self.review_record is not None:
            self.clinician_name_var.set(self.review_record.get("reviewer", ""))
            self.clinician_credentials_var.set(self.review_record.get("credentials_or_role", ""))
            self.review_plan_var.set(self.review_record.get("local_follow_up_plan", "Select a local follow-up plan"))
            self.clinician_notes.delete("1.0", "end")
            self.clinician_notes.insert("1.0", self.review_record.get("notes") or "")
            self.review_attested_var.set(bool(self.review_record.get("local_attestation")))
            self.review_status.configure(text=f"Resumed reviewed local summary from {self.review_record.get('reviewed_at', 'an earlier session')}.")
        self._refresh_review_summary()
        messagebox.showinfo("Local session resumed", "The local screening session was restored. Review and save again if you make changes.")

    def _save_report(self) -> None:
        if not self.consent_var.get():
            messagebox.showwarning("Permission required", "Confirm appropriate permission before saving sensitive screening information.")
            return
        try:
            destination = self.save_local_report()
        except (RuntimeError, ValueError) as error:
            messagebox.showerror("Cannot save report", str(error))
            return
        messagebox.showinfo("Local report saved", f"Saved locally to:\n{destination}\n\nDo not treat this screening result as a diagnosis.")

    def _generate_pdf_report(self) -> None:
        if not self.consent_var.get():
            messagebox.showwarning("Permission required", "Confirm appropriate permission before generating a sensitive local PDF report.")
            return
        try:
            destination = self.generate_pdf_report()
        except (RuntimeError, ValueError, OSError) as error:
            messagebox.showerror("Cannot generate PDF report", str(error))
            return
        messagebox.showinfo(
            "PDF report generated",
            f"The completed caregiver form and M-CHAT-R/F scoring record were saved locally to:\n{destination}\n\n"
            "This is a screening-support report, not a diagnosis or automated risk prediction.",
        )

    def _generate_qchat_pdf_report(self) -> None:
        if not self.consent_var.get():
            messagebox.showwarning("Permission required", "Confirm appropriate permission before generating a sensitive local research PDF.")
            return
        try:
            destination = self.generate_qchat_pdf_report()
        except (RuntimeError, ValueError, OSError) as error:
            messagebox.showerror("Cannot generate research PDF", str(error))
            return
        messagebox.showinfo(
            "Research PDF generated",
            f"The completed Q-CHAT-style research form was saved locally to:\n{destination}\n\n"
            "It is a separate research observation and is not combined with M-CHAT-R/F or used as a diagnosis/risk score.",
        )

    def _export_final_workflow_json(self) -> None:
        if not self.consent_var.get():
            messagebox.showwarning("Permission required", "Confirm appropriate permission before exporting a sensitive local workflow record.")
            return
        try:
            destination = self.export_final_workflow_json()
        except (RuntimeError, ValueError, OSError) as error:
            messagebox.showerror("Cannot export workflow JSON", str(error))
            return
        messagebox.showinfo(
            "Final workflow JSON exported",
            f"Saved locally to:\n{destination}\n\n"
            "The file preserves separate webcam research parameters, M-CHAT-R/F scoring, and official Follow-Up outcomes. It does not generate a combined risk or diagnosis.",
        )


if __name__ == "__main__":
    ScreeningApp().mainloop()
