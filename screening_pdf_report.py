"""PDF rendering for the local M-CHAT-R/F screening-support record.

The report is deliberately a faithful record of the completed local form and
published M-CHAT-R/F scoring outcome.  It must not be used to generate a new
ASD/ADHD likelihood, diagnosis, or automated referral.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from asd_screening import MCHAT_QUESTIONS
from qchat_research import QCHAT_ITEMS


DISCLAIMER = (
    "Screening support only. This report is not a diagnosis, ASD/ADHD likelihood "
    "score, automated referral, or treatment decision. Interpret it alongside "
    "clinical judgment and any caregiver concerns."
)


def _text(value: Any, empty: str = "Not recorded") -> str:
    if value is None:
        return empty
    rendered = str(value).strip()
    return rendered or empty


def _paragraph(text: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(_text(text)).replace("\n", "<br/>"), style)


def _footer(canvas: Any, document: Any) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#536471"))
    canvas.drawString(18 * mm, 12 * mm, "Local developmental record - confidential local data")
    canvas.drawRightString(192 * mm, 12 * mm, f"Page {document.page}")
    canvas.restoreState()


def create_screening_pdf(payload: Mapping[str, Any], destination: Path) -> Path:
    """Create a readable local PDF from an already scored questionnaire payload."""
    required = {"child_local_id", "session_id", "age_months", "answers", "result"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"The local record is missing required report fields: {', '.join(missing)}")

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.pdf")
    styles = getSampleStyleSheet()
    title = ParagraphStyle("ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=colors.HexColor("#102A43"))
    heading = ParagraphStyle("Heading", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=13, leading=17, textColor=colors.HexColor("#102A43"), spaceBefore=12, spaceAfter=6)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13)
    small = ParagraphStyle("Small", parent=body, fontSize=8.5, leading=11)
    warning = ParagraphStyle("Warning", parent=body, textColor=colors.HexColor("#7A271A"), borderColor=colors.HexColor("#FECACA"), borderWidth=0.8, borderPadding=8, backColor=colors.HexColor("#FFF7ED"))

    document = SimpleDocTemplate(
        str(temporary), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=20 * mm, title="Local M-CHAT-R/F screening-support record",
    )
    result = payload["result"]
    answers = payload["answers"]
    failed = result.get("failed_items", []) if isinstance(result, Mapping) else []
    story: list[Any] = [
        Paragraph("Local M-CHAT-R/F Screening Support Report", title),
        Spacer(1, 4 * mm),
        _paragraph(DISCLAIMER, warning),
        Spacer(1, 5 * mm),
        Paragraph("Record details", heading),
    ]
    details = [
        ["Child local ID", _text(payload["child_local_id"])],
        ["Session ID", _text(payload["session_id"])],
        ["Age at scoring", f"{payload['age_months']} months"],
        ["Recorded at (UTC)", _text(payload.get("created_at"))],
        ["Record status", _text(payload.get("record_status"), "Draft - not clinician reviewed")],
    ]
    detail_table = Table(details, colWidths=(43 * mm, 126 * mm))
    detail_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF2F8")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#102A43")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("LEADING", (0, 0), (-1, -1), 12),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B8C7D1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([detail_table, Paragraph("Published first-stage scoring record", heading)])
    score_rows = [
        ["First-stage score", f"{result.get('score', 'Not recorded')} / 20"],
        ["Published category", _text(result.get("category"))],
        ["Failed item numbers", ", ".join(map(str, failed)) or "None"],
        ["Published next step", _text(result.get("next_step"))],
    ]
    score_table = Table([[_paragraph(a, small), _paragraph(b, small)] for a, b in score_rows], colWidths=(45 * mm, 124 * mm))
    score_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#FFF3CD")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D4C39A")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([score_table, Paragraph("Caregiver concerns", heading), _paragraph(payload.get("caregiver_concerns"), body)])

    follow_up = payload.get("follow_up")
    story.append(Paragraph("Official Follow-Up outcome", heading))
    if isinstance(follow_up, Mapping):
        outcomes = follow_up.get("item_outcomes", {})
        outcome_rows = [["Failed Follow-Up items", _text(follow_up.get("failed_items"))], ["Recorded outcome", _text(follow_up.get("outcome"))]]
        if isinstance(outcomes, Mapping):
            outcome_rows.append(["Item outcomes", ", ".join(f"{key}: {_text(value)}" for key, value in outcomes.items()) or "Not recorded"])
        follow_table = Table([[_paragraph(a, small), _paragraph(b, small)] for a, b in outcome_rows], colWidths=(45 * mm, 124 * mm))
        follow_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF7EE")), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#AFC8AF")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        story.append(follow_table)
    else:
        story.append(_paragraph("No official Follow-Up outcome was recorded in this local session.", body))

    review = payload.get("clinician_review")
    story.append(Paragraph("Clinician review", heading))
    if isinstance(review, Mapping):
        review_rows = [[label, _text(review.get(key))] for label, key in (("Reviewer", "reviewer"), ("Credentials / role", "credentials_or_role"), ("Reviewed at (UTC)", "reviewed_at"), ("Local follow-up plan", "local_follow_up_plan"), ("Local attestation", "local_attestation"), ("Notes", "notes"))]
        review_table = Table([[_paragraph(a, small), _paragraph(b, small)] for a, b in review_rows], colWidths=(45 * mm, 124 * mm))
        review_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F2F4F7")), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C5CDD3")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        story.append(review_table)
    else:
        story.append(_paragraph("No clinician review has been recorded. This local record remains unreviewed.", body))

    story.extend([PageBreak(), Paragraph("Completed caregiver questionnaire", title), Spacer(1, 3 * mm), _paragraph("Answers are reproduced exactly as recorded in this local session. An asterisk marks an item counted as failed by the published first-stage scoring convention.", body), Spacer(1, 4 * mm)])
    answer_rows: list[list[Any]] = [[_paragraph("Item", small), _paragraph("Question", small), _paragraph("Recorded answer", small), _paragraph("Scoring", small)]]
    failed_set = {int(item) for item in failed}
    for item, question in enumerate(MCHAT_QUESTIONS, start=1):
        answer_rows.append([
            _paragraph(str(item), small), _paragraph(question, small), _paragraph(_text(answers.get(str(item))), small), _paragraph("Failed item" if item in failed_set else "Not failed", small),
        ])
    answers_table = Table(answer_rows, colWidths=(12 * mm, 111 * mm, 23 * mm, 23 * mm), repeatRows=1)
    answers_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#102A43")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C5CDD3")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    for row, item in enumerate(range(1, 21), start=1):
        if item in failed_set:
            answers_table.setStyle(TableStyle([("BACKGROUND", (0, row), (-1, row), colors.HexColor("#FFF7ED"))]))
    story.append(answers_table)
    story.extend([Spacer(1, 5 * mm), _paragraph("The M-CHAT-R/F instrument and its official Follow-Up interview are subject to their own terms. This application records local answers and final clinician-recorded Follow-Up outcomes; it does not reproduce the Follow-Up decision trees.", small)])

    try:
        document.build(story, onFirstPage=_footer, onLaterPages=_footer)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def create_qchat_research_pdf(payload: Mapping[str, Any], destination: Path) -> Path:
    """Create a local report of the separate Q-CHAT-style research form.

    This rendering intentionally records the documented dataset encoding only;
    it is not an interpretation of autism likelihood or a clinical result.
    """
    required = {"child_local_id", "session_id", "age_months", "answers", "score", "interpretation"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"The local Q-CHAT research record is missing: {', '.join(missing)}")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.pdf")
    styles = getSampleStyleSheet()
    title = ParagraphStyle("QTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=colors.HexColor("#102A43"))
    heading = ParagraphStyle("QHeading", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=13, leading=17, textColor=colors.HexColor("#102A43"), spaceBefore=12, spaceAfter=6)
    body = ParagraphStyle("QBody", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13)
    small = ParagraphStyle("QSmall", parent=body, fontSize=8.4, leading=10.5)
    warning = ParagraphStyle("QWarning", parent=body, textColor=colors.HexColor("#7A271A"), borderColor=colors.HexColor("#FECACA"), borderWidth=0.8, borderPadding=8, backColor=colors.HexColor("#FFF7ED"))
    document = SimpleDocTemplate(str(temporary), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm, topMargin=16 * mm, bottomMargin=20 * mm, title="Local Q-CHAT-style research observation")
    answers = payload["answers"]
    story: list[Any] = [
        Paragraph("Q-CHAT-10-style Research Observation", title), Spacer(1, 4 * mm),
        _paragraph("Research observation only. This local report uses the response encoding documented in the bundled toddler dataset. It is not a diagnosis, ASD/ADHD likelihood score, clinical screen, automated referral, or treatment decision. It is not combined with M-CHAT-R/F results.", warning), Spacer(1, 5 * mm),
        Paragraph("Local record details", heading),
    ]
    details = [
        ["Child local ID", _text(payload["child_local_id"])], ["Session ID", _text(payload["session_id"])],
        ["Age at recording", f"{payload['age_months']} months"], ["Recorded at (UTC)", _text(payload.get("created_at"))],
        ["Dataset reference age", "12-36 months"],
    ]
    details_table = Table(details, colWidths=(45 * mm, 124 * mm))
    details_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF2F8")), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B8C7D1")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story.extend([details_table, Paragraph("Documented dataset encoding record", heading)])
    summary = Table([[_paragraph("Recorded research score", small), _paragraph(f"{payload['score']} / 10", small)], [_paragraph("Research interpretation", small), _paragraph(payload["interpretation"], small)]], colWidths=(45 * mm, 124 * mm))
    summary.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF7EE")), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#AFC8AF")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story.extend([summary, Spacer(1, 3 * mm), _paragraph("The score is a documented research-data encoding. It should not be interpreted as an individual clinical threshold or used to make decisions about the child.", body), PageBreak(), Paragraph("Completed Q-CHAT-style research form", title), Spacer(1, 3 * mm)])
    rows: list[list[Any]] = [[_paragraph("Item", small), _paragraph("Research question", small), _paragraph("Recorded response", small)]]
    for item, (question, options) in enumerate(QCHAT_ITEMS, start=1):
        selected = _text(answers.get(str(item)))
        option_label = ""
        if selected in "ABCDE":
            option_label = options["ABCDE".index(selected)]
        rows.append([_paragraph(str(item), small), _paragraph(question, small), _paragraph(f"{selected}. {option_label}" if option_label else selected, small)])
    table = Table(rows, colWidths=(12 * mm, 116 * mm, 43 * mm), repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#102A43")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C5CDD3")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    story.extend([table, Spacer(1, 5 * mm), _paragraph("This report preserves the form's local A-E responses and the associated documented dataset encoding. No clinical meaning beyond that record is inferred.", small)])
    try:
        document.build(story, onFirstPage=_footer, onLaterPages=_footer)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination
