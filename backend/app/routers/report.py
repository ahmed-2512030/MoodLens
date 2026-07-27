"""Generate a downloadable PDF summary of a batch analysis.

The report mirrors the web dashboard: a summary table plus the four
visualisations — distribution pie, average intensity bars, emotion trend, and
keyword->emotion mapping — rendered server-side (see services/report_charts.py).
"""
import io

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.schemas import BatchResponse
from app.services import report_charts as rc

router = APIRouter(tags=["report"])


def _image(buf: io.BytesIO, width_cm: float, aspect: float) -> Image:
    """Embed a chart PNG at a fixed width, preserving aspect ratio."""
    w = width_cm * cm
    return Image(buf, width=w, height=w * aspect)


@router.post("/report")
def report(batch: BatchResponse) -> StreamingResponse:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title="MoodLens Report")
    styles = getSampleStyleSheet()

    story = [
        Paragraph("MoodLens — Emotion Analysis Report", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Documents analysed: {batch.summary.count}", styles["Normal"]),
        Spacer(1, 12),
        Paragraph("Emotion distribution (dominant label)", styles["Heading2"]),
    ]

    dist_rows = [["Emotion", "Count", "Avg score"]]
    for emo, cnt in batch.summary.distribution.items():
        avg = batch.summary.average_scores.get(emo, 0.0)
        dist_rows.append([emo, str(cnt), f"{avg:.3f}"])

    table = Table(dist_rows, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4f46e5")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.whitesmoke]),
            ]
        )
    )
    story.append(table)

    # --- Visualisations (same views as the dashboard) ---
    story += [
        Spacer(1, 16),
        Paragraph("Visualisations", styles["Heading2"]),
        _image(rc.pie_png(batch.summary.distribution), 12, 4 / 6.5),
        Spacer(1, 8),
        _image(rc.bars_png(batch.summary.average_scores), 15, 3.5 / 6.5),
        PageBreak(),
        _image(rc.trend_png(batch.items), 16, 3.6 / 7),
        Spacer(1, 12),
        _image(rc.keyword_png(batch.items), 16, 4 / 7),
    ]

    doc.build(story)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=moodlens-report.pdf"},
    )
