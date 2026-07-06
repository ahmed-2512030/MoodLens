"""Generate a downloadable PDF summary of a batch analysis."""
import io

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    Paragraph,
)

from app.schemas import BatchResponse

router = APIRouter(tags=["report"])


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

    doc.build(story)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=moodlens-report.pdf"},
    )
