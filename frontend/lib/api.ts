// Shared types + fetch helpers. Browser calls same-origin /api/* which
// next.config rewrites to the FastAPI backend.

export const EKMAN = [
  "joy",
  "anger",
  "sadness",
  "fear",
  "surprise",
  "disgust",
  "neutral",
] as const;

export type Emotion = (typeof EKMAN)[number];

export const EMOTION_COLORS: Record<Emotion, string> = {
  joy: "#f59e0b",
  anger: "#ef4444",
  sadness: "#3b82f6",
  fear: "#8b5cf6",
  surprise: "#ec4899",
  disgust: "#10b981",
  neutral: "#94a3b8",
};

export interface FineGrained {
  label: string;
  score: number;
}

export interface EmotionResult {
  dominant: Emotion;
  emotions: Record<Emotion, number>;
  fine_grained: FineGrained[];
}

// One chunk's Ekman vector (0-100 scale). Structurally identical to analytics'
// TrendPoint, so a document's arc feeds straight into <TrendChart/>.
export interface ArcPoint {
  bin: string;
  index: number;
  joy: number;
  anger: number;
  sadness: number;
  fear: number;
  surprise: number;
  disgust: number;
  neutral: number;
}

export interface AnalyzeResponse extends EmotionResult {
  text: string;
  chunk_count?: number;
  arc?: ArcPoint[] | null; // per-chunk trajectory for long docs; null when single-chunk
  truncated_chunks?: boolean;
}

export interface EmotionSummary {
  count: number;
  distribution: Record<Emotion, number>;
  average_scores: Record<Emotion, number>;
}

export interface BatchResponse {
  summary: EmotionSummary;
  items: AnalyzeResponse[];
}

// Turn any error response into a readable message. Never throws: the body may
// be JSON ({"detail": ...}), a 422 detail array, or non-JSON (a proxy/gateway
// HTML page, a plain-text 500). Blindly calling res.json() here is what surfaced
// the confusing "invalid JSON" error to users — this parses defensively instead.
async function errorMessage(res: Response, fallback: string): Promise<string> {
  const body = await res.text().catch(() => "");
  try {
    const data = JSON.parse(body);
    const detail = data?.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((e) => e?.msg ?? JSON.stringify(e))
        .join("; ");
    }
    if (detail) return JSON.stringify(detail);
  } catch {
    // Body wasn't JSON — fall through to a status-based message.
  }
  return `${fallback} (HTTP ${res.status}${res.statusText ? " " + res.statusText : ""})`;
}

export async function analyzeText(text: string): Promise<AnalyzeResponse> {
  const res = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error(await errorMessage(res, "Analysis failed"));
  return res.json();
}

export async function uploadFile(file: File): Promise<BatchResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/upload", { method: "POST", body: form });
  if (!res.ok) throw new Error(await errorMessage(res, "Upload failed"));
  return res.json();
}

// POST the batch back to the backend, which renders a PDF and streams it down.
export async function downloadReport(batch: BatchResponse): Promise<void> {
  const res = await fetch("/api/report", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(batch),
  });
  if (!res.ok) throw new Error(await errorMessage(res, "Report generation failed"));
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "moodlens-report.pdf";
  a.click();
  URL.revokeObjectURL(url);
}
