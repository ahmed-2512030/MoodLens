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

export interface AnalyzeResponse extends EmotionResult {
  text: string;
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

export async function analyzeText(text: string): Promise<AnalyzeResponse> {
  const res = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error((await res.json()).detail ?? "Request failed");
  return res.json();
}

export async function uploadFile(file: File): Promise<BatchResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/upload", { method: "POST", body: form });
  if (!res.ok) throw new Error((await res.json()).detail ?? "Upload failed");
  return res.json();
}

// POST the batch back to the backend, which renders a PDF and streams it down.
export async function downloadReport(batch: BatchResponse): Promise<void> {
  const res = await fetch("/api/report", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(batch),
  });
  if (!res.ok) throw new Error("Report generation failed");
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "moodlens-report.pdf";
  a.click();
  URL.revokeObjectURL(url);
}
