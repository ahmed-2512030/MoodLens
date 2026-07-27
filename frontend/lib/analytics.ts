// Client-side analytics derived from batch items. Computed in the browser so the
// dashboard updates instantly (no extra backend round-trip) — the "real-time" feel.

import { AnalyzeResponse, EKMAN, Emotion } from "@/lib/api";

// ---- Trend analysis -------------------------------------------------------
// Bin the ordered document stream and average each emotion's intensity per bin.
// If the uploaded data is chronological, this reads as emotion-over-time.

export interface TrendPoint {
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

export function buildTrend(items: AnalyzeResponse[], bins = 12): TrendPoint[] {
  const n = items.length;
  if (!n) return [];
  const size = Math.max(1, Math.ceil(n / bins));
  const points: TrendPoint[] = [];

  for (let start = 0; start < n; start += size) {
    const slice = items.slice(start, start + size);
    const point = {
      bin: `${start + 1}–${Math.min(start + size, n)}`,
      index: points.length,
    } as TrendPoint;
    for (const emo of EKMAN) {
      const mean =
        slice.reduce((sum, it) => sum + (it.emotions[emo] ?? 0), 0) / slice.length;
      point[emo] = Number((mean * 100).toFixed(1));
    }
    points.push(point);
  }
  return points;
}

// ---- Keyword → emotion mapping (explainability) ---------------------------
// Which words drive which emotions. Each document's words are attributed to its
// dominant emotion (deduped per document so one long text can't dominate).

export interface KeywordStat {
  word: string;
  total: number;
  dominant: Emotion;
  counts: Record<Emotion, number>;
}

const STOPWORDS = new Set(
  ("a an the and or but if then so of to in on at for with without from by as is are was " +
    "were be been being it its it's this that these those i you he she we they them his her " +
    "our your their my me him us do does did done have has had having not no nor can could " +
    "will would shall should may might must just really very too also about into over under " +
    "up down out off than there here what which who whom when where why how all any some " +
    "more most other such only own same more get got like one two get im ive dont cant" )
    .split(" ")
);

function tokenize(text: string): string[] {
  const raw = text.toLowerCase().match(/[a-z][a-z']{2,}/g) ?? [];
  return raw.filter((w) => !STOPWORDS.has(w));
}

export function buildKeywordMap(
  items: AnalyzeResponse[],
  topN = 30
): KeywordStat[] {
  const table = new Map<string, KeywordStat>();

  for (const item of items) {
    const dominant = item.dominant;
    const words = new Set(tokenize(item.text)); // dedupe per document
    for (const word of words) {
      let stat = table.get(word);
      if (!stat) {
        stat = {
          word,
          total: 0,
          dominant,
          counts: Object.fromEntries(EKMAN.map((e) => [e, 0])) as Record<
            Emotion,
            number
          >,
        };
        table.set(word, stat);
      }
      stat.counts[dominant] += 1;
      stat.total += 1;
    }
  }

  // Resolve each word's dominant emotion (argmax) and rank by frequency.
  const stats = [...table.values()];
  for (const stat of stats) {
    stat.dominant = EKMAN.reduce((best, e) =>
      stat.counts[e] > stat.counts[best] ? e : best
    );
  }
  return stats
    .filter((s) => s.total >= 2) // ignore one-off noise
    .sort((a, b) => b.total - a.total)
    .slice(0, topN);
}
