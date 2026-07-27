"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  AnalyzeResponse,
  BatchResponse,
  EMOTION_COLORS,
  Emotion,
  analyzeText,
  downloadReport,
  uploadFile,
} from "@/lib/api";
import { buildKeywordMap, buildTrend } from "@/lib/analytics";
import {
  AverageBars,
  DistributionPie,
  EmotionLegend,
  KeywordEmotionMap,
  TrendChart,
} from "@/components/EmotionCharts";

type Tab = "distribution" | "trend" | "keywords";

export default function Home() {
  const [text, setText] = useState("");
  const [single, setSingle] = useState<AnalyzeResponse | null>(null);
  const [batch, setBatch] = useState<BatchResponse | null>(null);
  const [liveOn, setLiveOn] = useState(true);
  const [loading, setLoading] = useState(false);
  const [reporting, setReporting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("distribution");
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  // ---- Real-time single analysis: debounce typing, auto-analyse. ----------
  useEffect(() => {
    if (!liveOn) return;
    if (text.trim().length < 3) {
      setSingle(null);
      return;
    }
    clearTimeout(timer.current);
    timer.current = setTimeout(async () => {
      try {
        setSingle(await analyzeText(text));
        setError(null);
      } catch (e) {
        setError((e as Error).message);
      }
    }, 450);
    return () => clearTimeout(timer.current);
  }, [text, liveOn]);

  async function runUpload(file: File) {
    setLoading(true);
    setError(null);
    try {
      setBatch(await uploadFile(file));
      setTab("distribution");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  async function runReport() {
    if (!batch) return;
    setReporting(true);
    try {
      await downloadReport(batch);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setReporting(false);
    }
  }

  // Derived analytics (memoised so charts recompute only when the batch changes).
  const trend = useMemo(() => (batch ? buildTrend(batch.items) : []), [batch]);
  const keywords = useMemo(
    () => (batch ? buildKeywordMap(batch.items) : []),
    [batch]
  );

  const topEmotion = batch
    ? (Object.entries(batch.summary.distribution).sort(
        (a, b) => b[1] - a[1]
      )[0]?.[0] as Emotion)
    : null;
  const activeEmotions = batch
    ? Object.values(batch.summary.distribution).filter((v) => v > 0).length
    : 0;

  return (
    <main className="mx-auto max-w-6xl px-6 py-10">
      <header className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight">MoodLens</h1>
        <p className="text-slate-500">
          BERT-based emotion &amp; sentiment analysis dashboard
        </p>
      </header>

      {error && (
        <div className="mb-6 rounded-md bg-red-50 px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {/* Real-time text analysis */}
      <section className="mb-10 rounded-xl bg-white p-6 shadow-sm">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-lg font-semibold">Analyse text</h2>
          <label className="flex items-center gap-2 text-sm text-slate-500">
            <input
              type="checkbox"
              checked={liveOn}
              onChange={(e) => setLiveOn(e.target.checked)}
            />
            Real-time
          </label>
        </div>

        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={4}
          placeholder="Type or paste text — analysis updates as you type…"
          className="w-full rounded-md border border-slate-300 p-3 focus:border-indigo-500 focus:outline-none"
        />
        {!liveOn && (
          <button
            onClick={async () => {
              if (!text.trim()) return;
              setLoading(true);
              try {
                setSingle(await analyzeText(text));
                setError(null);
              } catch (e) {
                setError((e as Error).message);
              } finally {
                setLoading(false);
              }
            }}
            disabled={loading}
            className="mt-3 rounded-md bg-indigo-600 px-4 py-2 text-white disabled:opacity-50"
          >
            {loading ? "Analysing…" : "Analyse"}
          </button>
        )}

        {single && (
          <div className="mt-5">
            <p className="mb-3">
              Dominant emotion:{" "}
              <span
                className="rounded px-2 py-0.5 font-medium text-white"
                style={{ background: EMOTION_COLORS[single.dominant] }}
              >
                {single.dominant}
              </span>
            </p>
            <div className="space-y-1">
              {Object.entries(single.emotions)
                .sort((a, b) => b[1] - a[1])
                .map(([emo, score]) => (
                  <div key={emo} className="flex items-center gap-2 text-sm">
                    <span className="w-20">{emo}</span>
                    <div className="h-3 flex-1 rounded bg-slate-100">
                      <div
                        className="h-3 rounded transition-all duration-300"
                        style={{
                          width: `${score * 100}%`,
                          background: EMOTION_COLORS[emo as Emotion],
                        }}
                      />
                    </div>
                    <span className="w-12 text-right tabular-nums">
                      {(score * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
            </div>
          </div>
        )}
      </section>

      {/* Batch dashboard */}
      <section className="rounded-xl bg-white p-6 shadow-sm">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-lg font-semibold">Dataset dashboard</h2>
          <div className="flex items-center gap-3">
            <input
              type="file"
              accept=".csv,.json"
              onChange={(e) =>
                e.target.files?.[0] && runUpload(e.target.files[0])
              }
              className="text-sm"
            />
            {batch && (
              <button
                onClick={runReport}
                disabled={reporting}
                className="rounded-md border border-indigo-600 px-3 py-1.5 text-sm font-medium text-indigo-600 hover:bg-indigo-50 disabled:opacity-50"
              >
                {reporting ? "Building…" : "Download PDF"}
              </button>
            )}
          </div>
        </div>

        {loading && !batch && (
          <p className="text-sm text-slate-400">Analysing dataset…</p>
        )}

        {batch && (
          <>
            {/* KPI tiles */}
            <div className="mb-6 grid grid-cols-2 gap-4 md:grid-cols-4">
              <StatTile label="Documents" value={String(batch.summary.count)} />
              <StatTile
                label="Top emotion"
                value={topEmotion ?? "—"}
                color={topEmotion ? EMOTION_COLORS[topEmotion] : undefined}
              />
              <StatTile
                label="Emotions present"
                value={`${activeEmotions} / 7`}
              />
              <StatTile label="Keywords mapped" value={String(keywords.length)} />
            </div>

            {/* Tabs */}
            <div className="mb-4 flex gap-1 border-b border-slate-200 text-sm">
              {(
                [
                  ["distribution", "Distribution"],
                  ["trend", "Trend"],
                  ["keywords", "Key emotions"],
                ] as [Tab, string][]
              ).map(([key, label]) => (
                <button
                  key={key}
                  onClick={() => setTab(key)}
                  className={`-mb-px border-b-2 px-4 py-2 font-medium ${
                    tab === key
                      ? "border-indigo-600 text-indigo-600"
                      : "border-transparent text-slate-500 hover:text-slate-700"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>

            {tab === "distribution" && (
              <div className="grid gap-6 md:grid-cols-2">
                <div>
                  <h3 className="mb-2 text-sm font-medium">
                    Dominant emotion distribution
                  </h3>
                  <DistributionPie summary={batch.summary} />
                </div>
                <div>
                  <h3 className="mb-2 text-sm font-medium">
                    Average emotion intensity
                  </h3>
                  <AverageBars summary={batch.summary} />
                </div>
              </div>
            )}

            {tab === "trend" && (
              <div>
                <h3 className="mb-2 text-sm font-medium">
                  Emotion intensity across the document stream
                </h3>
                <TrendChart data={trend} />
              </div>
            )}

            {tab === "keywords" && (
              <div>
                <h3 className="mb-2 text-sm font-medium">
                  Keyword → emotion mapping
                </h3>
                <p className="mb-3 text-sm text-slate-500">
                  Words sized by how often they appear; coloured by the emotion
                  they most associate with.
                </p>
                <KeywordEmotionMap stats={keywords} />
              </div>
            )}
          </>
        )}

        {!batch && !loading && (
          <div className="text-sm text-slate-400">
            Upload a CSV/JSON dataset to see distribution, trend, and keyword
            charts. <EmotionLegend />
          </div>
        )}
      </section>
    </main>
  );
}

function StatTile({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="rounded-lg border border-slate-200 p-4">
      <p className="text-xs uppercase tracking-wide text-slate-400">{label}</p>
      <p
        className="mt-1 text-xl font-semibold capitalize"
        style={color ? { color } : undefined}
      >
        {value}
      </p>
    </div>
  );
}
