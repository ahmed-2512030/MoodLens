"use client";

import { useState } from "react";
import {
  AnalyzeResponse,
  BatchResponse,
  EMOTION_COLORS,
  Emotion,
  analyzeText,
  uploadFile,
} from "@/lib/api";
import { AverageBars, DistributionPie } from "@/components/EmotionCharts";

export default function Home() {
  const [text, setText] = useState("");
  const [single, setSingle] = useState<AnalyzeResponse | null>(null);
  const [batch, setBatch] = useState<BatchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function runText() {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    try {
      setSingle(await analyzeText(text));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  async function runUpload(file: File) {
    setLoading(true);
    setError(null);
    try {
      setBatch(await uploadFile(file));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="mx-auto max-w-5xl px-6 py-10">
      <header className="mb-8">
        <h1 className="text-3xl font-bold">MoodLens</h1>
        <p className="text-slate-500">
          BERT-based emotion &amp; sentiment analysis
        </p>
      </header>

      {error && (
        <div className="mb-6 rounded-md bg-red-50 px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {/* Text analysis */}
      <section className="mb-10 rounded-xl bg-white p-6 shadow-sm">
        <h2 className="mb-3 text-lg font-semibold">Analyse text</h2>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={4}
          placeholder="Type or paste text to analyse…"
          className="w-full rounded-md border border-slate-300 p-3"
        />
        <button
          onClick={runText}
          disabled={loading}
          className="mt-3 rounded-md bg-indigo-600 px-4 py-2 text-white disabled:opacity-50"
        >
          {loading ? "Analysing…" : "Analyse"}
        </button>

        {single && (
          <div className="mt-5">
            <p className="mb-2">
              Dominant emotion:{" "}
              <span
                className="rounded px-2 py-0.5 font-medium text-white"
                style={{ background: EMOTION_COLORS[single.dominant] }}
              >
                {single.dominant}
              </span>
            </p>
            <div className="space-y-1">
              {Object.entries(single.emotions).map(([emo, score]) => (
                <div key={emo} className="flex items-center gap-2 text-sm">
                  <span className="w-20">{emo}</span>
                  <div className="h-3 flex-1 rounded bg-slate-100">
                    <div
                      className="h-3 rounded"
                      style={{
                        width: `${score * 100}%`,
                        background: EMOTION_COLORS[emo as Emotion],
                      }}
                    />
                  </div>
                  <span className="w-12 text-right">
                    {(score * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </section>

      {/* Batch upload */}
      <section className="rounded-xl bg-white p-6 shadow-sm">
        <h2 className="mb-3 text-lg font-semibold">Upload dataset (CSV / JSON)</h2>
        <input
          type="file"
          accept=".csv,.json"
          onChange={(e) => e.target.files?.[0] && runUpload(e.target.files[0])}
          className="text-sm"
        />

        {batch && (
          <div className="mt-6">
            <p className="mb-4 text-sm text-slate-500">
              {batch.summary.count} documents analysed
            </p>
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
          </div>
        )}
      </section>
    </main>
  );
}
