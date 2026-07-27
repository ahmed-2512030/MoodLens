"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { EMOTION_COLORS, EKMAN, Emotion, EmotionSummary } from "@/lib/api";
import { KeywordStat, TrendPoint } from "@/lib/analytics";

export function DistributionPie({ summary }: { summary: EmotionSummary }) {
  const data = Object.entries(summary.distribution)
    .filter(([, v]) => v > 0)
    .map(([name, value]) => ({ name, value }));

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer>
        <PieChart>
          <Pie
            data={data}
            dataKey="value"
            nameKey="name"
            outerRadius={100}
            label={(d) => d.name}
          >
            {data.map((d) => (
              <Cell key={d.name} fill={EMOTION_COLORS[d.name as Emotion]} />
            ))}
          </Pie>
          <Tooltip />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}

export function AverageBars({ summary }: { summary: EmotionSummary }) {
  const data = Object.entries(summary.average_scores).map(([name, value]) => ({
    name,
    value: Number((value * 100).toFixed(1)),
  }));

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer>
        <BarChart data={data}>
          <XAxis dataKey="name" tick={{ fontSize: 12 }} />
          <YAxis unit="%" />
          <Tooltip />
          <Bar dataKey="value" radius={[4, 4, 0, 0]}>
            {data.map((d) => (
              <Cell key={d.name} fill={EMOTION_COLORS[d.name as Emotion]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// Emotion intensity across the ordered document stream — trend analysis.
export function TrendChart({ data }: { data: TrendPoint[] }) {
  return (
    <div className="h-80 w-full">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey="bin" tick={{ fontSize: 11 }} />
          <YAxis unit="%" tick={{ fontSize: 11 }} />
          <Tooltip />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          {EKMAN.map((emo) => (
            <Line
              key={emo}
              type="monotone"
              dataKey={emo}
              stroke={EMOTION_COLORS[emo]}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// Keyword → emotion mapping. Chip size/opacity scales with frequency; colour is
// the word's dominant emotion. Explains WHICH words drove WHICH emotion.
export function KeywordEmotionMap({ stats }: { stats: KeywordStat[] }) {
  if (!stats.length) {
    return (
      <p className="text-sm text-slate-400">
        Not enough repeated words to map. Upload more text.
      </p>
    );
  }
  const max = stats[0].total;

  return (
    <div>
      <div className="flex flex-wrap gap-2">
        {stats.map((s) => {
          const strength = 0.45 + 0.55 * (s.total / max); // 0.45–1.0
          return (
            <span
              key={s.word}
              title={`${s.word} — ${s.dominant} (${s.total} docs)`}
              className="rounded-full px-3 py-1 font-medium text-white"
              style={{
                background: EMOTION_COLORS[s.dominant],
                opacity: strength,
                fontSize: `${0.75 + 0.5 * (s.total / max)}rem`,
              }}
            >
              {s.word}
            </span>
          );
        })}
      </div>
      <EmotionLegend />
    </div>
  );
}

export function EmotionLegend() {
  return (
    <div className="mt-4 flex flex-wrap gap-3 text-xs text-slate-500">
      {EKMAN.map((emo) => (
        <span key={emo} className="flex items-center gap-1.5">
          <span
            className="inline-block h-3 w-3 rounded-sm"
            style={{ background: EMOTION_COLORS[emo] }}
          />
          {emo}
        </span>
      ))}
    </div>
  );
}
