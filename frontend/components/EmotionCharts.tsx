"use client";

import {
  Bar,
  BarChart,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { EMOTION_COLORS, Emotion, EmotionSummary } from "@/lib/api";

export function DistributionPie({ summary }: { summary: EmotionSummary }) {
  const data = Object.entries(summary.distribution)
    .filter(([, v]) => v > 0)
    .map(([name, value]) => ({ name, value }));

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer>
        <PieChart>
          <Pie data={data} dataKey="value" nameKey="name" outerRadius={100} label>
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
          <Bar dataKey="value">
            {data.map((d) => (
              <Cell key={d.name} fill={EMOTION_COLORS[d.name as Emotion]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
