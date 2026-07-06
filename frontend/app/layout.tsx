import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MoodLens",
  description: "BERT-based emotion & sentiment analysis platform",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
