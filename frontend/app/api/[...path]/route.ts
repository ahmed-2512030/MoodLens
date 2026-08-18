// Catch-all proxy to the FastAPI backend.
//
// This replaces the previous next.config `rewrites()` proxy. That proxy routed
// browser /api/* calls through Next's internal undici fetch, which enforces a
// hidden ~response timeout and drops the socket ("socket hang up" / ECONNRESET)
// on long requests. A 5,000-row upload runs ML for ~3 minutes, well past that
// limit, so the connection was reset before the ~1.78 MB response came back —
// even though the backend completed successfully.
//
// We proxy with the raw node:http client instead, with every socket timeout
// disabled, so a request can stay open for as long as the backend needs. The
// response is streamed straight back to the browser.

import { NextRequest } from "next/server";
import http from "node:http";
import https from "node:https";
import { Readable } from "node:stream";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
// Serverless hint (harmless in dev): allow long-running ML jobs.
export const maxDuration = 600;

const BACKEND = process.env.BACKEND_URL || "http://localhost:8000";

// Hop-by-hop headers must not be forwarded across a proxy (RFC 7230 §6.1).
const HOP_BY_HOP = new Set([
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
]);

async function proxy(
  req: NextRequest,
  ctx: { params: Promise<{ path: string[] }> },
): Promise<Response> {
  const { path } = await ctx.params;
  const target = new URL(BACKEND);
  const client = target.protocol === "https:" ? https : http;

  const upstreamPath = "/" + path.join("/") + req.nextUrl.search;

  // Buffer the request body. Uploads here are small CSV/JSON files, so buffering
  // is simpler and more robust than half-duplex streaming, and it lets us set an
  // accurate content-length.
  const hasBody = req.method !== "GET" && req.method !== "HEAD";
  const body = hasBody ? Buffer.from(await req.arrayBuffer()) : undefined;

  const headers: Record<string, string> = {};
  req.headers.forEach((value, key) => {
    if (key === "host" || key === "content-length" || HOP_BY_HOP.has(key)) return;
    headers[key] = value;
  });
  if (body) headers["content-length"] = String(body.length);

  return new Promise<Response>((resolve) => {
    const upstream = client.request(
      {
        protocol: target.protocol,
        hostname: target.hostname,
        port: target.port || (target.protocol === "https:" ? 443 : 80),
        method: req.method,
        path: upstreamPath,
        headers,
      },
      (res) => {
        const outHeaders = new Headers();
        for (const [key, value] of Object.entries(res.headers)) {
          if (value === undefined || HOP_BY_HOP.has(key)) continue;
          outHeaders.set(key, Array.isArray(value) ? value.join(", ") : String(value));
        }
        const stream = Readable.toWeb(res) as unknown as ReadableStream<Uint8Array>;
        resolve(
          new Response(stream, {
            status: res.statusCode ?? 502,
            statusText: res.statusMessage ?? "",
            headers: outHeaders,
          }),
        );
      },
    );

    // Never let a long ML job be cut short by an idle-socket timeout.
    upstream.setTimeout(0);
    upstream.on("socket", (socket) => socket.setTimeout(0));

    upstream.on("error", (err) => {
      resolve(
        new Response(
          JSON.stringify({ detail: `Proxy to backend failed: ${err.message}` }),
          { status: 502, headers: { "content-type": "application/json" } },
        ),
      );
    });

    if (body) upstream.write(body);
    upstream.end();
  });
}

export const GET = proxy;
export const POST = proxy;
export const PUT = proxy;
export const PATCH = proxy;
export const DELETE = proxy;
export const HEAD = proxy;
export const OPTIONS = proxy;
