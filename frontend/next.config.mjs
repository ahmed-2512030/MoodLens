/** @type {import('next').NextConfig} */
const nextConfig = {
  // API calls to /api/* are proxied to the FastAPI backend by the catch-all
  // Route Handler at app/api/[...path]/route.ts (not a rewrite). The route
  // handler disables socket timeouts so long ML jobs — e.g. a 5,000-row upload
  // that runs for minutes — aren't reset mid-flight. See that file for details.
};

export default nextConfig;
