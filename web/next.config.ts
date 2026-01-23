import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/:path*", // Proxy to Backend
      },
      {
        source: "/chat/:path*",
        destination: "http://localhost:8000/chat/:path*", // Proxy to Chat
      },
      {
        source: "/agent/:path*",
        destination: "http://localhost:8000/agent/:path*", // Proxy to Agent
      },
    ];
  },
};

export default nextConfig;
