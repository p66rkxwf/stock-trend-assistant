import type { NextConfig } from "next";

// NEXT_PUBLIC_STATIC_DATA=1：輸出純靜態站（out/，部署到 Cloudflare Pages），
// 資料改讀每日排程匯出的 public/data/*.json（見 lib/api.ts 的 staticApi）
const STATIC = process.env.NEXT_PUBLIC_STATIC_DATA === "1";

const nextConfig: NextConfig = {
  ...(STATIC ? { output: "export" as const } : {}),
};

export default nextConfig;
