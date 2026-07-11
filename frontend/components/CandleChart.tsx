"use client";

/**
 * lightweight-charts v5 K 線圖。
 * - v5 API：chart.addSeries(CandlestickSeries, options)（v4 的 addCandlestickSeries 已移除）
 * - 顏色依台股慣例：漲紅、跌綠
 * - useEffect 建圖 + cleanup chart.remove()：React StrictMode 雙掛載不會重複建圖
 */

import { useEffect, useRef } from "react";
import {
  CandlestickSeries,
  ColorType,
  createChart,
  HistogramSeries,
  type IChartApi,
} from "lightweight-charts";
import type { Candle } from "@/lib/api";

const UP_COLOR = "#dc2626"; // 台股：漲紅
const DOWN_COLOR = "#16a34a"; // 台股：跌綠

export default function CandleChart({ candles }: { candles: Candle[] }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const chart = createChart(container, {
      height: 380,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#9ca3af",
      },
      grid: {
        vertLines: { color: "rgba(156,163,175,0.12)" },
        horzLines: { color: "rgba(156,163,175,0.12)" },
      },
      timeScale: { borderColor: "rgba(156,163,175,0.3)" },
      rightPriceScale: { borderColor: "rgba(156,163,175,0.3)" },
    });
    chartRef.current = chart;

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: UP_COLOR,
      downColor: DOWN_COLOR,
      borderUpColor: UP_COLOR,
      borderDownColor: DOWN_COLOR,
      wickUpColor: UP_COLOR,
      wickDownColor: DOWN_COLOR,
    });
    candleSeries.setData(
      candles.map((c) => ({
        time: c.time, // 'YYYY-MM-DD' 字串為 lightweight-charts 支援格式
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
      })),
    );

    const volumeSeries = chart.addSeries(HistogramSeries, {
      priceFormat: { type: "volume" },
      priceScaleId: "volume",
    });
    chart.priceScale("volume").applyOptions({
      scaleMargins: { top: 0.82, bottom: 0 },
    });
    volumeSeries.setData(
      candles.map((c) => ({
        time: c.time,
        value: c.volume,
        color: c.close >= c.open ? "rgba(220,38,38,0.35)" : "rgba(22,163,74,0.35)",
      })),
    );

    chart.timeScale().fitContent();

    const resize = () => chart.applyOptions({ width: container.clientWidth });
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(container);

    return () => {
      observer.disconnect();
      chart.remove();
      chartRef.current = null;
    };
  }, [candles]);

  return <div ref={containerRef} className="w-full" />;
}
