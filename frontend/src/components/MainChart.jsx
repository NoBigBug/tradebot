import React, { useEffect, useRef } from "react";
import { createChart } from "lightweight-charts";

const MainChart = () => {
  const chartContainerRef = useRef();

  useEffect(() => {
    const container = chartContainerRef.current;
    const chart = createChart(container, {
      width: container.clientWidth,
      height: container.clientHeight,
      layout: {
        background: { color: "#1e1e1e" },
        textColor: "#ccc",
      },
      grid: {
        vertLines: { color: "#2b2b2b" },
        horzLines: { color: "#2b2b2b" },
      },
      crosshair: { mode: 1 },
      rightPriceScale: {
        borderColor: "#71649C",
      },
      timeScale: {
        borderColor: "#71649C",
        timeVisible: true,
        secondsVisible: true,
      },
    });

    const candleSeries = chart.addCandlestickSeries();

    // 예시 데이터 (원하면 실제 API 데이터 fetch로 대체 가능)
    candleSeries.setData([
      { time: "2025-06-25T06:00:00Z", open: 100, high: 105, low: 95, close: 102 },
      { time: "2025-06-25T06:01:00Z", open: 102, high: 106, low: 101, close: 104 },
      { time: "2025-06-25T06:02:00Z", open: 104, high: 107, low: 103, close: 105 },
    ]);

    // 리사이즈 대응
    const resizeObserver = new ResizeObserver(() => {
      chart.applyOptions({
        width: container.clientWidth,
        height: container.clientHeight,
      });
    });

    resizeObserver.observe(container);

    return () => {
      resizeObserver.disconnect();
      chart.remove();
    };
  }, []);

  return <div ref={chartContainerRef} className="flex-1" />;
};

export default MainChart;