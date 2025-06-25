import React, { useEffect, useRef } from "react";

const Chart = () => {
  const chartContainerRef = useRef(null);
  const miniChartRefs = useRef([]);

  const miniSymbols = [
    { label: "Nasdaq", widget: "NASDAQ:NDX" },
    { label: "DXY", widget: "FXOPEN:DXY" },
    { label: "BTC.D", widget: "CRYPTOCAP:BTC.D" },
  ];

  useEffect(() => {
    const script = document.createElement("script");
    script.src = "https://s3.tradingview.com/tv.js";
    script.async = true;
    script.onload = () => {
      new window.TradingView.widget({
        container_id: "tv-main-chart",
        width: "100%",
        height: 600,
        symbol: "BINANCE:BTCUSDTPERP",
        interval: "1",
        timezone: "Asia/Seoul",
        theme: "dark",
        style: "1",
        locale: "ko",
        hide_top_toolbar: true,       // 툴바 숨김
        hide_legend: true,            // 범례 숨김
        hide_volume: true,            // 볼륨 숨김
        withdateranges: false,
        allow_symbol_change: false,
        details: false,
        hotlist: false,
        calendar: false,
      });

      // 미니 차트 생성
      miniSymbols.forEach((item, i) => {
        new window.TradingView.widget({
          container_id: `tv-mini-${i}`,
          width: "100%",
          height: 200,
          symbol: item.widget,
          interval: "15",
          timezone: "Asia/Seoul",
          theme: "dark",
          style: "1",
          locale: "ko",
          hide_top_toolbar: true,
          hide_side_toolbar: true,
          allow_symbol_change: false,
          withdateranges: false,
          save_image: false,
          details: false,
          hotlist: false,
          calendar: false,
          studies: [],
        });
      });
    };
    document.body.appendChild(script);
  }, []);

  return (
    <div className="w-full flex flex-col">
      <div className="h-[600px]">
        <div ref={chartContainerRef} id="tv-main-chart" className="w-full h-full" />
      </div>
       <div className="grid grid-cols-3 h-[200px] bg-gray-900">
        {miniSymbols.map((item, i) => (
          <div key={i} className="bg-gray-800 text-xs text-white">
            <div id={`tv-mini-${i}`} ref={(el) => (miniChartRefs.current[i] = el)} className="h-full"/>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Chart;
