import React, { useEffect, useState } from "react";
import dayjs from "dayjs";

const InfoPanel = () => {
  const [summary, setSummary] = useState(null);
  const [positions, setPositions] = useState([]);
  const [currentTime, setCurrentTime] = useState(dayjs().format("HH:mm:ss"));
  const [btcPrice, setBtcPrice] = useState(null);

  useEffect(() => {
    const ws = new WebSocket("wss://fstream.binance.com/ws/btcusdt@markPrice");
  
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      const price = parseFloat(data.p);
      setBtcPrice(price);
    };

    return () => ws.close();
  }, [btcPrice]);

  useEffect(() => {
      const timer = setInterval(() => {
        setCurrentTime(dayjs().format("HH:mm:ss"));
      }, 1000);
      return () => clearInterval(timer);
    }, []);

  useEffect(() => {
    const fetchSummary = async () => {
      try {
        const res = await fetch("http://localhost:4000/api/performance-summary");
        const data = await res.json();
        setSummary(data);
      } catch (err) {
        console.error("수익 요약 fetch 실패", err);
      }
    };
    fetchSummary();
    const interval = setInterval(fetchSummary, 60000);
    return () => clearInterval(interval);
  }, []);

  const fetchPositions = async () => {
    try {
      const res = await fetch("http://localhost:4000/api/binance/positions");
      const data = await res.json();
      setPositions(data);
    } catch (err) {
      console.error("포지션 정보 fetch 실패:", err);
    }
  };

  useEffect(() => {
    fetchPositions(); // 초기 요청
    const interval = setInterval(fetchPositions, 5000); // 10초마다 갱신

    return () => clearInterval(interval); // cleanup
  }, []);

  return (
    <aside className="w-[260px] bg-[#0e1117] text-white text-xs p-1 flex flex-col space-y-4">
      {/* 가격 및 시계 */}
      <div className="mt-auto pt-2">
        <div className="flex justify-between items-end">
          {/* 왼쪽: 가격 */}
          <div className="text-left">
            <div className="text-center text-[10px] text-gray-400 py-2">BTCUSDT Price</div>
            <div className="text-pink-400 font-bold text-[25px]">{btcPrice ? `${btcPrice.toFixed(2)}` : ""}</div>
          </div>

          {/* 오른쪽: 시계 */}
          <div className="text-right">
            <div className="text-center text-[10px] text-gray-400 py-2">UTC+9 (Seoul)</div>
            <div className="text-white font-mono text-[25px]">{currentTime}</div>
          </div>
        </div>
      </div>

      {/* 수익 요약 */}
      <div className="bg-gray-800 p-2 rounded space-y-1 text-[12px]">
        <div className="flex justify-between">
          <span className="text-gray-400">어제 수익</span>
          <span className="font-semibold text-right">
            {summary ? `${summary.yesterdayProfit.toFixed(1)} USDT` : '...'}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">오늘 수익</span>
          <span className={`font-semibold ${summary?.todayProfit < 0 ? 'text-red-400' : 'text-green-400'}`}>
            {summary ? `${summary.todayProfit.toFixed(1)} USDT` : '...'}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">90일 승률</span>
          <span className="font-semibold">
            {summary ? `${summary.winRate90d.toFixed(2)}%` : '...'}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">90일 총 수익</span>
          <span className={`font-semibold" ${summary?.totalProfit90d < 0 ? 'text-red-400' : 'text-green-400'}`}>
            {summary ? `${summary.totalProfit90d.toFixed(2)} USDT` : '...'}
          </span>
        </div>
      </div>

      {/* 포지션 테이블 */}
      <div className="text-[12px] space-y-1">
        <div className="grid grid-cols-[30px_1fr_1fr_1fr_1fr] text-gray-400 font-semibold">
          <div>레버</div>
          <div className="text-center">심볼</div>          
          <div className="text-center">평단</div>
          <div className="text-center">수량</div>
          <div className="text-center">미실현</div>
        </div>
        {positions.map((pos) => (
          <div
            key={pos.symbol}
            className="grid grid-cols-5 text-white"
          >
            <div>
              <span className="text-[10px] bg-teal-700 px-1 py-[1px] rounded whitespace-nowrap">{parseFloat(pos.leverage)}X</span>
            </div>
            <div>{pos.symbol.replace('USDT', '')}</div>
            <div className="text-center  whitespace-nowrap">{pos.entryPrice.toFixed(2)}</div>
            <div className="text-center  whitespace-nowrap">{pos.positionAmt.toFixed(2)}</div>
            <div className={`text-right  whitespace-nowrap ${pos.unRealizedProfit < 0 ? 'text-red-500' : 'text-green-400'}`}>
              {pos.unRealizedProfit.toFixed(2)}
            </div>
          </div>
        ))}
      </div>
    </aside>
  );
};

export default InfoPanel;
