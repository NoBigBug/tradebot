import React from "react";
import Chart from "./components/Chart";
import NewsPanel from "./components/NewsPanel";
import InfoPanel from "./components/InfoPanel";
import LogPanel from "./components/LogPanel";

function App() {
  return (
    <div className="w-screen h-screen bg-[#0e1117] text-white font-sans">
      {/* 본문 */}
      <main className="flex h-[802px]">
        {/* 좌측 시장 정보 */}
        <aside className="w-[267px] bg-[#0e1117] border-r border-gray-800 p-1 text-sm">
          <InfoPanel />
        </aside>

        {/* 중앙 차트 */}
        <section className="flex-1 bg-[#0e1117] overflow-hidden">
          <div className="w-full border border-gray-800 overflow-hidden">
            <Chart />
          </div>
        </section>

        {/* 우측 뉴스 + 로그 패널 */}
        <aside className="w-[300px] flex flex-col bg-[#0e1117] border-l border-gray-800 p-1">
          {/* 뉴스 영역 (상단 60%) */}
          <div className="flex-[0.4] overflow-y-auto mb-1">
            <NewsPanel />
          </div>
          {/* 로그 영역 (하단 40%) */}
          <h2 className="text-sm font-semibold text-gray-300 mb-1">tradeBot 로그</h2>
          <div className="flex-[0.6] overflow-y-auto">            
            <LogPanel />
          </div>
        </aside>
      </main>
    </div>
  );
}

export default App;