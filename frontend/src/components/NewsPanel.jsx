import React, { useEffect, useState } from "react";

const NewsPanel = () => {
  const [news, setNews] = useState([]);

  const fetchNews = async () => {
    try {
      const res = await fetch("http://localhost:4000/api/coinness");
      const json = await res.json();
      setNews(json.slice(0, 5)); // 최대 5개만
    } catch (err) {
      console.error("뉴스 fetch 실패:", err);
    }
  };

  useEffect(() => {
    fetchNews();
    const interval = setInterval(fetchNews, 60000);
    return () => clearInterval(interval);
  }, []);

  const formatTime = (iso) => {
    const date = new Date(iso);
    const mm = String(date.getMonth() + 1).padStart(2, "0");
    const dd = String(date.getDate()).padStart(2, "0");
    const hh = String(date.getHours()).padStart(2, "0");
    const mi = String(date.getMinutes()).padStart(2, "0");
    return `${mm}.${dd}. ${hh}:${mi}`;
  };

  return (
    <div className="h-full text-sm text-white bg-[#0e1117] px-0">
      <h2 className="text-[15px] font-semibold text-gray-300 mb-2">
        실시간 속보 뉴스
      </h2>
      <ul className="space-y-1">
        {news.map((item) => (
          <li
            key={item.id}
            className="border-b border-gray-700 pb-1 hover:bg-[#1a1e24] px-1 rounded cursor-default transition"
          >
            <div className="font-medium truncate leading-snug text-[13px] text-white">
              {item.title}
            </div>
            <div className="text-[11px] text-gray-400 mt-0.5">
              {formatTime(item.time)}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default NewsPanel;