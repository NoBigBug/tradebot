import React, { useEffect, useState, useRef } from "react";

const LogPanel = () => {
  const [logs, setLogs] = useState([]);
  const bottomRef = useRef();

  useEffect(() => {
    window.electronAPI.onBotLog((log) => {
      setLogs((prev) => [...prev.slice(-199), log]); // 최대 200줄
    });
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  return (
    <div className="bg-black text-green-300 text-xs font-mono p-2 h-full overflow-y-auto border border-gray-700 rounded">
      {logs.map((line, idx) => (
        <div key={idx} dangerouslySetInnerHTML={{ __html: line.replace(/\n/g, "<br/>") }} />
      ))}
      <div ref={bottomRef} />
    </div>
  );
};

export default LogPanel;