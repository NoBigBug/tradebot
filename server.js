import express from "express";
import cors from "cors";
import fetch from "node-fetch"; // Node.js v18+ 이면 필요 없음
import dotenv from 'dotenv';
import crypto from 'crypto';
import axios from 'axios';

dotenv.config();

const app = express();
app.use(cors());
const PORT = 4000;

const BASE_URL = 'https://fapi.binance.com';
const apiKey = process.env.BINANCE_API_KEY;
const secret = process.env.BINANCE_SECRET_KEY;

function getSignature(query) {
  return crypto.createHmac('sha256', secret).update(query).digest('hex');
}

app.get('/api/performance-summary', async (req, res) => {
  try {
    const endTime = Date.now();
    const startTime = endTime - 90 * 24 * 60 * 60 * 1000; // 90일 전

    const params = `incomeType=REALIZED_PNL&startTime=${startTime}&endTime=${endTime}&timestamp=${endTime}`;
    const signature = getSignature(params);

    const url = `${BASE_URL}/fapi/v1/income?${params}&signature=${signature}`;
    const { data } = await axios.get(url, {
      headers: { 'X-MBX-APIKEY': apiKey },
    });

    // 날짜별 수익 집계
    const daily = {};
    data.forEach((tx) => {
      const date = new Date(tx.time).toISOString().split('T')[0]; // yyyy-mm-dd
      const profit = parseFloat(tx.income);
      daily[date] = (daily[date] || 0) + profit;
    });

    const today = new Date().toISOString().split('T')[0];
    const yesterday = new Date(Date.now() - 86400000).toISOString().split('T')[0];

    const todayProfit = daily[today] || 0;
    const yesterdayProfit = daily[yesterday] || 0;

    const allProfits = Object.values(daily);
    const wins = allProfits.filter((p) => p > 0).length;
    const winRate = (wins / allProfits.length) * 100;

    const totalProfit = allProfits.reduce((a, b) => a + b, 0);

    res.json({
      yesterdayProfit,
      todayProfit,
      winRate90d: winRate,
      totalProfit90d: totalProfit,
    });
  } catch (err) {
    res.status(500).json({
      error: 'Failed to fetch income data',
      detail: err.message,
    });
  }
});


app.get('/api/binance/positions', async (req, res) => {
  try {
    const timestamp = Date.now();
    const query = `timestamp=${timestamp}`;
    const signature = getSignature(query);

    const response = await axios.get(`${BASE_URL}/fapi/v2/positionRisk?${query}&signature=${signature}`, {
      headers: { 'X-MBX-APIKEY': apiKey }
    });

    // 필요한 정보만 추출
    const positions = response.data
      .filter(p => parseFloat(p.positionAmt) !== 0)
      .map(p => ({
        symbol: p.symbol,
        entryPrice: parseFloat(p.entryPrice),
        positionAmt: parseFloat(p.positionAmt),
        unRealizedProfit: parseFloat(p.unRealizedProfit),
        leverage: p.leverage
      }));

    res.json(positions);
  } catch (err) {
    res.status(500).json({ error: 'Failed to fetch positions', detail: err.message });
  }
});

const COINNESS_API_URL = "https://api.coinness.com/feed/v1/breaking-news?languageCode=ko";

app.get("/api/coinness", async (req, res) => {
  try {
    const response = await fetch(COINNESS_API_URL);
    const data = await response.json();
    

    const news = (data || []).slice(0, 10).map((item) => ({
      id: item.id,
      title: item.title,
      time: item.publishAt,
      url: `https://coinness.com/news/detail/${item.id}`,
    }));
      
    console.log('news', news);

    res.json(news);
  } catch (error) {
    console.error("❌ CoinNess 뉴스 API 요청 실패:", error.message);
    res.status(500).json({
      error: "뉴스 가져오기 실패",
      detail: error.message,
    });
  }
});

app.listen(PORT, () => {
  console.log(`🚀 백엔드 실행됨: http://localhost:${PORT}`);
});