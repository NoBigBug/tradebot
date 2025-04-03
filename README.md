# 📈 BTCUSDT 자동 트레이딩 봇

Binance Futures + Python 기반의 자동 트레이딩 봇입니다.  
지지/저항 분석 + 변동성 필터 + TP/SL 자동 주문 + 텔레그램 실시간 알림 기능을 포함합니다.

---

### 🧠 주요 기능

- **지지/저항 분석**: `KMeans` 클러스터링 기반 가격 구간 추출
- **변동성 필터링**: 과도한 변동성 구간에서 진입 차단
- **자동 포지션 진입/청산**: TP/SL 설정 포함
- **중복 포지션 방지**
- **Telegram 알림**: 진입, 청산, 오류 발생 시 실시간 알림
- **월별 누적 PnL 관리 및 자동 초기화**
- **Tick Size 대응**: 지정가 주문 오류(-4014) 방지

---

### ⚙️ 사용 기술

- `Python 3.9+`
- `Binance API` (futures)
- `scikit-learn` (KMeans)
- `matplotlib`
- `pandas`
- `asyncio`, `logging`
- `python-telegram-bot`

---

### 🚀 실행 방법

1. `config.py` 생성  
```python
BINANCE_API_KEY = "your_binance_key"
BINANCE_API_SECRET = "your_binance_secret"
TELEGRAM_BOT_TOKEN = "your_telegram_bot_token"
TELEGRAM_CHAT_ID = "your_chat_id"
```

2. 의존성 설치  
```bash
pip install -r requirements.txt
```

3. 실행  
```bash
python new_tradeBot.py
```

---

### 🧪 실행 모드

| 모드 | 설명 |
|------|------|
| `live` | 실시간 매매 실행 (5분마다 동작) |
| `backtest` | 단일 루프 1회 실행하고 종료 (전략 테스트용) |

---

### 📊 전략 로직 요약

- 포지션 진입 조건: 현재가가 지지선 또는 저항선 근처 (`0.3%` 이내)
- TP: 진입가 기준 `+1.0%` 도달 시 익절  
- SL: 진입가 기준 `-0.5%` 도달 시 손절  
- 누적 손익이 `-10%` 이하 → 전체 봇 자동 종료  
- 누적 손익이 `+10%` 이상 → TP/SL 전략 강화 적용

---

### 🔐 주의사항

- 실계좌로 사용 시 반드시 충분히 백테스트하세요.
- 테스트 중에는 소액 또는 Binance Testnet 사용 권장.
- 해당 봇은 금융 투자 자문이 아닙니다.

---
