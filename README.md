# 🤖 Crypto Trading Bot with ML Entry Strategy

머신러닝 기반의 실시간 자동 트레이딩 봇입니다.  
Binance 선물 시장을 대상으로 추세 분석 및 진입 전략을 판단하고, TP/SL까지 자동으로 설정합니다.

---

## 📌 주요 기능

- ✅ **실시간 트렌드 예측** (XGBoost 기반)
- ✅ **추세/역추세 전략 판단**
- ✅ **TP (Take Profit), SL (Stop Loss) 자동 설정**
- ✅ **변동성 필터링 (시장 과열 회피)**
- ✅ **Telegram 실시간 알림 전송**
- ✅ **자동 재학습 (매일 trend, 매주 entry 전략)**  
- ✅ **백테스트 기능 포함**

---

## 📈 모델 설명

- `trend_model_xgb_<interval>.pkl`  
  → 시장 추세 예측 (상승📈 / 하락📉 / 횡보😐)

- `entry_strategy_model.pkl`  
  → 추세로 진입할지, 역추세로 진입할지 판단

---

## ⚙️ 설정파일 (.env 또는 config.py)

```python
BINANCE_API_KEY = "YOUR_BINANCE_API_KEY"
BINANCE_API_SECRET = "YOUR_BINANCE_SECRET"
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"
```

---

## 🚀 실행 방법

```bash
# 라이브 모드 (실시간 트레이딩)
python new_tradeBot.py
> live

# 백테스트 (단일 인터벌)
python new_tradeBot.py
> backtest

# 전체 백테스트 (1m, 5m, 15m, 1h)
python new_tradeBot.py
> all_backtest
```

---

## 🔄 자동 재학습 로직

- **매일 00:01 (KST)** → `trend_model_xgb_<interval>.pkl` 재학습
- **매주 월요일 00:10 (KST)** → `entry_strategy_model.pkl` 재학습

---

## 📊 주요 로직 흐름

1. 실시간 가격/캔들 수집
2. 추세 예측 (상승/하락/횡보)
3. 전략 판단 (추세진입/역추세진입)
4. 진입 여부 결정 (변동성, 신뢰도 필터)
5. 진입 → TP/SL 설정
6. 포지션 종료 판단 (목표 수익/손실 도달 시)
7. Telegram 알림 전송

---

## 🧠 사용 라이브러리

- `xgboost`, `joblib`
- `pandas`, `numpy`, `scikit-learn`
- `binance`, `python-telegram-bot`
- `matplotlib` (시각화 optional)

---

## 📦 데이터 학습

- 학습 데이터는 `get_klines`를 통해 실시간 수집
- `generate_entry_strategy_dataset()` → feature + label 자동 생성
- 학습 결과는 CSV로 저장되어 `train_entry_strategy_model_from_csv.py`로 학습됨

---

## 🔒 주의사항

- **실거래 주의**: Binance 실계정 연결 시 반드시 소량부터 테스트!
- **전략 튜닝 권장**: TP/SL 비율 및 진입 전략은 시장에 따라 조정 가능
- **비공개 키 보안 주의**: API 키는 깃허브에 절대 올리지 마세요!

---

## ✨ 향후 개선 아이디어

- [ ] Web UI 추가
- [ ] 실시간 그래프 대시보드
- [ ] 포지션 수량 자동 조절 (포트폴리오 비중 기반)
- [ ] 더 고도화된 모델 (딥러닝 기반)

---