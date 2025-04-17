# 🔧 Crypto Trading Bot with ML Entry Strategy
# ----------------------------------------------------------
# 이 코드는 Binance 선물 시장에서 머신러닝 기반으로 자동 트레이딩을 수행하는 봇입니다.
# 두 개의 XGBoost 모델을 사용합니다:
# 1. trend_model_xgb_<interval>.pkl → 시장 추세 (상승/하락/횡보) 예측
# 2. entry_strategy_model.pkl → 추세 진입 or 역추세 진입 판단
#
# 주요 기능:
# - 실시간 추세 분석 및 진입 판단
# - TP/SL 자동 설정
# - 변동성 필터
# - 정기적인 모델 재학습 (trend: 매일, entry: 매주 월요일 00:10)
# - Telegram 알림 연동
#
# 주요 용어:
# - TP (Take Profit): 목표 수익 도달 시 자동 청산
# - SL (Stop Loss): 손실 제한 도달 시 자동 청산
# - 추세 진입: 추세 방향으로 진입 (예: 상승 추세 → 롱)
# - 역추세 진입: 추세 반대로 진입 (예: 상승 추세 → 숏)
# ----------------------------------------------------------

# 라이브러리 임포트
import asyncio
import numpy as np
import pandas as pd
import matplotlib as mpl
import logging
import joblib
import subprocess
import os
import csv
from sklearn.cluster import KMeans
from telegram import Bot
from telegram.request import HTTPXRequest
from binance.client import Client
from datetime import datetime, timedelta, timezone, time

# 외부 설정파일 및 학습 함수 import
from train_entry_strategy_model_from_csv import train_entry_strategy_from_csv
from config import BINANCE_API_KEY, BINANCE_API_SECRET, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

# matplotlib 폰트 및 마이너스 깨짐 방지 설정
mpl.rcParams['font.family'] = 'AppleGothic'
mpl.rcParams['axes.unicode_minus'] = False

# Binance, Telegram 봇 클라이언트 생성
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
bot = Bot(token=TELEGRAM_BOT_TOKEN, request=HTTPXRequest(connect_timeout=10.0, read_timeout=10.0))

# 포지션 및 거래 상태 전역 변수
position_state = None  # 현재 포지션: 'long', 'short', 또는 None
entry_price = None     # 진입 가격
tp_order_id = None     # TP 주문 ID
sl_order_id = None     # SL 주문 ID
quantity = 0.05        # 거래 수량 (예: 0.05 BTC)

# 전략 설정 (기본 TP/SL 및 리스크 제한)
TP_PERCENT = 1.0        # 목표 수익률 (Take Profit)
SL_PERCENT = 0.5        # 손절 기준 (Stop Loss)
VOLATILITY_THRESHOLD = 2.5  # 변동성 기준 (%)
volatility_blocked = False  # 변동성 초과 시 거래 금지
cumulative_pnl = 0.0        # 누적 수익률
STOP_LOSS_LIMIT = -10.0     # 누적 손실 한계 (이하일 경우 중단)
last_reset_month = datetime.now().month

# 시간대 설정 (KST: 한국 시간)
KST = timezone(timedelta(hours=9))

# 트레이딩 인터벌 설정 ('1m', '5m', '15m', '1h' 등)
TRADING_INTERVAL = '1m'

# 로깅 레벨 설정
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# 바이낸스에서 캔들 데이터 불러오기
def get_klines(symbol='BTCUSDT', interval=TRADING_INTERVAL, limit=100):
    klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'num_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    return df

# RSI 계산 함수 (14일 기준)
def compute_rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# 현재 시각 기준 다음 봉 마감까지 남은 시간 계산
def get_next_bar_close_time(interval_str='15m', buffer_seconds=5):
    now = datetime.now(timezone.utc)
    interval_minutes = interval_to_minutes(interval_str)

    # 현재 시각에서 interval 단위로 올림된 시간 계산
    total_minutes = now.hour * 60 + now.minute
    next_total_minutes = ((total_minutes // interval_minutes) + 1) * interval_minutes

    # 마감 시간 계산
    next_bar_hour = next_total_minutes // 60
    next_bar_minute = next_total_minutes % 60

    # 다음 봉의 마감 시각 (오늘 또는 내일로 넘어갈 수도 있음)
    next_bar_time = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(hours=next_bar_hour, minutes=next_bar_minute)

    return (next_bar_time - now).total_seconds() + buffer_seconds

# 인터벌 문자열 ('5m', '1h')을 분 단위 정수로 변환
def interval_to_minutes(interval_str):
    if interval_str.endswith('m'):
        return int(interval_str[:-1])
    elif interval_str.endswith('h'):
        return int(interval_str[:-1]) * 60
    elif interval_str.endswith('d'):
        return int(interval_str[:-1]) * 1440
    else:
        raise ValueError("Invalid interval format")

# 추세 예측 (trend_model_xgb 사용)
# 결과: trend (0: 하락, 1: 횡보, 2: 상승), confidence (확률)
def predict_trend_with_proba(df: pd.DataFrame, model_path=f"trend_model_xgb_{TRADING_INTERVAL}.pkl"):
    from xgboost import XGBClassifier

    df = df.copy()
    df['return'] = df['close'].pct_change()
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ma_ratio'] = df['ma5'] / df['ma10']
    df['volatility'] = df['return'].rolling(window=5).std()
    df['rsi'] = compute_rsi(df['close'], 14)

    # 추가된 부분: MACD & Signal
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()

    # 추가된 부분: Bollinger Band Width
    ma20 = df['close'].rolling(window=20).mean()
    std20 = df['close'].rolling(window=20).std()
    df['bb_width'] = (2 * std20) / ma20

    df = df.dropna()

    if len(df) < 1:
        return 1, 0.0

    expected_features = ['ma_ratio', 'volatility', 'rsi', 'macd', 'macd_signal', 'bb_width']
    if not all(col in df.columns for col in expected_features):
        logging.error("❌ 필요한 feature가 누락되었습니다. 재학습이 필요할 수 있습니다.")
        return 1, 0.0

    features = df[expected_features].iloc[-1:]

    try:
        model = joblib.load(model_path)
        if not hasattr(model, 'predict_proba'):
            raise TypeError("모델이 'predict_proba'를 지원하지 않음")
    except Exception as e:
        logging.error(f"⚠️ 모델 로딩 실패 또는 유효하지 않음: {e} → 외부 학습 스크립트 실행")
        if not retrain_model_by_script("train_trend_model_xgb.py"):
            return 1, 0.0
        try:
            model = joblib.load(model_path)
        except Exception as e:
            logging.error(f"❌ 모델 재로딩 실패: {e}")
            return 1, 0.0

    try:
        proba = model.predict_proba(features)[0]
        pred = int(np.argmax(proba))
        confidence = float(proba[pred])
    except Exception as e:
        logging.error(f"❌ 예측 실패: {e}")
        return 1, 0.0

    return pred, confidence

# 진입 전략 학습용 데이터셋 생성
# 출력: features + label (0: 역추세, 1: 추세)
def generate_entry_strategy_dataset(df: pd.DataFrame, trend_model_path: str, future_window: int = 10):
    from xgboost import XGBClassifier

    data = []
    df = df.copy()

    # feature 생성
    df['return'] = df['close'].pct_change()
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ma_ratio'] = df['ma5'] / df['ma10']
    df['volatility'] = df['return'].rolling(window=5).std()
    df['rsi'] = compute_rsi(df['close'], 14)
    df['ema12'] = df['close'].ewm(span=12).mean()
    df['ema26'] = df['close'].ewm(span=26).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    ma20 = df['close'].rolling(window=20).mean()
    std20 = df['close'].rolling(window=20).std()
    df['bb_width'] = (2 * std20) / ma20

    # 지지/저항 계산
    support, resistance = calculate_support_resistance(df)

    model = joblib.load(trend_model_path)

    for i in range(30, len(df) - future_window):
        row = df.iloc[i]
        current_price = row['close']

        # 지지/저항 근접 조건 (±0.3%)
        support_dist = abs(current_price - support) / current_price * 100
        resistance_dist = abs(current_price - resistance) / current_price * 100
        if support_dist > 0.3 and resistance_dist > 0.3:
            continue

        # trend 예측
        features = df[['ma_ratio', 'volatility', 'rsi', 'macd', 'macd_signal', 'bb_width']].iloc[i:i+1]
        proba = model.predict_proba(features)[0]
        trend = int(np.argmax(proba))
        confidence = proba[trend]

        if trend == 1:
            continue  # 횡보는 생략

        # 수익률 시뮬레이션 (future_window 기간 동안 최대 수익/손실 계산)
        future_prices = df['close'].iloc[i+1:i+future_window+1].values
        entry = current_price

        # 추세 진입 vs 역추세 진입 결과 계산
        if trend == 2:  # 상승
            pnl_trend = (max(future_prices) - entry) / entry * 100
            pnl_counter = (entry - min(future_prices)) / entry * 100
        elif trend == 0:  # 하락
            pnl_trend = (entry - min(future_prices)) / entry * 100
            pnl_counter = (max(future_prices) - entry) / entry * 100

        # 라벨 결정: 누가 더 나은 수익률을 냈는가?
        label = 1 if pnl_trend > pnl_counter else 0

        data.append({
            'ma_ratio': row['ma_ratio'],
            'volatility': row['volatility'],
            'rsi': row['rsi'],
            'macd': row['macd'],
            'macd_signal': row['macd_signal'],
            'bb_width': row['bb_width'],
            'dist_support': support_dist,
            'dist_resistance': resistance_dist,
            'trend': trend,
            'confidence': confidence,
            'label': label
        })

    return pd.DataFrame(data)

def load_last_retrain_date():
    if os.path.exists("last_trend_retrain.txt"):
        with open("last_trend_retrain.txt", "r") as f:
            return f.read().strip()
    return None

def save_last_retrain_date(date_str):
    with open("last_trend_retrain.txt", "w") as f:
        f.write(date_str)

# 매일 trend 모델 재학습 여부 확인 및 실행
async def maybe_retrain_daily():
    now_kst = datetime.now(KST)
    target_time = time(hour=0, minute=1)  # KST 기준 00:01
    today_str = now_kst.date().isoformat()
    last_retrain_str = load_last_retrain_date()

    if (
        now_kst.time() >= target_time and
        (last_retrain_str is None or last_retrain_str < today_str)
    ):
        await send_telegram_message("🔁 매일 Trend 모델 재학습 시작")
        if retrain_model_by_script("train_trend_model_xgb.py"):
            await send_telegram_message("✅ Trend 모델 재학습 완료")
            save_last_retrain_date(today_str)
        else:
            await send_telegram_message("❌ Trend 모델 재학습 실패")

def load_last_entry_retrain_date():
    if os.path.exists("last_entry_retrain.txt"):
        with open("last_entry_retrain.txt", "r") as f:
            return f.read().strip()
    return None

def save_last_entry_retrain_date(date_str):
    with open("last_entry_retrain.txt", "w") as f:
        f.write(date_str)

# 매주 월요일 00:10 entry 전략 재학습
async def maybe_retrain_entry_strategy():
    now_kst = datetime.now(KST)
    target_time = time(hour=0, minute=10)  # 월요일 00:10 기준
    today_str = now_kst.date().isoformat()
    last_entry_str = load_last_entry_retrain_date()

    # 월요일 + 00:10 이후 + 아직 안 한 경우만 실행
    if (
        now_kst.weekday() == 0 and  # 0 = Monday
        now_kst.time() >= target_time and
        (last_entry_str is None or last_entry_str < today_str)
    ):
        intervals = ['5m', '15m', '1h']

        for interval in intervals:
            try:
                await send_telegram_message(f"🔁 [{interval}] 전략 모델 재학습 시작")

                # 캔들 데이터 가져오기
                limit = get_auto_limit(interval=interval)
                df = get_klines(symbol='BTCUSDT', interval=interval, limit=limit)

                # 학습 데이터셋 생성
                dataset = generate_entry_strategy_dataset(
                    df,
                    trend_model_path=f"trend_model_xgb_{interval}.pkl"
                )

                if dataset.empty:
                    await send_telegram_message(f"⚠️ [{interval}] 학습 데이터 부족으로 재학습 생략")
                    continue

                # CSV 저장 (선택, 분석용)
                csv_path = f"entry_strategy_dataset_{interval}.csv"
                dataset.to_csv(csv_path, index=False)

                # 모델 재학습 실행
                from train_entry_strategy_model_from_csv import train_entry_strategy_from_csv
                train_entry_strategy_from_csv(csv_path=csv_path, interval=interval)

                await send_telegram_message(f"✅ [{interval}] 전략 모델 재학습 완료")
            except Exception as e:
                await send_telegram_message(f"❌ [{interval}] 전략 모델 재학습 실패: {e}")

        save_last_entry_retrain_date(today_str)

def retrain_model_by_script(script_path="train_trend_model_xgb.py"):
    try:
        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            encoding='utf-8',
            check=True
        )
        logging.info(f"✅ 모델 재학습 성공")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ 모델 재학습 실패\n{e.stderr}")
        return False
    
def get_auto_limit(interval: str) -> int:
    if interval == '1m':
        return 1500
    elif interval == '5m':
        return 1000
    elif interval == '15m':
        return 1000
    elif interval == '1h':
        return 1000
    elif interval == '4h':
        return 500
    elif interval == '1d':
        return 365
    else:
        return 1000  # 기본값

def predict_trend_text(trend: int) -> str:
    if trend == 2:
        return "상승 📈"
    elif trend == 0:
        return "하락 📉"
    else:
        return "횡보 😐"

def calculate_support_resistance(df, n_clusters=6):
    df['rounded_price'] = df['close'].astype(int)
    grouped = df.groupby('rounded_price')['volume'].sum().reset_index()
    X = grouped[['rounded_price', 'volume']].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(X)
    centers = kmeans.cluster_centers_
    prices = np.sort(centers[:, 0].astype(int))
    support = prices[0]
    resistance = prices[-1]
    return support, resistance

def analyze_volatility(df):
    returns = df['close'].pct_change().dropna()
    volatility = returns.std() * 100
    return volatility
 
def place_order(side: str, quantity: float):
    order = client.futures_create_order(
        symbol='BTCUSDT',
        side='BUY' if side == 'long' else 'SELL',
        type='MARKET',
        quantity=quantity
    )
    return order

def close_position(current_side: str, quantity: float):
    close_side = 'SELL' if current_side == 'long' else 'BUY'
    order = client.futures_create_order(
        symbol='BTCUSDT',
        side=close_side,
        type='MARKET',
        quantity=quantity
    )
    return order

def get_tick_size(symbol='BTCUSDT'):
    info = client.futures_exchange_info()
    for s in info['symbols']:
        if s['symbol'] == symbol:
            for f in s['filters']:
                if f['filterType'] == 'PRICE_FILTER':
                    return float(f['tickSize'])
    raise ValueError("Tick size not found.")

def round_to_tick(price, tick_size):
    return round(round(price / tick_size) * tick_size, 8)

def place_tp_sl_orders(entry_price: float, side: str, quantity: float):
    tick_size = get_tick_size('BTCUSDT')

    tp_price = entry_price * (1 + TP_PERCENT / 100) if side == 'long' else entry_price * (1 - TP_PERCENT / 100)
    sl_price = entry_price * (1 - SL_PERCENT / 100) if side == 'long' else entry_price * (1 + SL_PERCENT / 100)

    tp_price = str(round_to_tick(tp_price, tick_size))
    sl_price = str(round_to_tick(sl_price, tick_size))

    tp_order = client.futures_create_order(
        symbol='BTCUSDT',
        side='SELL' if side == 'long' else 'BUY',
        type='LIMIT',
        price=tp_price,
        quantity=quantity,
        timeInForce='GTC',
        reduceOnly=True
    )

    sl_order = client.futures_create_order(
        symbol='BTCUSDT',
        side='SELL' if side == 'long' else 'BUY',
        type='STOP_MARKET',
        stopPrice=sl_price,
        quantity=quantity,
        reduceOnly=True
    )

    return tp_order['orderId'], sl_order['orderId']

def cancel_order(symbol: str):
    try:
        client.futures_cancel_all_open_orders(symbol=symbol)
        logging.info(f"✅ 모든 열린 주문 취소 완료 ({symbol})")
    except Exception as e:
        logging.error(f"❌ 전체 주문 취소 실패: {e}")

async def send_telegram_message(message: str):
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)

def get_current_position(symbol='BTCUSDT'):
    positions = client.futures_position_information(symbol=symbol)
    for p in positions:
        pos_amt = float(p['positionAmt'])
        if abs(pos_amt) > 1e-5:  # 아주 작은 수량 무시
            side = 'long' if pos_amt > 0 else 'short'
            entry_price = float(p['entryPrice'])
            return side, entry_price
    return None, None

def check_existing_tp_sl_orders(symbol='BTCUSDT'):
    open_orders = client.futures_get_open_orders(symbol=symbol)
    tp_exists = any(o['type'] == 'LIMIT' and o['reduceOnly'] for o in open_orders)
    sl_exists = any(o['type'] == 'STOP_MARKET' and o['reduceOnly'] for o in open_orders)
    return tp_exists, sl_exists

def load_sorted_intervals_from_backtest(filepath='backtest_summary.csv') -> list:
    try:
        df = pd.read_csv(filepath)
        sorted_df = df.sort_values(by='PnL', ascending=False)
        return sorted_df['Interval'].tolist()
    except Exception as e:
        logging.warning(f"📄 백테스트 결과 로딩 실패 → 기본 순서 사용: {e}")
        return ['15m', '5m', '1h']

async def multi_tf_trading_loop():
    global position_state, entry_price, volatility_blocked, cumulative_pnl
    global TP_PERCENT, SL_PERCENT, last_reset_month, tp_order_id, sl_order_id

    symbol = 'BTCUSDT'
    
    try:
        intervals = load_sorted_intervals_from_backtest()  # 자동 우선순위 반영
    except:
        intervals = ['15m', '5m', '1h']  # fallback
    
    support = resistance = None
    just_entered = False
    
    # 복구 로직 (단, just_entered인 경우는 스킵)
    if not just_entered and position_state is None and entry_price is None:
        position_state, entry_price = get_current_position()
        if position_state:
            await send_telegram_message(f"🔁 기존 포지션 복구: {position_state.upper()} @ {entry_price}")
            tp_exists, sl_exists = check_existing_tp_sl_orders()
            if not tp_exists or not sl_exists:
                tp_order_id, sl_order_id = place_tp_sl_orders(entry_price, position_state, quantity)
                await send_telegram_message("🛠️ 누락된 TP/SL 주문을 재설정했습니다.")

    # 월별 초기화 / 손실 중단
    current_month = datetime.now().month
    if current_month != last_reset_month:
        last_reset_month = current_month
        cumulative_pnl = 0.0
        await send_telegram_message("🔄 새 달이 시작되어 누적 수익률을 초기화합니다.")

    if cumulative_pnl <= STOP_LOSS_LIMIT:
        await send_telegram_message(f"🛑 누적 손실 {cumulative_pnl:.2f}%로 자동 중단됩니다.")
        raise SystemExit

    # 중복 진입 방지
    if position_state is not None:
        # 실물 포지션 조회
        actual_pos, actual_entry = get_current_position()

        if actual_pos is None:
            position_state = None
            entry_price = None
            await send_telegram_message("🔁 기존 포지션 종료")
            logging.warning("⚠️ Binance에는 포지션 없지만 상태 남아 있음 → 초기화")
        else:
            logging.info("중복 진입 방지: 이미 포지션이 존재함")
            return  # 이미 포지션 있음 → 진입 불가

    for interval in intervals:
        trend_model_path = f"trend_model_xgb_{interval}.pkl"
        entry_model_path = f"entry_strategy_model_{interval}.pkl"
        entry_model = joblib.load(entry_model_path)
        
        df = get_klines(symbol=symbol, interval=interval, limit=get_auto_limit(interval))
        support, resistance = calculate_support_resistance(df)

        # 변동성 필터
        volatility = analyze_volatility(df)
        if volatility >= VOLATILITY_THRESHOLD and position_state is None:
            if not volatility_blocked:
                await send_telegram_message(f"⚠️ 변동성 과도 ({volatility:.2f}%) → 포지션 진입 회피 중")
                volatility_blocked = True
            return
        elif volatility < VOLATILITY_THRESHOLD and volatility_blocked:
            await send_telegram_message(f"✅ 변동성 정상화 ({volatility:.2f}%) → 진입 가능 상태로 전환")
            volatility_blocked = False

        trend, confidence = predict_trend_with_proba(df, model_path=trend_model_path)
        if trend == 1:
            logging.info(f"😐 [{interval}] 횡보 예측 → 진입 회피")
            continue
        
        if confidence < 0.6:
            logging.info(f"❌ [{interval}] 신뢰도 낮음({confidence * 100:.2f}%) → 진입 회피")
            continue

        # entry 전략 예측을 위한 feature 생성
        entry_features_df = generate_entry_strategy_dataset(df, trend_model_path=trend_model_path)
        if entry_features_df.empty:
            logging.info(f"🚫 [{interval}] 유효한 진입 포인트 없음 → 회피")
            continue

        entry_row = entry_features_df.iloc[-1]
        X_entry = entry_row.drop('label', errors='ignore').values.reshape(1, -1)
        strategy = int(entry_model.predict(X_entry)[0])  # 0 = 역추세, 1 = 추세

        def trend_to_signal(t): return 'long' if t == 2 else 'short' if t == 0 else None
        def reverse_signal(s): return 'short' if s == 'long' else 'long'

        signal = trend_to_signal(trend) if strategy == 1 else reverse_signal(trend_to_signal(trend))
        if signal is None:
            logging.info(f"🚫 [{interval}] 진입 신호 없음 (None)")
            continue

        # confidence 기반 TP/SL 조정
        if confidence >= 0.8:
            TP_PERCENT, SL_PERCENT = 1.8, 0.3
        elif confidence >= 0.6:
            TP_PERCENT, SL_PERCENT = 1.0, 0.5
        else:
            TP_PERCENT, SL_PERCENT = 0.7, 0.5

        await send_telegram_message(
            f"📡 [{interval}] 진입 신호 발생\n"
            f"🧠 추세 예측: {predict_trend_text(trend)} / 전략: {'추세' if strategy == 1 else '역추세'}\n"
            f"📊 신뢰도: {confidence * 100:.2f}%\n"
            f"🎯 TP: {TP_PERCENT}%, SL: {SL_PERCENT}%\n"
            f"지지선: {support} / 저항선: {resistance}\n"
            f"📌 진입 방향: {signal.upper()}"
        )

        # if trend == 2 and signal == 'short':
        #     await send_telegram_message(f"📈 [{interval}] 상승 추세인데 숏 진입 시도 → 회피")
        #     continue
        # if trend == 0 and signal == 'long':
        #     await send_telegram_message(f"📉 [{interval}] 하락 추세인데 롱 진입 시도 → 회피")
        #     continue

        current_price = float(client.futures_mark_price(symbol=symbol)['markPrice'])

        # 포지션 진입
        try:
            order = place_order(signal, quantity)
        except Exception as e:
            await send_telegram_message(f"❌ 주문 실패: {e}")
            return
        await asyncio.sleep(0.5)  # 체결 대기 (Binance 응답 속도 고려)

        # 포지션 체결 여부 확인 (최대 3회 재조회)
        retries = 0
        position_side = None
        real_entry_price = None

        # 체결 확인
        while retries < 3:
            position_side, real_entry_price = get_current_position()
            if position_side:
                break
            retries += 1
            logging.warning(f"📡 포지션 진입 확인 실패 (시도 {retries}) → 1초 후 재시도")
            await asyncio.sleep(1.0)

        if not position_side:
            await send_telegram_message("❌ 포지션 진입 실패 감지 → 트레이딩 스킵")
            return

        # TP/SL 설정 (최대 3회 재시도)
        retries = 0
        while retries < 3:
            try:
                tp_order_id, sl_order_id = place_tp_sl_orders(real_entry_price, signal, quantity)
                logging.info("✅ TP/SL 주문 설정 완료")
                break
            except Exception as e:
                retries += 1
                logging.warning(f"❌ TP/SL 설정 실패 (1초 후 재시도): {e}")
                await asyncio.sleep(1.0)

        # TP/SL 설정 실패 시 포지션 강제 종료
        if retries == 3:
            close_position(signal, quantity)
            await send_telegram_message("🚨 TP/SL 주문 실패 → 포지션 강제 종료")
            return

        # 모든 게 정상이면 상태 저장
        position_state = signal
        entry_price = real_entry_price
        just_entered = True

        tp_price = round(entry_price * (1 + TP_PERCENT / 100), 2) if signal == 'long' else round(entry_price * (1 - TP_PERCENT / 100), 2)
        sl_price = round(entry_price * (1 - SL_PERCENT / 100), 2) if signal == 'long' else round(entry_price * (1 + SL_PERCENT / 100), 2)

        await send_telegram_message(
            f"🚀 [{interval}] 진입 완료: {signal.upper()} @ {entry_price}\n"
            f"🎯 TP: {tp_price}\n"
            f"⚠️ SL: {sl_price}"
        )

        logging.info(f"✅ [{interval}] 진입 완료: {signal.upper()} @ {entry_price:.2f} | TP: {tp_price}, SL: {sl_price}")
        break  # 하나라도 진입했으면 루프 종료

async def start_bot():
    await send_telegram_message(f"⏳ 프로그램 시작.")
    logging.info("⏳ 프로그램 시작됨. 다음 봉 마감까지 대기 중...")

    # 자동 백테스트 스케줄러 백그라운드 실행
    asyncio.create_task(auto_backtest_scheduler())

    while True:
        await maybe_retrain_daily()                # 기존 trend 모델 재학습
        await maybe_retrain_entry_strategy()       # 새로운 entry 전략 모델 재학습    

        # 다음 봉 마감 시점 계산 (예: 현재 시각이 09:14:53 → 09:15:00 마감까지 7초 남음)
        sleep_sec = get_next_bar_close_time(TRADING_INTERVAL)
        logging.info(f"⏱️ 다음 봉 마감까지 {sleep_sec:.2f}초 대기...")
        await asyncio.sleep(sleep_sec)

        try:
            # await trading_loop()
            await multi_tf_trading_loop()
        except SystemExit:
            break
        except Exception as e:
            await send_telegram_message(f"❌ 오류 발생: {e}")

def predict_trend_sync(df: pd.DataFrame, model_path=f"trend_model_xgb_{TRADING_INTERVAL}.pkl") -> tuple[int, float]:
    df = df.copy()
    df['return'] = df['close'].pct_change()
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ma_ratio'] = df['ma5'] / df['ma10']
    df['volatility'] = df['return'].rolling(window=5).std()
    df['rsi'] = compute_rsi(df['close'], 14)

    # 추가된 부분: MACD & Signal
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()

    # 추가된 부분: Bollinger Band Width
    ma20 = df['close'].rolling(window=20).mean()
    std20 = df['close'].rolling(window=20).std()
    df['bb_width'] = (2 * std20) / ma20

    df = df.dropna()

    if len(df) < 1:
        return 1, 0.0

    expected_features = ['ma_ratio', 'volatility', 'rsi', 'macd', 'macd_signal', 'bb_width']
    if not all(col in df.columns for col in expected_features):
        logging.error("❌ 필요한 feature가 누락되었습니다. 재학습이 필요할 수 있습니다.")
        return 1, 0.0

    features = df[expected_features]

    try:
        model = joblib.load(model_path)
        if not hasattr(model, 'predict_proba'):
            raise TypeError("모델이 'predict_proba'를 지원하지 않음")
    except Exception as e:
        logging.error(f"❌ 모델 로딩 실패 또는 유효하지 않음: {e}")
        return 1, 0.0

    try:
        proba = model.predict_proba(features.iloc[-1:])[0]
        pred = int(np.argmax(proba))
        confidence = float(proba[pred])
    except Exception as e:
        logging.error(f"❌ 예측 실패: {e}")
        return 1, 0.0

    return pred, confidence

async def run_backtest_and_save(intervals: list, csv_path='backtest_summary.csv'):
    # 기존 결과 불러오기
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
    else:
        existing_df = pd.DataFrame(columns=["Interval", "PnL"])

    # 새 백테스트 실행 결과 저장
    new_data = []
    for interval in intervals:
        pnl = await backtest_bot(interval, False)
        new_data.append({'Interval': interval, 'PnL': pnl})

    new_df = pd.DataFrame(new_data)

    # 기존 결과에서 새 interval 제외하고 병합
    merged_df = pd.concat([
        existing_df[~existing_df['Interval'].isin(new_df['Interval'])],
        new_df
    ])

    # 저장
    merged_df.to_csv(csv_path, index=False)
    logging.info(f"✅ 백테스트 결과 갱신 완료 → {csv_path}")

async def auto_backtest_scheduler():
    while True:
        now = datetime.now(KST)

        # 매시 정각 (00, 01, 02...)마다 5m, 15m 백테스트
        if now.minute == 0:
            await run_backtest_and_save(['5m', '15m'])

        # 매일 00:30에는 1h 백테스트
        if now.hour == 0 and now.minute == 30:
            await run_backtest_and_save(['1h'])

        await asyncio.sleep(60)  # 1분마다 확인

async def run_all_backtests():
    intervals = ['5m', '15m', '1h']
    summary_results = {}

    for interval in intervals:
        logging.info(f"\n🧪 [{interval}] 백테스트 시작\n")
        pnl = await backtest_bot(interval=interval)
        summary_results[interval] = pnl

    # 결과 요약 출력
    logging.info("\n📊 전체 백테스트 요약\n")
    for interval, pnl in summary_results.items():
        sign = "+" if pnl >= 0 else ""
        logging.info(f"⏱ {interval:>3}  →  누적 PnL: {sign}{pnl:.2f}%")

def predict_entry_strategy_from_row(row: pd.Series, model_path: str):
    import joblib

    model = joblib.load(model_path)
    features = row.drop(labels=['label'], errors='ignore').values.reshape(1, -1)
    pred = model.predict(features)
    return int(pred[0])  # 0 = 역추세, 1 = 추세

summary_results = {}

async def backtest_bot(interval='5m', isLogShow=True) -> float:
    import joblib
    global position_state, entry_price, volatility_blocked, cumulative_pnl
    global TP_PERCENT, SL_PERCENT, last_reset_month, tp_order_id, sl_order_id

    limit = get_auto_limit(interval)
    df = get_klines(symbol='BTCUSDT', interval=interval, limit=limit)
    
    if isLogShow:
        logging.info(f"\n📊 백테스트 시작: {interval} / 캔들 수: {limit}개\n")

    trend_model_path = f"trend_model_xgb_{interval}.pkl"
    entry_model_path = f"entry_strategy_model_{interval}.pkl"
    trend_model = joblib.load(trend_model_path)
    entry_model = joblib.load(entry_model_path)

    def trend_to_signal(trend: int):
        return 'long' if trend == 2 else 'short' if trend == 0 else None

    def reverse_signal(signal: str):
        return 'short' if signal == 'long' else 'long'

    for i in range(100, len(df)):
        sliced_df = df.iloc[:i].copy()
        current_price = sliced_df['close'].iloc[-1]
        timestamp = pd.to_datetime(sliced_df['timestamp'].iloc[-1], unit='ms')

        if cumulative_pnl <= STOP_LOSS_LIMIT:
            if isLogShow:
                logging.info(f"\n🛑 누적 손실 {cumulative_pnl:.2f}%로 자동 종료")
            break

        # 추세 예측 + confidence 체크
        trend, confidence = predict_trend_sync(sliced_df, model_path=trend_model_path)
        if trend == 1 or confidence < 0.6:
            continue

        # TP/SL 조정
        if confidence >= 0.8:
            TP_PERCENT, SL_PERCENT = 1.8, 0.3
        elif confidence >= 0.6:
            TP_PERCENT, SL_PERCENT = 1.0, 0.5
        else:
            TP_PERCENT, SL_PERCENT = 0.7, 0.5

        # 진입 전략 feature 생성
        entry_features_df = generate_entry_strategy_dataset(sliced_df, trend_model_path=trend_model_path)
        if entry_features_df.empty:
            continue

        entry_row = entry_features_df.iloc[-1]
        X_entry = entry_row.drop('label', errors='ignore').values.reshape(1, -1)
        strategy = int(entry_model.predict(X_entry)[0])  # 0 = 역추세, 1 = 추세

        signal = trend_to_signal(trend) if strategy == 1 else reverse_signal(trend_to_signal(trend))
        if signal is None:
            continue

        # 실전 상충 필터
        # if trend == 2 and signal == 'short':
        #     if isLogShow:
        #         logging.info("📈 상승 추세인데 숏 진입 시도 → 회피")
        #     continue
        # elif trend == 0 and signal == 'long':
        #     if isLogShow:
        #         logging.info("📉 하락 추세인데 롱 진입 시도 → 회피")
        #     continue

        # 🔽 포지션 종료 조건 (TP / SL / 신호 소멸)
        if position_state and entry_price:
            change_pct = (current_price - entry_price) / entry_price * 100
            if position_state == 'short':
                change_pct *= -1

            hit_tp = change_pct >= TP_PERCENT
            hit_sl = change_pct <= -SL_PERCENT
            signal_disappeared = signal is None  # 신호 자체 사라짐

            if hit_tp or hit_sl or signal_disappeared:
                label = (
                    "🎯 TP" if hit_tp else
                    "⚠️ SL" if hit_sl else
                    "❌ 신호 소멸"
                )
                cumulative_pnl += change_pct
                if isLogShow:
                    logging.info(
                        f"{label} → {position_state.upper()} 종료 | "
                        f"PnL: {change_pct:.2f}%, 누적: {cumulative_pnl:.2f}%"
                    )
                position_state = None
                entry_price = None
                continue

        # 🔼 진입 조건 (포지션 없고, 조건 충족)
        if not volatility_blocked and position_state is None:
            position_state = signal
            entry_price = current_price
            if isLogShow:
                logging.info(
                    f"\n🧠 {timestamp} | 추세: {trend} / 전략: {'추세' if strategy == 1 else '역추세'} / "
                    f"방향: {signal.upper()} / 신뢰도: {confidence:.2f}"
                )
                logging.info(
                    f"🔥 진입 @ {entry_price:.2f} | TP: {TP_PERCENT}%, SL: {SL_PERCENT}%"
                )
            continue

    if isLogShow:
        logging.info(f"\n✅ 백테스트 종료 → 최종 누적 PnL: {cumulative_pnl:.2f}%\n")

    return cumulative_pnl

if __name__ == "__main__":
    mode = input("실행 모드 선택 (live / backtest / all_backtest): ").strip()
    if mode == "live":
        asyncio.run(start_bot())
    elif mode == "backtest":
        asyncio.run(backtest_bot(interval=TRADING_INTERVAL))
    elif mode == "all_backtest":
        asyncio.run(run_all_backtests())