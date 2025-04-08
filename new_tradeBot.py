import asyncio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import logging
import joblib
import subprocess

from sklearn.cluster import KMeans
from telegram import Bot
from telegram.request import HTTPXRequest
from binance.client import Client
from datetime import datetime, timedelta, timezone, time

from config import BINANCE_API_KEY, BINANCE_API_SECRET, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

mpl.rcParams['font.family'] = 'AppleGothic'
mpl.rcParams['axes.unicode_minus'] = False

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
bot = Bot(
    token=TELEGRAM_BOT_TOKEN,
    request=HTTPXRequest(connect_timeout=10.0, read_timeout=10.0)
)

position_state = None
entry_price = None
tp_order_id = None
sl_order_id = None
quantity = 0.05

TP_PERCENT = 1.0
SL_PERCENT = 0.5
VOLATILITY_THRESHOLD = 2.5
volatility_blocked = False
cumulative_pnl = 0.0
STOP_LOSS_LIMIT = -10.0
last_reset_month = datetime.now().month

KST = timezone(timedelta(hours=9))
last_retrain_date = None

# 트레이딩 인터벌 설정 ('1m', '3m', '5m', '15m', '30m', '1h' 등)
TRADING_INTERVAL = '5m'  # Binance 기준 문자열

logging.basicConfig(level=logging.INFO)

async def maybe_retrain_daily():
    global last_retrain_date

    now_kst = datetime.now(KST)
    target_time = time(hour=0, minute=1)  # KST 기준 00:01

    if (
        now_kst.time() >= target_time and
        (last_retrain_date is None or last_retrain_date < now_kst.date())
    ):
        await send_telegram_message("🔁 매일 정기 재학습 시작 (KST 기준)")
        if retrain_model_by_script("train_trend_model_xgb.py"):
            await send_telegram_message("✅ 정기 모델 재학습 완료")
        else:
            await send_telegram_message("❌ 모델 재학습 실패")
        last_retrain_date = now_kst.date()

def retrain_model_by_script(script_path="train_trend_model_xgb.py"):
    try:
        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            check=True
        )
        logging.info(f"✅ 모델 재학습 성공")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ 모델 재학습 실패\n{e.stderr}")
        return False

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

def interval_to_minutes(interval_str):
    if interval_str.endswith('m'):
        return int(interval_str[:-1])
    elif interval_str.endswith('h'):
        return int(interval_str[:-1]) * 60
    elif interval_str.endswith('d'):
        return int(interval_str[:-1]) * 1440
    else:
        raise ValueError("Invalid interval format")
    
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

def compute_rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def predict_trend_with_proba(df: pd.DataFrame, model_path=f"trend_model_xgb_{TRADING_INTERVAL}.pkl"):
    from xgboost import XGBClassifier

    df = df.copy()
    df['return'] = df['close'].pct_change()
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ma_ratio'] = df['ma5'] / df['ma10']
    df['volatility'] = df['return'].rolling(window=5).std()
    df['rsi'] = compute_rsi(df['close'], 14)

    # ✅ 추가된 부분: MACD & Signal
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()

    # ✅ 추가된 부분: Bollinger Band Width
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
        logging.warning(f"⚠️ 모델 로딩 실패 또는 유효하지 않음: {e} → 외부 학습 스크립트 실행")
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

def should_enter_position(current_price, support, resistance, threshold=0.3):
    diff_support = abs(current_price - support) / current_price * 100
    diff_resistance = abs(current_price - resistance) / current_price * 100
    if diff_support <= threshold:
        return 'long'
    elif diff_resistance <= threshold:
        return 'short'
    return None
 
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

def cancel_order(order_id: int):
    try:
        client.futures_cancel_order(symbol='BTCUSDT', orderId=order_id)
    except Exception as e:
        logging.warning(f"❌ 주문 취소 실패: {e}")

async def send_telegram_message(message: str):
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)

def get_current_position(symbol='BTCUSDT'):
    positions = client.futures_position_information(symbol=symbol)
    for p in positions:
        pos_amt = float(p['positionAmt'])
        if pos_amt != 0:
            side = 'long' if pos_amt > 0 else 'short'
            entry_price = float(p['entryPrice'])
            return side, entry_price
    return None, None

def check_existing_tp_sl_orders(symbol='BTCUSDT'):
    open_orders = client.futures_get_open_orders(symbol=symbol)
    tp_exists = any(o['type'] == 'LIMIT' and o['reduceOnly'] for o in open_orders)
    sl_exists = any(o['type'] == 'STOP_MARKET' and o['reduceOnly'] for o in open_orders)
    return tp_exists, sl_exists

async def trading_loop(backtest=False):
    global position_state, entry_price, volatility_blocked, cumulative_pnl
    global TP_PERCENT, SL_PERCENT, last_reset_month, tp_order_id, sl_order_id

    symbol = 'BTCUSDT'

    if position_state is None and entry_price is None:
        position_state, entry_price = get_current_position()
        if position_state:
            await send_telegram_message(
                f"🔁 기존 포지션 복구: {position_state.upper()} @ {entry_price}\n"
                f"❗ 예약된 TP/SL 주문을 확인 중..."
            )
            tp_exists, sl_exists = check_existing_tp_sl_orders()
            if not tp_exists or not sl_exists:
                tp_order_id, sl_order_id = place_tp_sl_orders(entry_price, position_state, quantity)
                await send_telegram_message("🛠️ 누락된 TP/SL 주문을 재설정했습니다.")

    current_month = datetime.now().month
    if current_month != last_reset_month:
        last_reset_month = current_month
        await send_telegram_message("🔄 새 달이 시작되어 누적 수익률을 초기화합니다.")
        cumulative_pnl = 0.0

    if cumulative_pnl <= STOP_LOSS_LIMIT:
        await send_telegram_message(f"🛑 누적 손실 {cumulative_pnl:.2f}%로 자동 중단됩니다.")
        raise SystemExit

    TP_PERCENT, SL_PERCENT = (1.5, 0.3) if cumulative_pnl > 10 else (0.7, 0.3) if cumulative_pnl < -5 else (1.0, 0.5)

    df = get_klines(symbol=symbol)
    support, resistance = calculate_support_resistance(df)
    current_price = float(client.futures_mark_price(symbol=symbol)['markPrice'])
    volatility = analyze_volatility(df)

    if volatility >= VOLATILITY_THRESHOLD and position_state is None:
        if not volatility_blocked:
            await send_telegram_message(f"⚠️ 변동성 과도 ({volatility:.2f}%) → 포지션 진입 회피 중")
            volatility_blocked = True
        return
    elif volatility < VOLATILITY_THRESHOLD and volatility_blocked:
        await send_telegram_message(f"✅ 변동성 정상화 ({volatility:.2f}%) → 진입 가능 상태로 전환")
        volatility_blocked = False

    if position_state and entry_price:
        change_pct = (current_price - entry_price) / entry_price * 100
        if position_state == 'short':
            change_pct *= -1

        if change_pct >= TP_PERCENT or change_pct <= -SL_PERCENT:
            cumulative_pnl += change_pct
            order = close_position(position_state, quantity)
            if tp_order_id:
                cancel_order(tp_order_id)
                tp_order_id = None
            if sl_order_id:
                cancel_order(sl_order_id)
                sl_order_id = None
            label = "🎯 TP 도달" if change_pct >= TP_PERCENT else "⚠️ SL 도달"
            await send_telegram_message(
                f"{label}. {position_state.upper()} 종료\n"
                f"PnL: {change_pct:.2f}%\n"
                f"누적 PnL: {cumulative_pnl:.2f}%\n"
                f"📉 포지션 종료 완료"
            )
            position_state = None
            entry_price = None
            return

    # 진입 조건 판단
    signal = should_enter_position(current_price, support, resistance)

    # 머신러닝 추세 예측 + confidence
    model_path = f"trend_model_xgb_{TRADING_INTERVAL}.pkl"
    trend, confidence = predict_trend_with_proba(df, model_path=model_path)
    decoded_trend = {0: '하락 📉', 1: '횡보 😐', 2: '상승 📈'}[trend]

    # TP/SL 동적 조정 (confidence 기반)
    if confidence >= 0.8:
        TP_PERCENT, SL_PERCENT = 1.8, 0.3
    elif confidence >= 0.6:
        TP_PERCENT, SL_PERCENT = 1.0, 0.5
    else:
        TP_PERCENT, SL_PERCENT = 0.7, 0.5

    if signal:
        await send_telegram_message(
            f"🧠 머신러닝 추세 예측: {decoded_trend}\n"
            f"📊 신뢰도: {confidence * 100:.2f}%"
            f"🎯 TP: {TP_PERCENT}%, ⚠️ SL: {SL_PERCENT}%\n"
            f"🔍 진입 시도: {signal.upper()}"
        )

        # confidence threshold 적용 예시 (60% 이상만 진입 허용)
        if confidence < 0.6:
            await send_telegram_message("❌ 신뢰도 낮음 → 진입 회피")
            return

        if trend == 1:
            await send_telegram_message("😐 머신러닝 예측: 횡보 → 진입 회피")
            return

        if trend == 2 and signal == 'short':
            await send_telegram_message("📈 추세는 상승인데 숏 진입 시도 → 회피")
            return
        elif trend == 0 and signal == 'long':
            await send_telegram_message("📉 추세는 하락인데 롱 진입 시도 → 회피")
            return
        
        # scale-in: 동일 방향 + 고신뢰
        actual_quantity = quantity
        if position_state == signal and confidence >= 0.85:
            actual_quantity *= 2
            await send_telegram_message("💹 고신뢰도 재진입 (Scale-in) → 수량 2배")

        if position_state is not None:
            logging.info("중복 진입 방지: 이미 포지션이 존재함")
            return
        
        if not backtest:
            await send_telegram_message(f"🧠 BTC 지지/저항 분석\n지지선: {support}, 저항선: {resistance}")

        order = place_order(signal, actual_quantity)
        entry_price = current_price
        position_state = signal
        tp_order_id, sl_order_id = place_tp_sl_orders(entry_price, signal, actual_quantity)
        await send_telegram_message(
            f"🔥 {signal.upper()} 진입: {entry_price} USDT\n"
            f"🎯 TP 예약: {round(entry_price * (1 + TP_PERCENT / 100 if signal == 'long' else 1 - TP_PERCENT / 100), 2)}\n"
            f"⚠️ SL 예약: {round(entry_price * (1 - SL_PERCENT / 100 if signal == 'long' else 1 + SL_PERCENT / 100), 2)}"
        )
        return

    elif not signal and position_state:
        change_pct = (current_price - entry_price) / entry_price * 100
        if position_state == 'short':
            change_pct *= -1
        cumulative_pnl += change_pct
        order = close_position(position_state, quantity)
        if tp_order_id:
            cancel_order(tp_order_id)
        if sl_order_id:
            cancel_order(sl_order_id)
        await send_telegram_message(
            f"❌ 신호 없음. {position_state.upper()} 종료\n"
            f"PnL: {change_pct:.2f}%\n"
            f"누적 PnL: {cumulative_pnl:.2f}%\n"
            f"📉 포지션 종료 완료"
        )
        position_state = None
        entry_price = None
        tp_order_id = None
        sl_order_id = None
    else:
        logging.info(f"신호 없음. 현재 변동성: {volatility:.2f}%")

async def start_bot():
    print("⏳ 프로그램 시작됨. 다음 봉 마감까지 대기 중...")

    while True:
        sleep_sec = get_next_bar_close_time(TRADING_INTERVAL)
        print(f"⏱️ 다음 봉 마감까지 {sleep_sec:.2f}초 대기...")
        await maybe_retrain_daily()  # 매 루프마다 재학습 조건 확인
        await asyncio.sleep(sleep_sec)

        try:
            await trading_loop()
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

    # ✅ 추가된 부분: MACD & Signal
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()

    # ✅ 추가된 부분: Bollinger Band Width
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

async def run_all_backtests():
    intervals = ['5m', '15m', '1h']
    summary_results = {}

    for interval in intervals:
        print(f"\n🧪 [{interval}] 백테스트 시작\n")
        pnl = await backtest_bot(interval=interval)
        summary_results[interval] = pnl

    # 결과 요약 출력
    print("\n📊 전체 백테스트 요약\n")
    for interval, pnl in summary_results.items():
        sign = "+" if pnl >= 0 else ""
        print(f"⏱ {interval:>3}  →  누적 PnL: {sign}{pnl:.2f}%")

summary_results = {}

async def backtest_bot(interval='5m') -> float:
    global position_state, entry_price, volatility_blocked, cumulative_pnl
    global TP_PERCENT, SL_PERCENT, last_reset_month, tp_order_id, sl_order_id

    limit = get_auto_limit(interval)
    df = get_klines(symbol='BTCUSDT', interval=interval, limit=limit)  # 과거 데이터 사용

    print(f"\n📊 백테스트 시작: {interval} / 캔들 수: {limit}개\n")

    for i in range(100, len(df)):  # 최소 100개는 있어야 분석 가능
        sliced_df = df.iloc[:i].copy()

        # 지지/저항 및 현재 정보
        support, resistance = calculate_support_resistance(sliced_df)
        current_price = sliced_df['close'].iloc[-1]
        volatility = analyze_volatility(sliced_df)

        current_month = pd.to_datetime(sliced_df['timestamp'].iloc[-1], unit='ms').month
        if current_month != last_reset_month:
            last_reset_month = current_month
            print(f"\n🔄 새 달이 시작됨 → 누적 수익 초기화")
            cumulative_pnl = 0.0

        if cumulative_pnl <= STOP_LOSS_LIMIT:
            print(f"\n🛑 누적 손실 {cumulative_pnl:.2f}%로 자동 중단")
            break
        
        model_path = f"trend_model_xgb_{interval}.pkl"
        trend, confidence = predict_trend_sync(sliced_df, model_path=model_path)
        
        # TP_PERCENT, SL_PERCENT = (1.5, 0.3) if cumulative_pnl > 10 else (0.7, 0.3) if cumulative_pnl < -5 else (1.0, 0.5)

        # ✅ TP/SL 동적 조정 (confidence 기반)
        if confidence >= 0.8:
            TP_PERCENT, SL_PERCENT = 1.8, 0.3
        elif confidence >= 0.6:
            TP_PERCENT, SL_PERCENT = 1.0, 0.5
        else:
            TP_PERCENT, SL_PERCENT = 0.7, 0.5

        # 포지션 종료 조건 체크
        if position_state and entry_price:
            change_pct = (current_price - entry_price) / entry_price * 100
            if position_state == 'short':
                change_pct *= -1

            if change_pct >= TP_PERCENT or change_pct <= -SL_PERCENT:
                cumulative_pnl += change_pct
                label = "🎯 TP" if change_pct >= TP_PERCENT else "⚠️ SL"
                print(f"{label} 도달 → {position_state.upper()} 종료 | PnL: {change_pct:.2f}%, 누적: {cumulative_pnl:.2f}%")
                position_state = None
                entry_price = None
                continue

        # 포지션 진입 여부
        if not volatility_blocked and position_state is None:
            signal = should_enter_position(current_price, support, resistance)
            # trend = predict_trend_sync(sliced_df, model_path='trend_model_xgb.pkl')
            decoded = {0: '하락 📉', 1: '횡보 😐', 2: '상승 📈'}
 
            if signal:
                # print(f"\n🧠 추세 예측: {'상승 📈' if trend == 2 else '하락 📉' if trend == 0 else '횡보 😐'} | 신호: {signal.upper()}")
                print(f"🧠 추세 예측: {decoded[trend]} | 확률: {confidence*100:.2f}% | 신호: {signal.upper()}")

                # 신뢰도 필터 (예: 60% 미만이면 진입 회피)
                if confidence < 0.6:
                    print("⚠️ 신뢰도 낮음 → 진입 회피")
                    continue

                if trend == 1:
                    print("😐 머신러닝 예측: 횡보 → 진입 회피")
                    continue

                if trend == 2 and signal == 'short':
                    print("📈 상승 추세인데 숏 시도 → 진입 회피")
                    continue
                elif trend == 0 and signal == 'long':
                    print("📉 하락 추세인데 롱 시도 → 진입 회피")
                    continue

                # ✅ TP/SL 동적 조정 (confidence 기반)
                # if confidence >= 0.8:
                #     TP_PERCENT, SL_PERCENT = 1.8, 0.3
                # elif confidence >= 0.6:
                #     TP_PERCENT, SL_PERCENT = 1.0, 0.5
                # else:
                #     TP_PERCENT, SL_PERCENT = 0.7, 0.5

                # ✅ scale-in: 동일 방향 + 고신뢰
                actual_quantity = quantity
                if position_state == signal and confidence >= 0.85:
                    actual_quantity *= 2
                    print("💹 고신뢰도 재진입 (Scale-in) → 수량 2배")

                position_state = signal
                entry_price = current_price
                print(f"\n🧠 지지: {support}, 저항: {resistance}")
                print(f"🔥 {signal.upper()} 진입 @ {entry_price:.2f} | 수량: {actual_quantity:.3f} | TP: {TP_PERCENT}%, SL: {SL_PERCENT}%")
                continue

        # 포지션 종료 (신호 없을 경우)
        if not should_enter_position(current_price, support, resistance) and position_state:
            change_pct = (current_price - entry_price) / entry_price * 100
            if position_state == 'short':
                change_pct *= -1
            cumulative_pnl += change_pct
            print(f"❌ 신호 없음 → {position_state.upper()} 종료 | PnL: {change_pct:.2f}%, 누적: {cumulative_pnl:.2f}%")
            position_state = None
            entry_price = None

    # print(f"\n✅ [{interval}] 백테스트 종료 → 누적 PnL: {cumulative_pnl:.2f}%\n")
    return cumulative_pnl  # 누적 수익률 반환

if __name__ == "__main__":
    mode = input("실행 모드 선택 (live / backtest / all_backtest): ").strip()
    if mode == "live":
        asyncio.run(start_bot())
    elif mode == "backtest":
        asyncio.run(backtest_bot(interval=TRADING_INTERVAL))
    elif mode == "all_backtest":
        asyncio.run(run_all_backtests())