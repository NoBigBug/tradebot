import asyncio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import logging
import joblib

from sklearn.cluster import KMeans
from telegram import Bot
from binance.client import Client
from datetime import datetime, timedelta, timezone

from config import BINANCE_API_KEY, BINANCE_API_SECRET, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

mpl.rcParams['font.family'] = 'AppleGothic'
mpl.rcParams['axes.unicode_minus'] = False

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
bot = Bot(token=TELEGRAM_BOT_TOKEN)

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

# 트레이딩 인터벌 설정 ('1m', '3m', '5m', '15m', '30m', '1h' 등)
TRADING_INTERVAL = '15m'  # Binance 기준 문자열

logging.basicConfig(level=logging.INFO)

def get_next_bar_close_time(interval_str='15m', buffer_seconds=5):
    now = datetime.now(timezone.utc)
    interval_minutes = interval_to_minutes(interval_str)

    # 다음 봉 마감 시각 계산
    next_minute = (now.minute // interval_minutes + 1) * interval_minutes
    next_bar_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=next_minute)

    # 시간이 60분 넘어가면 hour/day 보정
    if next_minute >= 60:
        next_bar_time += timedelta(hours=1)

    return (next_bar_time - now).total_seconds() + buffer_seconds

def interval_to_minutes(interval: str) -> int:
    unit = interval[-1]
    value = int(interval[:-1])
    if unit == 'm':
        return value
    elif unit == 'h':
        return value * 60
    elif unit == 'd':
        return value * 60 * 24
    else:
        raise ValueError(f"지원하지 않는 인터벌 형식: {interval}")
    
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

def predict_trend(df: pd.DataFrame, model_path='trend_model.pkl') -> int:
    model = joblib.load(model_path)
    df = df.copy()
    df['return'] = df['close'].pct_change()
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ma_ratio'] = df['ma5'] / df['ma10']
    df['volatility'] = df['return'].rolling(window=5).std()
    df['rsi'] = compute_rsi(df['close'], 14)
    df = df.dropna()
    latest = df[['ma_ratio', 'volatility', 'rsi']].iloc[-1:]
    return model.predict(latest)[0]  # 1=상승, -1=하락, 0=횡보

def predict_trend_text(trend: int) -> str:
    if trend == 1:
        return "상승 📈"
    elif trend == -1:
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
    return support, resistance, prices

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

    symbol = 'BTCUSDT'
    df = get_klines(symbol=symbol)
    support, resistance, clusters = calculate_support_resistance(df)
    
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

    signal = should_enter_position(current_price, support, resistance)
    trend = predict_trend(df)
    trend_text = predict_trend_text(trend)

    if signal:
        await send_telegram_message(
            f"🧠 머신러닝 추세 예측: {trend_text}\n"
            f"🔍 진입 시도: {signal.upper()}"
        )

        if trend == 0:
            await send_telegram_message("📉 추세가 '횡보' 상태입니다. 진입 회피합니다.")
            return
        elif trend == 1 and signal == 'short':
            await send_telegram_message("📈 추세는 상승인데 숏 진입 시도 → 회피")
            return
        elif trend == -1 and signal == 'long':
            await send_telegram_message("📉 추세는 하락인데 롱 진입 시도 → 회피")
            return

        if position_state is not None:
            logging.info("중복 진입 방지: 이미 포지션이 존재함")
            return
        
        if not backtest:
            await send_telegram_message(f"🧠 BTC 지지/저항 분석\n지지선: {support}, 저항선: {resistance}")

        order = place_order(signal, quantity)
        entry_price = current_price
        position_state = signal
        tp_order_id, sl_order_id = place_tp_sl_orders(entry_price, signal, quantity)
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
        await asyncio.sleep(sleep_sec)

        try:
            await trading_loop()
        except SystemExit:
            break
        except Exception as e:
            await send_telegram_message(f"❌ 오류 발생: {e}")

def predict_trend_sync(df: pd.DataFrame, model_path='trend_model.pkl') -> int:
    df = df.copy()
    df['return'] = df['close'].pct_change()
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ma_ratio'] = df['ma5'] / df['ma10']
    df['volatility'] = df['return'].rolling(window=5).std()
    df['rsi'] = compute_rsi(df['close'], 14)
    df = df.dropna()

    if len(df) < 1:
        return 0  # 예측 불가 → 횡보로 처리

    features = df[['ma_ratio', 'volatility', 'rsi']]
    model = joblib.load(model_path)
    return model.predict(features.iloc[-1:])[0]

async def backtest_bot():
    global position_state, entry_price, volatility_blocked, cumulative_pnl
    global TP_PERCENT, SL_PERCENT, last_reset_month, tp_order_id, sl_order_id

    limit = get_auto_limit(TRADING_INTERVAL)
    df = get_klines(symbol='BTCUSDT', interval=TRADING_INTERVAL, limit=limit)  # 과거 데이터 사용

    print(f"\n📊 백테스트 시작: {TRADING_INTERVAL} / 캔들 수: {limit}개\n")

    for i in range(100, len(df)):  # 최소 100개는 있어야 분석 가능
        sliced_df = df.iloc[:i].copy()

        # 지지/저항 및 현재 정보
        support, resistance, clusters = calculate_support_resistance(sliced_df)
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

        TP_PERCENT, SL_PERCENT = (1.5, 0.3) if cumulative_pnl > 10 else (0.7, 0.3) if cumulative_pnl < -5 else (1.0, 0.5)

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
            trend = predict_trend_sync(sliced_df)
            
            if signal:
                print(f"\n🧠 추세 예측: {'상승 📈' if trend == 1 else '하락 📉' if trend == -1 else '횡보 😐'} | 신호: {signal.upper()}")

                if trend == 0:
                    print("😐 횡보 추세 → 진입 회피")
                    continue
                elif trend == 1 and signal == 'short':
                    print("📈 상승 추세인데 숏 시도 → 진입 회피")
                    continue
                elif trend == -1 and signal == 'long':
                    print("📉 하락 추세인데 롱 시도 → 진입 회피")
                    continue

                position_state = signal
                entry_price = current_price
                print(f"\n🧠 지지: {support}, 저항: {resistance}")
                print(f"🔥 {signal.upper()} 진입 @ {entry_price} | Volatility: {volatility:.2f}%")
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

    print(f"\n📊 백테스트 종료 → 최종 누적 수익률: {cumulative_pnl:.2f}%")

if __name__ == "__main__":
    mode = input("실행 모드 선택 (live / backtest): ").strip()
    if mode == "live":
        asyncio.run(start_bot())
    else:
        asyncio.run(backtest_bot())