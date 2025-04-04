import asyncio                      # 비동기 처리를 위한 라이브러리
import numpy as np                  # numpy: 수치 계산을 위한 라이브러리
import pandas as pd                 # pandas: 데이터 분석용 라이브러리
import matplotlib as mpl            # matplotlib: 데이터 시각화를 위한 라이브러리
import logging                      # logging: 로그 메시지를 기록하기 위한 라이브러리
import joblib                       # joblib: 머신러닝 모델 저장 및 불러오기를 위한 라이브러리

# scikit-learn의 KMeans 클러스터링 (유사한 데이터를 군집화하는 알고리즘)
from sklearn.cluster import KMeans
# Telegram 봇 API (메시지 전송 등 자동화에 사용)
from telegram import Bot
# Binance API 클라이언트 (암호화폐 거래소와 연동)
from binance.client import Client
# 날짜와 시간 관련 모듈
from datetime import datetime, timedelta, timezone

# config 파일에서 API 키와 토큰, 챗 ID를 불러옴
from config import BINANCE_API_KEY, BINANCE_API_SECRET, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

# ──────────────────────────────────────────────
# [폰트 및 matplotlib 설정]
# 한글 폰트 설정 (Mac 기준)과 마이너스(-) 부호가 깨지지 않도록 설정합니다.
mpl.rcParams['font.family'] = 'AppleGothic'
mpl.rcParams['axes.unicode_minus'] = False

# ──────────────────────────────────────────────
# [Binance 및 Telegram 초기화]
# Binance API를 사용하기 위해 Client 객체를 생성합니다.
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
# Telegram 봇 객체를 생성하여 메시지 전송 기능을 사용합니다.
bot = Bot(token=TELEGRAM_BOT_TOKEN)

# ──────────────────────────────────────────────
# [전역 변수 설정]
# 포지션 상태 관련 변수:
position_state = None  # 현재 포지션 상태: 'long'(매수), 'short'(매도) 또는 None(포지션 없음)
entry_price = None     # 포지션에 진입한 가격
tp_order_id = None     # 익절(Take Profit) 주문 ID
sl_order_id = None     # 손절(Stop Loss) 주문 ID
quantity = 0.05        # 기본 진입 수량 (예: 0.05 BTC)

# TP/SL(익절/손절) 및 리스크 관리 파라미터:
TP_PERCENT = 1.0             # 기본 익절 목표 수익률 (1%)
SL_PERCENT = 0.5             # 기본 손절 한계 (0.5%)
VOLATILITY_THRESHOLD = 2.5   # 과도한 변동성 기준치 (%)
volatility_blocked = False   # 변동성에 의한 진입 차단 여부
cumulative_pnl = 0.0         # 누적 수익률 (PnL: Profit and Loss)
STOP_LOSS_LIMIT = -10.0      # 누적 손실 한도 (%)
last_reset_month = datetime.now().month  # 매월 누적 수익률을 초기화하는 기준 월

# Binance의 캔들(봉) 인터벌 설정 (예: '15m'는 15분 봉, '1h'는 1시간 봉)
TRADING_INTERVAL = '15m'

# ──────────────────────────────────────────────
# [로깅 설정]
# 로그 레벨을 INFO로 설정하여 정보 메시지를 출력합니다.
logging.basicConfig(level=logging.INFO)

# ──────────────────────────────────────────────
# [함수 정의: 시간 인터벌 및 데이터 제한 관련]
def interval_to_minutes(interval: str) -> int:
    """
    문자열로 표현된 시간 간격을 분 단위의 정수로 변환하는 함수.
    예: '15m' → 15, '1h' → 60, '1d' → 1440
    """
    unit = interval[-1]             # 마지막 글자: 시간 단위 (m, h, d)
    value = int(interval[:-1])      # 숫자 부분 추출
    if unit == 'm':
        return value
    elif unit == 'h':
        return value * 60
    elif unit == 'd':
        return value * 60 * 24
    else:
        raise ValueError(f"지원하지 않는 인터벌 형식: {interval}")
  
def get_auto_limit(interval: str) -> int:
    """
    각 캔들 인터벌에 따라 Binance API 요청 시 가져올 데이터 개수(limit)를 반환.
    인터벌에 따라 적절한 데이터 수를 기본값으로 설정합니다.
    """
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

# ──────────────────────────────────────────────
# [함수 정의: Binance 데이터 관련]
def get_klines(symbol='BTCUSDT', interval=None, limit=None):
    """
    Binance에서 캔들(봉) 데이터를 가져와 pandas DataFrame으로 변환하는 함수.
    - symbol: 거래쌍 (예: BTCUSDT)
    - interval: 캔들 간격 (예: '15m')
    - limit: 가져올 데이터 개수
    반환되는 DataFrame은 캔들 데이터의 주요 항목들을 포함합니다.
    """
    klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'num_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    # 문자열 형태의 가격과 거래량을 float형으로 변환
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    return df

# ──────────────────────────────────────────────
# [함수 정의: 기술적 지표 계산]
def compute_rsi(series: pd.Series, period: int = 14):
    """
    RSI(상대강도지수)를 계산하는 함수.
    RSI는 가격의 상승과 하락 강도를 나타내는 모멘텀 지표입니다.
    period: RSI를 계산할 기간 (기본 14)
    """
    delta = series.diff()  # 가격 차이 계산
    gain = delta.where(delta > 0, 0.0)  # 상승분
    loss = -delta.where(delta < 0, 0.0)   # 하락분 (음수를 양수로 변환)
    avg_gain = gain.rolling(window=period).mean()  # 기간 평균 상승분
    avg_loss = loss.rolling(window=period).mean()   # 기간 평균 하락분
    rs = avg_gain / avg_loss  # 상대 강도
    rsi = 100 - (100 / (1 + rs))  # RSI 공식 적용
    return rsi

# ──────────────────────────────────────────────
# [함수 정의: 머신러닝을 이용한 추세 예측]
def predict_trend(df: pd.DataFrame, model_path='trend_model_xgb.pkl') -> tuple[int, float]:
    """
    머신러닝 모델(XGBoost 등)을 사용하여 현재 시장 추세를 예측하는 함수.
    예측 결과로 추세 (0: 하락, 1: 횡보, 2: 상승)와 해당 예측의 신뢰도를 반환합니다.
    """
    model = joblib.load(model_path)  # 저장된 머신러닝 모델 불러오기
    df = df.copy()
    # 수익률(변화율) 계산
    df['return'] = df['close'].pct_change()
    # 이동평균 계산 (단기: 5, 중기: 10)
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    # 단기와 중기 이동평균 비율 (추세 판단에 도움)
    df['ma_ratio'] = df['ma5'] / df['ma10']
    # 변동성: 최근 5개 수익률의 표준편차
    df['volatility'] = df['return'].rolling(window=5).std()
    # RSI 계산 (모멘텀 지표)
    df['rsi'] = compute_rsi(df['close'], 14)
    # 결측값 제거
    df = df.dropna()

    # 데이터가 부족하면 기본값 반환 (횡보로 간주)
    if len(df) < 1:
        return 1, 0.0  # 1은 횡보, 신뢰도 0.0

    # 마지막 데이터를 이용해 예측
    latest = df[['ma_ratio', 'volatility', 'rsi']].iloc[-1:]
    proba = model.predict_proba(latest)[0]
    pred = int(np.argmax(proba))      # 예측 결과 (가장 확률이 높은 클래스)
    confidence = float(proba[pred])     # 해당 예측의 신뢰도

    return pred, confidence

# ──────────────────────────────────────────────
# [함수 정의: 지지선과 저항선 계산]
def calculate_support_resistance(df, n_clusters=6):
    """
    KMeans 클러스터링을 활용하여 지지선(support)과 저항선(resistance)을 계산합니다.
    - 캔들 종가를 정수형으로 반올림한 후, 거래량 합계를 기준으로 군집화합니다.
    - 군집의 중앙값을 이용해 지지와 저항 가격을 도출합니다.
    """
    # 종가를 정수로 반올림하여 그룹화 (가격대별 거래량 집계)
    df['rounded_price'] = df['close'].astype(int)
    grouped = df.groupby('rounded_price')['volume'].sum().reset_index()
    # KMeans 클러스터링: 가격과 거래량 데이터를 군집화
    X = grouped[['rounded_price', 'volume']].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(X)
    centers = kmeans.cluster_centers_
    # 군집 중심의 가격 값을 정렬하여 최저가와 최고가를 지지선, 저항선으로 간주
    prices = np.sort(centers[:, 0].astype(int))
    support = prices[0]
    resistance = prices[-1]
    return support, resistance

# ──────────────────────────────────────────────
# [함수 정의: 변동성 분석]
def analyze_volatility(df):
    """
    캔들 데이터의 종가 변동성을 백분율(%)로 계산하는 함수.
    변동성은 수익률의 표준편차로 산출됩니다.
    """
    returns = df['close'].pct_change().dropna()
    volatility = returns.std() * 100
    return volatility

# ──────────────────────────────────────────────
# [함수 정의: 진입 신호 판단]
def should_enter_position(current_price, support, resistance, threshold=0.3):
    """
    현재 가격이 지지선 또는 저항선에 근접했는지를 확인하여 진입 방향을 결정합니다.
    - 가격이 지지선과 가까우면 'long'(매수) 신호
    - 가격이 저항선과 가까우면 'short'(매도) 신호
    - threshold: 가격과의 허용 오차 (%)를 지정
    """
    diff_support = abs(current_price - support) / current_price * 100
    diff_resistance = abs(current_price - resistance) / current_price * 100
    if diff_support <= threshold:
        return 'long'
    elif diff_resistance <= threshold:
        return 'short'
    return None

# ──────────────────────────────────────────────
# [함수 정의: 주문 실행 관련]
def place_order(side: str, quantity: float):
    """
    시장가 주문을 실행하는 함수.
    - side: 'long'이면 매수, 'short'이면 매도
    - quantity: 주문 수량
    """
    order = client.futures_create_order(
        symbol='BTCUSDT',
        side='BUY' if side == 'long' else 'SELL',
        type='MARKET',
        quantity=quantity
    )
    return order

def close_position(current_side: str, quantity: float):
    """
    현재 포지션을 종료하는 주문을 실행하는 함수.
    - current_side가 'long'이면 매도하여 포지션 종료, 'short'이면 매수하여 포지션 종료
    """
    close_side = 'SELL' if current_side == 'long' else 'BUY'
    order = client.futures_create_order(
        symbol='BTCUSDT',
        side=close_side,
        type='MARKET',
        quantity=quantity
    )
    return order

# ──────────────────────────────────────────────
# [함수 정의: 최소 호가 단위 관련]
def get_tick_size(symbol='BTCUSDT'):
    """
    Binance 선물 거래소에서 해당 거래쌍의 최소 가격 단위(tick size)를 가져옵니다.
    tick size는 가격이 움직일 수 있는 최소 단위를 의미합니다.
    """
    info = client.futures_exchange_info()
    for s in info['symbols']:
        if s['symbol'] == symbol:
            for f in s['filters']:
                if f['filterType'] == 'PRICE_FILTER':
                    return float(f['tickSize'])
    raise ValueError("Tick size not found.")

def round_to_tick(price, tick_size):
    """
    지정된 tick size에 맞춰 가격을 반올림하는 함수.
    유효한 가격 형식을 맞추기 위해 사용합니다.
    """
    return round(round(price / tick_size) * tick_size, 8)

# ──────────────────────────────────────────────
# [함수 정의: 익절(TP) 및 손절(SL) 주문 실행]
def place_tp_sl_orders(entry_price: float, side: str, quantity: float):
    """
    포지션 진입 후 익절(TP)과 손절(SL) 주문을 동시에 설정합니다.
    - entry_price: 진입 가격
    - side: 'long' 또는 'short'
    - quantity: 주문 수량
    TP/SL 가격은 설정된 퍼센트에 따라 계산됩니다.
    """
    tick_size = get_tick_size('BTCUSDT')

    # 익절 가격: 매수 시 가격 상승, 매도 시 가격 하락을 예상하여 설정
    tp_price = entry_price * (1 + TP_PERCENT / 100) if side == 'long' else entry_price * (1 - TP_PERCENT / 100)
    # 손절 가격: 매수 시 가격 하락, 매도 시 가격 상승을 대비하여 설정
    sl_price = entry_price * (1 - SL_PERCENT / 100) if side == 'long' else entry_price * (1 + SL_PERCENT / 100)

    # tick size에 맞춰 가격 반올림 및 문자열 변환
    tp_price = str(round_to_tick(tp_price, tick_size))
    sl_price = str(round_to_tick(sl_price, tick_size))

    # 익절 주문: 지정가 주문(LIMIT)
    tp_order = client.futures_create_order(
        symbol='BTCUSDT',
        side='SELL' if side == 'long' else 'BUY',
        type='LIMIT',
        price=tp_price,
        quantity=quantity,
        timeInForce='GTC',  # Good-Til-Canceled: 취소될 때까지 유지
        reduceOnly=True    # 포지션 축소만 가능하도록 설정
    )

    # 손절 주문: 스탑 마켓 주문(STOP_MARKET)
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
    """
    지정된 주문(order_id)을 취소하는 함수.
    예외 발생 시 로그에 경고 메시지를 출력합니다.
    """
    try:
        client.futures_cancel_order(symbol='BTCUSDT', orderId=order_id)
    except Exception as e:
        logging.warning(f"❌ 주문 취소 실패: {e}")

# ──────────────────────────────────────────────
# [함수 정의: Telegram 메시지 전송]
async def send_telegram_message(message: str):
    """
    Telegram 봇을 통해 메시지를 전송하는 비동기 함수.
    """
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)

# ──────────────────────────────────────────────
# [함수 정의: 현재 포지션 조회]
def get_current_position(symbol='BTCUSDT'):
    """
    Binance 선물 거래소에서 현재 보유 중인 포지션을 조회합니다.
    포지션이 있을 경우, 포지션 방향('long' 또는 'short')과 진입 가격을 반환합니다.
    """
    positions = client.futures_position_information(symbol=symbol)
    for p in positions:
        pos_amt = float(p['positionAmt'])
        if pos_amt != 0:
            side = 'long' if pos_amt > 0 else 'short'
            entry_price = float(p['entryPrice'])
            return side, entry_price
    return None, None

# ──────────────────────────────────────────────
# [함수 정의: 기존 TP/SL 주문 존재 여부 확인]
def check_existing_tp_sl_orders(symbol='BTCUSDT'):
    """
    열려 있는 주문 중 익절(TP)과 손절(SL) 주문이 있는지 확인합니다.
    """
    open_orders = client.futures_get_open_orders(symbol=symbol)
    tp_exists = any(o['type'] == 'LIMIT' and o['reduceOnly'] for o in open_orders)
    sl_exists = any(o['type'] == 'STOP_MARKET' and o['reduceOnly'] for o in open_orders)
    return tp_exists, sl_exists

# ──────────────────────────────────────────────
# [함수 정의: 메인 트레이딩 루프]
async def trading_loop(backtest=False):
    """
    실시간 트레이딩을 위한 메인 루프 함수.
    - 캔들 데이터 수집, 기술적 지표 및 머신러닝 추세 예측, 진입 및 청산 조건 판단,
      주문 실행, 손익 관리 등을 수행합니다.
    - Telegram 메시지로 진행 상황을 알립니다.
    """
    global position_state, entry_price, volatility_blocked, cumulative_pnl
    global TP_PERCENT, SL_PERCENT, last_reset_month, tp_order_id, sl_order_id

    symbol = 'BTCUSDT'

    # 만약 프로그램 시작 시 포지션이 이미 있다면 복구 작업 수행
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

    # 매월 시작 시 누적 수익률(PnL) 초기화
    current_month = datetime.now().month
    if current_month != last_reset_month:
        last_reset_month = current_month
        await send_telegram_message("🔄 새 달이 시작되어 누적 수익률을 초기화합니다.")
        cumulative_pnl = 0.0

    # 누적 손실이 한도를 초과하면 거래 중단
    if cumulative_pnl <= STOP_LOSS_LIMIT:
        await send_telegram_message(f"🛑 누적 손실 {cumulative_pnl:.2f}%로 자동 중단됩니다.")
        raise SystemExit
    
    # 캔들 데이터 불러오기 (과거 데이터)
    limit = get_auto_limit(TRADING_INTERVAL)
    df = get_klines(symbol=symbol, interval=TRADING_INTERVAL, limit=limit)
    # 지지선과 저항선 계산
    support, resistance = calculate_support_resistance(df)
    # 현재 시장 가격 조회
    current_price = float(client.futures_mark_price(symbol=symbol)['markPrice'])
    # 현재 시장 변동성 분석
    volatility = analyze_volatility(df)

    # 변동성이 과도하면 진입 차단
    if volatility >= VOLATILITY_THRESHOLD and position_state is None:
        if not volatility_blocked:
            await send_telegram_message(f"⚠️ 변동성 과도 ({volatility:.2f}%) → 포지션 진입 회피 중")
            volatility_blocked = True
        return
    elif volatility < VOLATILITY_THRESHOLD and volatility_blocked:
        await send_telegram_message(f"✅ 변동성 정상화 ({volatility:.2f}%) → 진입 가능 상태로 전환")
        volatility_blocked = False

    # 이미 포지션이 있다면 수익 실현 또는 손절 조건 체크
    if position_state and entry_price:
        change_pct = (current_price - entry_price) / entry_price * 100
        if position_state == 'short':
            change_pct *= -1  # 숏 포지션은 반대 방향 수익 계산

        if change_pct >= TP_PERCENT or change_pct <= -SL_PERCENT:
            cumulative_pnl += change_pct
            close_position(position_state, quantity)
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

    # 진입 조건 판단 (지지선/저항선 근접 여부)
    signal = should_enter_position(current_price, support, resistance)
    
    # 머신러닝 모델을 통해 시장 추세 예측 및 신뢰도 계산
    trend, confidence = predict_trend(df)
    trend_text = {0: '하락 📉', 1: '횡보 😐', 2: '상승 📈'}[trend]

    # 신뢰도에 따라 TP/SL 퍼센트를 동적으로 조정
    if confidence >= 0.8:
        TP_PERCENT, SL_PERCENT = 1.8, 0.3
    elif confidence >= 0.6:
        TP_PERCENT, SL_PERCENT = 1.0, 0.5
    else:
        TP_PERCENT, SL_PERCENT = 0.7, 0.5

    if signal:
        await send_telegram_message(
            f"🧠 머신러닝 추세 예측: {trend_text}\n"
            f"📊 신뢰도: {confidence * 100:.2f}%\n"
            f"🎯 TP: {TP_PERCENT}%, ⚠️ SL: {SL_PERCENT}%\n"
            f"🔍 진입 시도: {signal.upper()}"
        )

        # 신뢰도가 낮으면 거래 회피
        if confidence < 0.6:
            await send_telegram_message("❌ 신뢰도 낮음 → 진입 회피")
            return

        # 추세 방향과 진입 신호가 상반되면 진입 회피
        if trend == 2 and signal == 'short':
            await send_telegram_message("📈 상승 추세인데 숏 시도 → 진입 회피")
            return
        if trend == 0 and signal == 'long':
            await send_telegram_message("📉 하락 추세인데 롱 시도 → 진입 회피")
            return
    
        # 동일 포지션 재진입 (Scale-in): 고신뢰일 경우 수량 2배
        actual_quantity = quantity
        if position_state == signal and confidence >= 0.85:
            actual_quantity *= 2
            await send_telegram_message("💹 고신뢰도 재진입 (Scale-in) → 수량 2배")

        # 시장가 주문으로 포지션 진입
        place_order(signal, actual_quantity)

        # 포지션 진입 시 가격 및 상태 갱신
        entry_price = current_price
        position_state = signal
        # 익절(TP)과 손절(SL) 주문 예약
        tp_order_id, sl_order_id = place_tp_sl_orders(entry_price, signal, actual_quantity)
        
        await send_telegram_message(
            f"🔥 {signal.upper()} 진입: {entry_price} USDT\n"
            f"🎯 TP 예약: {round(entry_price * (1 + TP_PERCENT / 100 if signal == 'long' else 1 - TP_PERCENT / 100), 2)}\n"
            f"⚠️ SL 예약: {round(entry_price * (1 - SL_PERCENT / 100 if signal == 'long' else 1 + SL_PERCENT / 100), 2)}"
        )
        return

    # 포지션이 있지만 진입 신호가 없을 경우 포지션 종료 처리
    elif not signal and position_state:
        change_pct = (current_price - entry_price) / entry_price * 100
        if position_state == 'short':
            change_pct *= -1
        cumulative_pnl += change_pct
        close_position(position_state, quantity)
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

# ──────────────────────────────────────────────
# [함수 정의: 다음 봉 종료 시간 계산]
def get_next_bar_close_time(interval_str=TRADING_INTERVAL, buffer_seconds=5):
    """
    현재 시각을 기준으로 다음 캔들(봉)의 종료 시각까지 남은 시간을 초 단위로 계산하는 함수.
    buffer_seconds: 추가 대기 시간(초)으로, API 지연 등을 보완
    """
    now = datetime.now(timezone.utc)
    interval_minutes = interval_to_minutes(interval_str)

    # 다음 봉이 끝나는 시각 계산
    next_minute = (now.minute // interval_minutes + 1) * interval_minutes
    next_bar_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=next_minute)

    # 분 계산 후 시간이 60분 이상이면 시간 보정
    if next_minute >= 60:
        next_bar_time += timedelta(hours=1)

    return (next_bar_time - now).total_seconds() + buffer_seconds

# ──────────────────────────────────────────────
# [함수 정의: 라이브 모드 트레이딩 봇 실행]
async def start_bot():
    """
    라이브 트레이딩 모드에서 봇을 실행하는 함수.
    주기적으로 다음 봉 종료 시각까지 대기한 후, trading_loop()를 실행합니다.
    """
    while True:
        sleep_sec = get_next_bar_close_time(TRADING_INTERVAL)
        readable_interval = TRADING_INTERVAL.upper()
        
        print(f"⏳ 프로그램 시작됨. 다음 봉 마감까지 {sleep_sec:.2f}초 대기 중...")

        # ✅ 텔레그램으로 대기 정보 전송
        await send_telegram_message(
            f"🕒 트레이딩 봇 실행됨\n"
            f"⏱️ 다음 봉 마감까지 {int(sleep_sec)}초 대기 중 ({readable_interval})"
        )

        # 다음 봉 종료까지 대기
        await asyncio.sleep(sleep_sec)

        try:
            await trading_loop()
        except SystemExit:
            break
        except Exception as e:
            await send_telegram_message(f"❌ 오류 발생: {e}")

# ──────────────────────────────────────────────
# [함수 정의: 동기 방식 추세 예측 (백테스트용)]
def predict_trend_sync(df: pd.DataFrame, model_path='trend_model_xgb.pkl') -> tuple[int, float]:
    """
    백테스트에서 사용하기 위해 동기 방식으로 머신러닝 추세 예측을 수행합니다.
    """
    df = df.copy()
    df['return'] = df['close'].pct_change()
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ma_ratio'] = df['ma5'] / df['ma10']
    df['volatility'] = df['return'].rolling(window=5).std()
    df['rsi'] = compute_rsi(df['close'], 14)
    df = df.dropna()

    if len(df) < 1:
        return 1, 0.0  # 데이터 부족 시 기본값: 횡보, 신뢰도 0

    features = df[['ma_ratio', 'volatility', 'rsi']]
    model = joblib.load(model_path)

    proba = model.predict_proba(features.iloc[-1:])[0]
    pred = int(np.argmax(proba))
    confidence = float(proba[pred])

    return pred, confidence

# ──────────────────────────────────────────────
# [함수 정의: 백테스트 모드 실행]
async def backtest_bot():
    """
    과거 데이터를 사용하여 전략의 성과를 시뮬레이션하는 백테스트 함수.
    실제 주문을 실행하지 않고, 가상 환경에서 누적 수익률과 거래 결과를 출력합니다.
    """
    global position_state, entry_price, volatility_blocked, cumulative_pnl
    global TP_PERCENT, SL_PERCENT, last_reset_month, tp_order_id, sl_order_id

    limit = get_auto_limit(TRADING_INTERVAL)
    df = get_klines(symbol='BTCUSDT', interval=TRADING_INTERVAL, limit=limit)  # 과거 데이터 사용

    print(f"\n📊 백테스트 시작: {TRADING_INTERVAL} / 캔들 수: {limit}개\n")

    # 최소 100개의 캔들이 있어야 충분한 데이터로 판단 가능
    for i in range(100, len(df)):
        sliced_df = df.iloc[:i].copy()

        # 현재까지의 데이터로 지지/저항 계산 및 현재 가격, 변동성 분석
        support, resistance = calculate_support_resistance(sliced_df)
        current_price = sliced_df['close'].iloc[-1]
        volatility = analyze_volatility(sliced_df)

        # 월이 바뀌면 누적 수익률 초기화
        current_month = pd.to_datetime(sliced_df['timestamp'].iloc[-1], unit='ms').month
        if current_month != last_reset_month:
            last_reset_month = current_month
            print(f"\n🔄 새 달이 시작됨 → 누적 수익 초기화")
            cumulative_pnl = 0.0

        # 누적 손실이 한도를 넘으면 백테스트 종료
        if cumulative_pnl <= STOP_LOSS_LIMIT:
            print(f"\n🛑 누적 손실 {cumulative_pnl:.2f}%로 자동 중단")
            break
        
        # 누적 수익률에 따른 TP/SL 퍼센트 동적 조정
        TP_PERCENT, SL_PERCENT = (1.5, 0.3) if cumulative_pnl > 10 else (0.7, 0.3) if cumulative_pnl < -5 else (1.0, 0.5)

        # 이미 포지션이 있으면 종료 조건 확인
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

        # 포지션 진입 조건 확인 (변동성 차단이 아니고, 아직 포지션이 없을 경우)
        if not volatility_blocked and position_state is None:
            signal = should_enter_position(current_price, support, resistance)

            if signal:
                # 머신러닝 추세 예측 + 확률
                trend, confidence = predict_trend_sync(sliced_df)

                decoded = {0: '하락 📉', 1: '횡보 😐', 2: '상승 📈'}
                print(f"🧠 추세 예측: {decoded[trend]} | 확률: {confidence * 100:.2f}% | 신호: {signal.upper()}")

                # 신뢰도 필터
                if confidence < 0.6:
                    print("⚠️ 신뢰도 낮음 → 진입 회피")
                    continue

                # 추세 반대면 진입 회피
                if trend == 2 and signal == 'short':
                    print("📈 상승 추세인데 숏 시도 → 진입 회피")
                    continue
                if trend == 0 and signal == 'long':
                    print("📉 하락 추세인데 롱 시도 → 진입 회피")
                    continue

                # ✅ 진입
                position_state = signal
                entry_price = current_price
                print(f"\n🧠 지지: {support}, 저항: {resistance}")
                print(f"🔥 {signal.upper()} 진입 @ {entry_price:.2f} | Volatility: {volatility:.2f}%")
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

# ──────────────────────────────────────────────
# [메인 실행 부분]
if __name__ == "__main__":
    # 실행 모드를 선택: 실시간(live) 또는 백테스트(backtest)
    mode = input("실행 모드 선택 (live / backtest): ").strip()
    if mode == "live":
        asyncio.run(start_bot())
    else:
        asyncio.run(backtest_bot())
