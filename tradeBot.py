# 외부 라이브러리 import
import pandas as pd  # 데이터 조작 및 분석을 위한 라이브러리
import asyncio       # 비동기 프로그래밍 지원 (비동기 I/O)
import logging       # 로깅(로그 기록) 기능 제공

# Binance API와 관련된 모듈 import
from binance.client import Client  # 바이낸스 API 클라이언트 (거래소와의 통신)
from binance.enums import *        # Binance API에 사용되는 상수 (예: 주문 타입, 거래 방향 등)

# 기술적 분석 지표 라이브러리 (ta: Technical Analysis)
from ta.trend import EMAIndicator, MACD, ADXIndicator, CCIIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

# 텔레그램 봇을 통한 메시지 전송 기능 (거래 알림 등에 사용)
from telegram import Bot

# API 키와 기타 설정값이 저장된 별도 파일 (보안 및 재사용을 위해 분리)
from config import BINANCE_API_KEY, BINANCE_API_SECRET, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

# 로깅 설정: 로그 레벨을 INFO로 지정하고, 출력 포맷 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =======================
# 글로벌 파라미터 설정
# =======================
SYMBOL = "BTCUSDT"               # 거래 심볼: 비트코인/USDT (Tether)
LEVERAGE = 20                    # 레버리지: 20배 레버리지 (투자금 대비 증폭 효과)
USE_CALCULATED_QUANTITY = False  # 거래 수량 산정 방식 선택: True이면 calculate_max_quantity 함수 사용, False이면 테스트용으로 고정 수량 사용
USE_RISK_MANAGEMENT = False      # 리스크 관리 활성화 여부 (포지션 사이즈 계산에 사용)

# 기술적 지표 파라미터 (튜닝 가능, 시장 상황에 맞게 조절 가능)
EMA_SHORT_WINDOW = 9   # 단기 EMA(지수 이동평균) 기간
EMA_LONG_WINDOW = 21   # 장기 EMA 기간
RSI_WINDOW = 14        # RSI(상대강도지수) 기간
MACD_FAST = 12         # MACD 계산 시 빠른(단기) EMA 기간
MACD_SLOW = 26         # MACD 계산 시 느린(장기) EMA 기간
MACD_SIGNAL = 9        # MACD 시그널 라인 계산 기간
ATR_WINDOW = 14        # ATR(평균 진폭 범위) 기간, 변동성 측정 지표
CCI_WINDOW = 20        # CCI(상품 채널 지표) 계산에 사용되는 기간

# =======================
# 바이낸스 및 텔레그램 API 설정
# =======================
# --- 바이낸스 API 설정 ---
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)  # 바이낸스 클라이언트 생성

# 레버리지 변경 시도 (선물 거래의 경우 레버리지 설정이 필요)
try:
    client.futures_change_leverage(symbol=SYMBOL, leverage=LEVERAGE)
except Exception as e:
    logging.warning(f"[WARNING] 레버리지 변경 실패: {e}")

# --- 텔레그램 API 설정 ---
bot = Bot(token=TELEGRAM_BOT_TOKEN)  # 텔레그램 봇 객체 생성

# 비동기 방식으로 텔레그램 메시지를 전송하는 함수
async def send_telegram_message(message):
    try:
        async with bot:
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
    except Exception as e:
        logging.error(f"[ERROR] 텔레그램 메시지 전송 실패: {e}")

# =======================
# 데이터 및 지표 계산 함수
# =======================
# --- Binance 선물 데이터 가져오기 ---
def get_binance_data(symbol, interval, limit=100):
    """
    Binance 선물 API를 통해 캔들스틱 데이터(OHLCV)를 가져와 DataFrame으로 반환하는 함수
    :param symbol: 거래 심볼 (예: BTCUSDT)
    :param interval: 캔들 간격 (예: 15분봉)
    :param limit: 가져올 데이터 개수 (기본값: 100)
    """
    try:
        # Binance API의 futures_klines 함수를 통해 캔들 데이터 조회
        klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        # DataFrame으로 변환 및 컬럼명 지정
        df = pd.DataFrame(klines, columns=[
            'time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        # 필요한 컬럼만 선택: 시간, 시가, 고가, 저가, 종가, 거래량
        df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
        # 밀리초 단위를 datetime 형식으로 변환
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        # 숫자형 데이터로 변환
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        return df
    except Exception as e:
        logging.error(f"[ERROR] 데이터를 가져오는 중 오류 발생: {e}")
        return None

# --- 기술적 지표 계산 ---
def calculate_indicators(df):
    """
    주어진 데이터프레임에 다양한 기술적 지표를 계산하여 컬럼으로 추가
    사용 지표: EMA, RSI, MACD, ADX, ATR, Bollinger Bands, Stochastic Oscillator, OBV, CCI
    """
    # EMA (Exponential Moving Average): 최근 데이터에 더 큰 가중치를 부여한 이동평균
    df['ema_9'] = EMAIndicator(df['close'], window=EMA_SHORT_WINDOW).ema_indicator()
    df['ema_21'] = EMAIndicator(df['close'], window=EMA_LONG_WINDOW).ema_indicator()

    # RSI (Relative Strength Index): 과매수/과매도 상태 판단에 사용
    df['rsi'] = RSIIndicator(df['close'], window=RSI_WINDOW).rsi()
    
    # MACD (Moving Average Convergence Divergence): 두 이동평균 간의 차이를 이용한 모멘텀 지표
    macd_obj = MACD(df['close'], window_fast=MACD_FAST, window_slow=MACD_SLOW, window_sign=MACD_SIGNAL)
    df['macd'] = macd_obj.macd()
    df['macd_signal'] = macd_obj.macd_signal()
    
    # ADX (Average Directional Index): 추세 강도 측정 지표
    df['adx'] = ADXIndicator(df['high'], df['low'], df['close']).adx()
    # ATR (Average True Range): 시장 변동성 측정 지표
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=ATR_WINDOW).average_true_range()

    # Bollinger Bands: 가격의 상한/하한 밴드 계산 (변동성 측정)
    bollinger = BollingerBands(df['close'])
    df['bollinger_high'] = bollinger.bollinger_hband()
    df['bollinger_low'] = bollinger.bollinger_lband()

    # Stochastic Oscillator: 모멘텀 기반 지표 (과매수/과매도 판단)
    stoch = StochasticOscillator(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    # OBV (On Balance Volume): 거래량 기반 모멘텀 지표
    df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
    # CCI (Commodity Channel Index): 가격과 이동평균 간의 차이를 측정하여 과매수/과매도 판단
    df['cci'] = CCIIndicator(high=df['high'], low=df['low'], close=df['close'], window=CCI_WINDOW).cci()

    return df

# --- 매물대 기반 지지선/저항선 계산 ---
def calculate_support_resistance(df):
    """
    최근 데이터에서 자주 거래된 가격(매물대)를 기반으로 지지선과 저항선을 추정
    :return: support (지지선), resistance (저항선)
    """
    # 가격을 소수점 둘째자리로 반올림하여 그룹화 (거래 밀집도를 파악)
    df['rounded_price'] = df['close'].round(2)  # 가격을 반올림하여 그룹화
    # 각 가격대별 거래 건수에 따라 내림차순 정렬 후 가격 리스트 추출
    price_levels = df['rounded_price'].value_counts().sort_values(ascending=False).index.tolist()
    if len(price_levels) < 5:
        return min(price_levels), max(price_levels)
    # 상위 5개의 가격 중 최소값을 지지선, 최대값을 저항선으로 사용
    support = min(price_levels[:5])
    resistance = max(price_levels[:5])
    return support, resistance

# =======================
# 시장 환경 분류 함수
# =======================
def determine_market_regime(df):
    """
    시장의 현재 상태를 분류하는 함수
    - ADX, Bollinger Bands 폭, 그리고 EMA 간의 차이를 함께 고려하여
      추세가 뚜렷하면 'trending'을, 그렇지 않으면 'sideways'를 반환.
    """
    adx = df['adx'].iloc[-1]
    bb_width = (df['bollinger_high'].iloc[-1] - df['bollinger_low'].iloc[-1]) / df['close'].iloc[-1]
    ema_9 = df['ema_9'].iloc[-1]
    ema_21 = df['ema_21'].iloc[-1]

    # EMA 간의 상대적 차이가 0.5% 이상이면 추세로 판단 (이전 1%보다 민감하게)
    if abs(ema_9 - ema_21) / ema_21 >= 0.005:
        return "trending"
    # ADX와 Bollinger Bands 조건도 충족하면 추세로 판단
    elif adx > 25 and bb_width > 0.05:
        return "trending"
    else:
        return "sideways"
    
# =======================
# 개선된 진입 필터
# =======================
def improved_entry_filter(df, strategy):
    """
    진입 신호의 강도를 확인하기 위해 여러 조건 (MACD, 거래량, RSI, OBV, CCI)을 추가로 평가
    :param df: 기술적 지표가 추가된 DataFrame
    :param strategy: 선택된 전략 (예: 'trend_following', 'mean_reversion_buy' 등)
    :return: 조건 충족 시 True, 아니면 False
    """
    # 데이터가 충분치 않으면 False 반환
    if len(df) < 2:
        return False

    # MACD 차이 계산: 현재와 이전의 MACD와 시그널 라인의 차이
    current_diff = df['macd'].iloc[-1] - df['macd_signal'].iloc[-1]
    previous_diff = df['macd'].iloc[-2] - df['macd_signal'].iloc[-2]
    
    # 롱(매수) 포지션 조건 체크
    if strategy in ['trend_following', 'mean_reversion_buy']:
        if previous_diff == 0:
            # 이전 차이가 0이면 현재 차이가 미미하면 진입 차단
            if abs(current_diff) < 0.01:
                logging.info("MACD 모멘텀이 충분하지 않음 (롱): 이전 차이가 0이고 현재 차이가 너무 작음")
                return False
        else:
            ratio = current_diff / previous_diff
            # 수정: 롱 진입 조건 임계치를 기존 0.9에서 0.8로 완화하여 롱 진입 기회를 확대함
            if ratio <= 0.8:
                logging.info(f"MACD 모멘텀이 완화된 조건 미달 (롱): 비율 {ratio:.2f} <= 0.8")
                return False
        
        # 수정: 롱 포지션에 대해 RSI 조건 완화 (기존 RSI > 70 대신 RSI > 80 차단)
        rsi = df['rsi'].iloc[-1]
        if rsi > 70:
            logging.info(f"RSI 과매수 상태 (롱): RSI {rsi:.2f} > 70")
            return False

    # 숏(매도) 포지션 조건 체크 (비율 조건은 롱과 반대)
    if strategy in ['trend_following_down', 'mean_reversion_sell']:
        if previous_diff == 0:
            if abs(current_diff) < 0.01:
                logging.info("MACD 모멘텀이 충분하지 않음 (숏): 이전 차이가 0이고 현재 차이가 너무 작음")
                return False
        else:
            ratio = current_diff / previous_diff
            if ratio >= 1.1:
                logging.info(f"MACD 모멘텀이 완화된 조건 미달 (숏): 비율 {ratio:.2f} >= 1.1")
                return False
            
        rsi = df['rsi'].iloc[-1]
        if rsi < 40:
            logging.info(f"RSI 과매도 상태 (숏): RSI {rsi:.2f} < 40")
            return False

    # 거래량 조건: 최근 20개 거래량의 평균과 비교하여 현재 거래량이 낮으면 진입 차단
    avg_volume = df['volume'].rolling(20).mean().iloc[-1]
    current_volume = df['volume'].iloc[-1]
    if current_volume < avg_volume:
        logging.info("거래량 부족: 현재 거래량이 평균보다 낮음")
        return False

    # OBV (On Balance Volume) 조건: 거래량 기반 모멘텀 체크
    obv_current = df['obv'].iloc[-1]
    obv_previous = df['obv'].iloc[-2]
    ratio_obv = obv_current / obv_previous if obv_previous != 0 else 1

    if strategy in ['trend_following', 'mean_reversion_buy']:
        # 롱의 경우, OBV가 약 5% 미만만 하락한 경우만 허용
        if ratio_obv < 0.95:
            logging.info(f"OBV 상승 미달 (롱): 비율 {ratio_obv:.2f} < 0.95")
            return False

    if strategy in ['trend_following_down', 'mean_reversion_sell']:
        # 숏의 경우, OBV가 약 5% 이상 상승한 경우만 차단
        if ratio_obv > 1.05:
            logging.info(f"OBV 하락 미달 (숏): 비율 {ratio_obv:.2f} > 1.05")
            return False

    # CCI (Commodity Channel Index) 조건: 과매수(CCI > 100) 또는 과매도(CCI < -100) 상태 체크
    cci_current = df['cci'].iloc[-1]
    if strategy in ['trend_following', 'mean_reversion_buy'] and cci_current > 100:
        logging.info("CCI가 과매수 상태여서 진입 차단 (롱)")
        return False
    if strategy in ['trend_following_down', 'mean_reversion_sell'] and cci_current < -100:
        logging.info("CCI가 과매도 상태여서 진입 차단 (숏)")
        return False

    return True

# =======================
# 동적 TP/SL 계산 (변동성 고려)
# =======================
def calculate_dynamic_tp_sl(price, atr, support, resistance, strategy, volatility_factor=1.0):
    """
    동적 TP (Take Profit)와 SL (Stop Loss) 가격을 계산하는 함수
    - ATR: Average True Range, 시장 변동성 지표
    - support, resistance: 매물대 기반 지지선/저항선
    - volatility_factor: 변동성에 따른 조정 인자
    """
    # 변동성이 클 경우 기본 배수를 낮춰서 위험을 줄임
    if volatility_factor > 1.5:
        tp_multiplier_base = 1.5
        sl_multiplier_base = 1.0
    else:
        tp_multiplier_base = 2.0
        sl_multiplier_base = 1.5

    if strategy in ['trend_following', 'mean_reversion_buy']:
        distance_to_resistance = resistance - price
        tp_multiplier = tp_multiplier_base if distance_to_resistance > atr else 0.5
        tp = price + (atr * tp_multiplier)
        distance_to_support = price - support
        sl_multiplier = sl_multiplier_base if distance_to_support > atr else 0.5
        sl = price - (atr * sl_multiplier)
    elif strategy in ['trend_following_down', 'mean_reversion_sell']:
        distance_to_support = price - support
        tp_multiplier = tp_multiplier_base if distance_to_support > atr else 0.5
        tp = price - (atr * tp_multiplier)
        distance_to_resistance = resistance - price
        sl_multiplier = sl_multiplier_base if distance_to_resistance > atr else 0.5
        sl = price + (atr * sl_multiplier)
    return round(tp, 2), round(sl, 2)

# =======================
# 리스크 관리: 포지션 사이즈 계산 (예: 1% 위험)
# =======================
def calculate_position_size(price, sl, risk_percent=0.01):
    """
    계좌 잔고와 위험 비율을 바탕으로 포지션(거래) 크기를 산출
    - risk_percent: 계좌 잔고의 몇 %를 위험에 노출할지 (예: 0.01은 1%)
    """
    balance = get_available_balance()
    risk_amount = balance * risk_percent  # 위험 금액 계산
    risk_per_unit = abs(price - sl)       # 단위당 위험 금액
    if risk_per_unit == 0:
        return 0
    quantity = risk_amount / risk_per_unit
    return round(quantity, 3)

# =======================
# 포지션 관리 함수
# =======================
# --- 현재 포지션 확인 ---
def get_open_position():
    """
    현재 열려 있는 포지션을 확인하는 함수
    Binance 선물 API의 futures_position_information() 함수를 사용하여
    특정 SYMBOL에 대해 포지션이 0이 아닌 경우 반환
    """
    try:
        positions = client.futures_position_information()
        for pos in positions:
            if pos['symbol'] == SYMBOL and float(pos['positionAmt']) != 0:
                return pos
    except Exception as e:
        logging.error(f"[ERROR] 포지션 정보를 가져오는 중 오류 발생: {e}")
    return None

# --- 포지션 종료 함수 추가 ---
def close_current_position():
    """
    현재 열려 있는 포지션을 종료하는 함수
    - 롱 포지션은 매도 주문, 숏 포지션은 매수 주문을 통해 청산
    - 기존의 모든 미체결 주문을 취소한 후 실행
    """
    pos = get_open_position()
    if pos:
        position_amt = float(pos['positionAmt'])
        # 포지션의 방향에 따라 반대 주문 결정
        closing_side = SIDE_SELL if position_amt > 0 else SIDE_BUY
        try:
            # 기존 미체결 주문 취소
            client.futures_cancel_all_open_orders(symbol=SYMBOL)
            order = client.futures_create_order(
                symbol=SYMBOL,
                side=closing_side,
                type='MARKET',
                quantity=abs(position_amt),
                reduceOnly=True
            )
            logging.info(f"현재 포지션 종료: {closing_side} 주문, 수량: {abs(position_amt)}")
            # 텔레그램 알림 전송 (비동기)
            asyncio.create_task(send_telegram_message(f"현재 포지션 종료: {closing_side} 주문, 수량: {abs(position_amt)}"))
            return order
        except Exception as e:
            logging.error(f"[ERROR] 포지션 종료 실패: {e}")
            asyncio.create_task(send_telegram_message(f"[ERROR] 포지션 종료 실패: {e}"))
            return None
    return None

# =======================
# 전략 선택: 시장 환경 분기 및 개선된 진입 필터 적용
# =======================
def determine_trading_strategy(df):
    """
    시장 데이터와 계산된 기술적 지표를 바탕으로 거래 전략을 결정하는 함수
    전략 종류:
      - 'trend_following': 상승 추세 추종 (롱)
      - 'trend_following_down': 하락 추세 추종 (숏)
      - 'mean_reversion_buy': 평균회귀 매수 (횡보장에서 과매도 시)
      - 'mean_reversion_sell': 평균회귀 매도 (횡보장에서 과매수 시)
      - 'neutral': 거래 신호 없음
    """
    reasons = []  # 관망 상태인 이유를 저장
    regime = determine_market_regime(df)
    macd_line = df['macd'].iloc[-1]
    macd_signal = df['macd_signal'].iloc[-1]
    ema_9 = df['ema_9'].iloc[-1]
    ema_21 = df['ema_21'].iloc[-1]
    ema_uptrend = ema_9 > ema_21
    ema_downtrend = ema_9 < ema_21

    # 추가: 디버깅 로그로 주요 지표 확인 (롱/숏 판단에 도움)
    logging.info(f"전략 결정 디버깅 - regime: {regime}, EMA: {ema_9:.2f}/{ema_21:.2f}, MACD: {macd_line:.2f}/{macd_signal:.2f}, RSI: {df['rsi'].iloc[-1]:.2f}")

    # 과매수/과매도 조건 판단 (RSI, Stochastic, Bollinger 기준)
    oversold = (df['rsi'].iloc[-1] < 30 and
                df['stoch_k'].iloc[-1] < df['stoch_d'].iloc[-1] and
                df['close'].iloc[-1] <= df['bollinger_low'].iloc[-1])
    overbought = (df['rsi'].iloc[-1] > 70 and
                  df['stoch_k'].iloc[-1] > df['stoch_d'].iloc[-1] and
                  df['close'].iloc[-1] >= df['bollinger_high'].iloc[-1])

    if regime == "trending":
        if ema_uptrend:
            # 수정: 롱 진입 조건 완화 - MACD 차이가 약간 음수(-0.05 미만이면 차단)인 경우에도 롱으로 판단
            if (macd_line - macd_signal) > -0.05:
                strategy = 'trend_following'
            else:
                strategy = 'neutral'
                reasons.append("MACD long condition not met")
        elif ema_downtrend:
            if macd_line < macd_signal:
                strategy = 'trend_following_down'
            else:
                strategy = 'neutral'
                reasons.append("MACD short condition not met")
        else:
            strategy = 'neutral'
            reasons.append("EMA 추세 확인 불가")
    else:  # regime == "sideways"
        # 횡보장에서는 EMA가 상승이면 평균회귀 매도/매수 신호를 활용
        if ema_uptrend:
            if overbought:
                strategy = 'mean_reversion_sell'
            elif oversold:
                strategy = 'mean_reversion_buy'
            else:
                strategy = 'neutral'
                reasons.append("횡보장에서 과매수/과매도 조건 미충족")
        # 횡보장이라도 EMA가 하락세라면, 하락 추세의 가능성이 있으므로 trend_following_down 적용
        elif ema_downtrend:
            # 개선: 횡보장에서 EMA가 하락하더라도 과매도 상태이면 매수 신호 허용
            if oversold:
                strategy = 'mean_reversion_buy'
            else:
                strategy = 'trend_following_down'
        else:
            strategy = 'neutral'

    # 개선된 진입 필터 적용 (필터 통과 실패 시 중립)
    if strategy != 'neutral' and not improved_entry_filter(df, strategy):
        reasons.append("개선된 진입 필터 미충족")
        strategy = 'neutral'

    if strategy == 'neutral' and reasons:
        logging.info("관망 상태 유지: " + ", ".join(reasons))

    return strategy

# =======================
# 변동성 체크
# =======================
atr_threshold = 2.5          # ATR 배수 기준: ATR 값이 평균의 2.5배 이상이면 변동성 과다로 판단
max_volatility_duration = 5  # 변동성이 지속되는 최대 시간 (분 단위)

# 최근 N개의 ATR 평균을 구해서 변동성이 큰지 확인하는 함수
def is_high_volatility(df, atr_threshold=atr_threshold):
    """
    최근의 ATR 값과 최근 20개의 ATR 평균을 비교하여 변동성이 지나치게 큰지 판단
    :return: 변동성이 크면 True, 그렇지 않으면 False
    """
    recent_atr = df['atr'].iloc[-1]  # 현재 ATR 값
    avg_atr = df['atr'].rolling(20).mean().iloc[-1]  # 최근 20개 ATR 평균

    if recent_atr > avg_atr * atr_threshold:
        logging.warning(f"[WARNING] 변동성이 너무 큼 (ATR: {recent_atr}, 평균 ATR: {avg_atr}) -> 진입 금지")
        asyncio.create_task(send_telegram_message(f"[WARNING] 변동성이 너무 큼 (ATR: {recent_atr}, 평균 ATR: {avg_atr}) -> 진입 금지"))
        return True
    return False

# =======================
# 주문 실행: 동적 TP/SL 및 리스크 관리 적용
# =======================
def place_order(symbol, side, price, atr, df, strategy):
    """
    주문을 실행하는 함수
    - 동적 TP/SL 계산, 포지션 사이즈 산정(리스크 관리) 후 시장가 주문 실행
    - 주문 실행 후 TP (이익 실현) 및 SL (손절) 주문도 별도로 생성
    """
    try:
        # 변동성이 높으면 주문 실행하지 않음
        if is_high_volatility(df):
            return

        # 시장 평균 ATR과 현재 ATR을 이용한 변동성 팩터 계산
        avg_atr = df['atr'].rolling(20).mean().iloc[-1]
        volatility_factor = (atr / avg_atr) if avg_atr else 1.0

        # 매물대 기반 지지선 및 저항선 계산
        support, resistance = calculate_support_resistance(df)
        # 동적 TP/SL 계산
        tp, sl = calculate_dynamic_tp_sl(price, atr, support, resistance, strategy, volatility_factor)

        # 포지션 사이즈 결정:
        # - 리스크 관리가 활성화되면 calculate_position_size 함수 사용
        # - USE_CALCULATED_QUANTITY가 True이면 calculate_max_quantity 함수 사용
        # - 그렇지 않으면 고정 수량 사용 (0.05)
        if USE_RISK_MANAGEMENT:
            quantity = calculate_position_size(price, sl)
        elif USE_CALCULATED_QUANTITY:
            quantity = calculate_max_quantity(price)
        else:
            quantity = 0.05

        if quantity <= 0:
            logging.warning("[WARNING] 주문 가능 수량이 없습니다.")
            return

        # 기존 미체결 주문 취소 후 주문 실행
        client.futures_cancel_all_open_orders(symbol=symbol)
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type='MARKET',  # 시장가 주문: 현재 가격으로 즉시 체결
            quantity=quantity
        )
        logging.info(f"{side} 주문 실행: {quantity} {symbol} | TP: {tp}, SL: {sl}")
        asyncio.create_task(send_telegram_message(f"{side} 주문 실행: {quantity} {symbol}\nTP: {tp}\nSL: {sl}"))

        # 이익 실현(Take Profit) 주문 생성 (반대 주문으로 TP 달성 시 포지션 종료)
        client.futures_create_order(
            symbol=symbol,
            side=SIDE_SELL if side == SIDE_BUY else SIDE_BUY,
            type='TAKE_PROFIT_MARKET',
            stopPrice=tp,
            quantity=quantity,
            reduceOnly=True
        )

        # 손절(Stop Loss) 주문 생성
        client.futures_create_order(
            symbol=symbol,
            side=SIDE_SELL if side == SIDE_BUY else SIDE_BUY,
            type='STOP_MARKET',
            stopPrice=sl,
            quantity=quantity,
            reduceOnly=True
        )
        return order
    except Exception as e:
        logging.error(f"[ERROR] TP/SL 주문 실행 실패: {e}")
        asyncio.create_task(send_telegram_message(f"[ERROR] TP/SL 주문 실행 실패: {e}"))
        return None

# --- 최대 거래 수량 계산 ---
def calculate_max_quantity(price):
    """
    사용 가능한 잔고와 레버리지를 이용해 최대 거래 가능한 수량을 계산
    :return: 최대 수량 (소수점 2자리까지 반올림)
    """
    available_balance = get_available_balance()
    max_position_value = available_balance * LEVERAGE  # 레버리지 적용 최대 포지션 가치
    max_quantity = (max_position_value - 100) / price  # 100 단위의 마진을 제외한 최대 수량 산출
    return round(max_quantity, 2)  # 정밀도 맞춤

# --- 잔고 조회 함수 ---
def get_available_balance():
    """
    Binance 선물 계좌의 USDT 잔고를 반환하는 함수
    """
    try:
        balance_info = client.futures_account_balance()
        for asset in balance_info:
            if asset['asset'] == 'USDT':
                return float(asset['availableBalance'])  # 사용 가능한 USDT 잔고
    except Exception as e:
        logging.error(f"[ERROR] 잔고 조회 실패: {e}")
        return 0

# =======================
# 라이브 트레이딩 루프
# =======================
high_volatility_counter = 0  # 연속된 변동성 경고 횟수를 기록 (분 단위)
position_alert_sent = False  # 중복 알림 방지 플래그

async def run_trading_bot():
    """
    메인 거래 루프: 주기적으로 데이터를 조회, 지표 계산, 전략 판단 후 주문 실행 또는 포지션 관리
    """
    global high_volatility_counter, position_alert_sent
    while True:
        try:
            # 15분봉 데이터를 Binance에서 조회
            df = get_binance_data(SYMBOL, Client.KLINE_INTERVAL_15MINUTE)
            if df is not None:
                # 기술적 지표 계산
                df = calculate_indicators(df)
                # 전략 결정 (추세 추종, 평균회귀 등)
                strategy = determine_trading_strategy(df)
                price = df.iloc[-1]['close']
                atr = df.iloc[-1]['atr']
                logging.info(f"현재 BTC 가격: {price}")

                # 변동성이 높은 경우 카운터 증가 후 일정 시간 지속 시 거래 중단 알림
                if is_high_volatility(df):
                    high_volatility_counter += 1
                    logging.warning(f"[WARNING] 변동성이 강하게 유지됨. {high_volatility_counter}/{max_volatility_duration}분")
                    if high_volatility_counter >= max_volatility_duration:
                        logging.warning("[WARNING] 변동성이 너무 강하여 일시적으로 거래 중단.")
                        await send_telegram_message("⚠️ 강한 변동성이 지속되어 거래를 일시적으로 중단합니다.")
                        await asyncio.sleep(60)
                        continue
                else:
                    high_volatility_counter = 0

                # 현재 열려 있는 포지션 확인
                open_position = get_open_position()
                if open_position is None:
                    position_alert_sent = False  # 새 포지션 생성 시 알림 리셋
                    if strategy in ['trend_following', 'mean_reversion_buy']:
                        place_order(SYMBOL, SIDE_BUY, price, atr, df, strategy)
                        await send_telegram_message("🚀 전략 실행: 매수 주문")
                    elif strategy in ['trend_following_down', 'mean_reversion_sell']:
                        place_order(SYMBOL, SIDE_SELL, price, atr, df, strategy)
                        await send_telegram_message("📉 전략 실행: 매도 주문")
                else:
                    if not position_alert_sent:
                        logging.info(f"이미 열린 포지션이 있음, 추가 주문 방지, strategy: {strategy}")
                        await send_telegram_message(f"🔄 이미 열린 포지션이 있어 추가 주문을 방지합니다. strategy: {strategy}")
                        position_alert_sent = True
            await asyncio.sleep(60)  # 1분마다 반복
        except Exception as e:
            logging.error(f"[ERROR] 트레이딩 봇 오류: {e}")
            await send_telegram_message(f"[ERROR] 트레이딩 봇 오류: {e}")
            await asyncio.sleep(60)

# --- 메인 실행 (실시간 거래) ---
async def main():
    """
    프로그램 시작 시 텔레그램 메시지 전송 후 메인 거래 루프 실행
    """
    await send_telegram_message(f"🚀 BTC 트레이딩 봇 시작! 심볼: {SYMBOL}, 레버리지: {LEVERAGE}")
    await run_trading_bot()

# =======================
# 백테스팅 기능 (라이브 로직과 동일)
# =======================
def backtest_trading_strategy(df):
    """
    과거 데이터를 기반으로 거래 전략을 테스트하는 백테스팅 함수
    - 각 거래(진입, 청산) 시점과 PnL(손익)을 기록
    :return: 거래 내역(trades) 리스트
    """
    trades = []
    position = None
    position_size = 0.05  # 고정 포지션 사이즈 (실제 거래와 다를 수 있음)
    # 초기 100개 데이터는 건너뛰고 이후 데이터부터 시뮬레이션
    for i in range(100, len(df)):
        current_data = df.iloc[:i+1].copy()
        current_row = current_data.iloc[-1]
        price = current_row['close']
        atr = current_row['atr']
        strategy = determine_trading_strategy(current_data)
        support, resistance = calculate_support_resistance(current_data)
        avg_atr = current_data['atr'].rolling(20).mean().iloc[-1]
        volatility_factor = (atr / avg_atr) if avg_atr else 1.0
        
        # 포지션이 없으면 새 포지션 생성
        if position is None:
            if strategy in ['trend_following', 'mean_reversion_buy']:
                tp, sl = calculate_dynamic_tp_sl(price, atr, support, resistance, strategy, volatility_factor)
                position = {
                    'type': 'long',
                    'entry_price': price,
                    'tp': tp,
                    'sl': sl,
                    'entry_index': i,
                    'entry_time': current_row['time']
                }
            elif strategy in ['trend_following_down', 'mean_reversion_sell']:
                tp, sl = calculate_dynamic_tp_sl(price, atr, support, resistance, strategy, volatility_factor)
                position = {
                    'type': 'short',
                    'entry_price': price,
                    'tp': tp,
                    'sl': sl,
                    'entry_index': i,
                    'entry_time': current_row['time']
                }
        else:
            # 포지션이 있을 경우, 가격이 TP나 SL에 도달했는지 확인하여 포지션 청산
            if position['type'] == 'long':
                if current_row['low'] <= position['sl']:
                    exit_price = position['sl']
                    trade = {
                        'type': 'long',
                        'entry_index': position['entry_index'],
                        'exit_index': i,
                        'entry_time': position['entry_time'],
                        'exit_time': current_row['time'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'pnl': (exit_price - position['entry_price']) * position_size
                    }
                    trades.append(trade)
                    position = None
                elif current_row['high'] >= position['tp']:
                    exit_price = position['tp']
                    trade = {
                        'type': 'long',
                        'entry_index': position['entry_index'],
                        'exit_index': i,
                        'entry_time': position['entry_time'],
                        'exit_time': current_row['time'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'pnl': (exit_price - position['entry_price']) * position_size
                    }
                    trades.append(trade)
                    position = None
            elif position['type'] == 'short':
                if current_row['high'] >= position['sl']:
                    exit_price = position['sl']
                    trade = {
                        'type': 'short',
                        'entry_index': position['entry_index'],
                        'exit_index': i,
                        'entry_time': position['entry_time'],
                        'exit_time': current_row['time'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'pnl': (position['entry_price'] - exit_price) * position_size
                    }
                    trades.append(trade)
                    position = None
                elif current_row['low'] <= position['tp']:
                    exit_price = position['tp']
                    trade = {
                        'type': 'short',
                        'entry_index': position['entry_index'],
                        'exit_index': i,
                        'entry_time': position['entry_time'],
                        'exit_time': current_row['time'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'pnl': (position['entry_price'] - exit_price) * position_size
                    }
                    trades.append(trade)
                    position = None
    return trades

def run_backtest():
    """
    백테스트 전체 실행 함수:
      1. Binance 데이터 조회
      2. 지표 계산
      3. 거래 전략 백테스팅 실행
      4. 결과 출력 및 로그 기록
    """
    df = get_binance_data(SYMBOL, Client.KLINE_INTERVAL_15MINUTE, limit=1000)
    if df is None:
        logging.error("백테스트 데이터를 가져오지 못했습니다.")
        return
    df = calculate_indicators(df)
    trades = backtest_trading_strategy(df)
    if trades:
        trades_df = pd.DataFrame(trades)
        total_pnl = trades_df['pnl'].sum()
        logging.info(f"백테스트 완료: {len(trades)}건의 거래, 총 PnL: {total_pnl:.2f}")
        print("백테스트 거래 내역:")
        print(trades_df)
    else:
        logging.info("백테스트 결과 거래가 발생하지 않았습니다.")

# =======================
# 실행 모드 선택
# =======================
if __name__ == "__main__":
    mode = input("실행 모드를 선택하세요 ('live' 또는 'backtest'): ").strip().lower()
    if mode == 'backtest':
        run_backtest()
    else:
        asyncio.run(main())
