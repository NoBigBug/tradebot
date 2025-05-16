# 라이브러리 임포트
import asyncio
import numpy as np
import pandas as pd
import matplotlib as mpl
import logging
import joblib
import subprocess
import os
import json
import csv
import tweepy
import requests
import nltk

nltk.download('vader_lexicon')

from sklearn.cluster import KMeans
from telegram import Bot
from telegram.request import HTTPXRequest
from binance.client import Client
from datetime import datetime, timedelta, timezone, time
from textblob import TextBlob
from collections import deque
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# 외부 설정파일 및 학습 함수 import
from train_entry_strategy_model_from_csv import train_entry_strategy_from_csv
from config import BINANCE_API_KEY, BINANCE_API_SECRET, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TRADING_INTERVAL, CRYPTO_PANIC_API_KEY, NEWS_KEYWORDS, TWITTER_BEARER_TOKEN, TWITTER_KEYWORDS

# matplotlib 폰트 및 마이너스 깨짐 방지 설정
mpl.rcParams['font.family'] = 'AppleGothic'
mpl.rcParams['axes.unicode_minus'] = False

# Binance, Telegram 봇 클라이언트 생성
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
client.API_URL = 'https://fapi.binance.com'  # 선물 주소 (이미 설정돼있을 것임)
client.futures_time()  # 연결 테스트

# 서버 시간과 동기화
client.timestamp_offset = client.futures_time()['serverTime'] - int(datetime.now(timezone.utc).timestamp() * 1000)
bot = Bot(token=TELEGRAM_BOT_TOKEN, request=HTTPXRequest(connect_timeout=10.0, read_timeout=10.0))

# 포지션 및 거래 상태 전역 변수
position_state = None  # 현재 포지션: 'long', 'short', 또는 None
bak_position_state = None
entry_price = None     # 진입 가격
bak_entry_price = None
tp_order_id = None     # TP 주문 ID
sl_order_id = None     # SL 주문 ID
quantity = 0.5        # 거래 수량 (예: 0.05 BTC)
strategy_used_at_entry = None  # 0 = 역추세, 1 = 추세

# 뉴스 감지
latest_news_ids = set()

# 감정 분석기 초기화 (전역)
vader = SentimentIntensityAnalyzer()

# 최근 감정 로그 저장용 (최대 100개)
sentiment_log = deque(maxlen=100)

# 전략 설정 (기본 TP/SL 및 리스크 제한)
TP_PERCENT = 1.0        # 목표 수익률 (Take Profit)
BAK_TP_PERCENT = 1.0 
SL_PERCENT = 0.5        # 손절 기준 (Stop Loss)
BAK_SL_PERCENT = 0.5 
VOLATILITY_THRESHOLD = 2.5  # 변동성 기준 (%)
volatility_blocked = False  # 변동성 초과 시 거래 금지
bak_volatility_blocked = False
cumulative_pnl = 0.0        # 누적 수익률
bak_cumulative_pnl = 0.0
STOP_LOSS_LIMIT = -10.0     # 누적 손실 한계 (이하일 경우 중단)
last_reset_month = datetime.now().month

NEWS_API_LIMIT = 900  # 무료 요금제 기준

# 시간대 설정 (KST: 한국 시간)
KST = timezone(timedelta(hours=9))

API_USAGE_PATH = "api_usage.json"
# 기본값
api_usage = {
    "date": datetime.now().strftime("%Y-%m-%d"),
    "news_calls": 0,
    "twitter_calls": 0
}

# 로깅 레벨 설정
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# ==================================
def load_api_usage():
    global api_usage
    if os.path.exists(API_USAGE_PATH):
        try:
            with open(API_USAGE_PATH, "r") as f:
                api_usage = json.load(f)
        except Exception as e:
            logging.warning(f"⚠️ API 사용량 로드 실패: {e}")

def save_api_usage():
    try:
        with open(API_USAGE_PATH, "w") as f:
            json.dump(api_usage, f)
    except Exception as e:
        logging.warning(f"⚠️ API 사용량 저장 실패: {e}")

def check_and_increment_api_calls(api_type: str, limit: int) -> bool:
    global api_usage

    today_str = datetime.now().strftime("%Y-%m-%d")
    if api_usage["date"] != today_str:
        # 날짜 변경 시 초기화
        api_usage = {
            "date": today_str,
            "news_calls": 0,
            "twitter_calls": 0
        }

    key = f"{api_type}_calls"
    if api_usage.get(key, 0) >= limit:
        logging.warning(f"🚫 {api_type.upper()} API 호출 한도 도달 → 호출 차단")
        return False

    api_usage[key] = api_usage.get(key, 0) + 1
    save_api_usage()
    return True
# =================================
# 감정 점수 기록 함수
def log_sentiment(polarity, timestamp=None):
    timestamp = timestamp or datetime.now(timezone.utc)
    sentiment_log.append((timestamp, polarity))

def get_recent_sentiment_score(window_minutes=30):
    now = datetime.now(timezone.utc)
    scores = [score for ts, score in sentiment_log if (now - ts).total_seconds() <= window_minutes * 60]
    return np.mean(scores) if scores else 0.0  # 기본값 중립

async def monitor_twitter_loop():
    logging.info("🐦 트위터 감시 루프 시작됨")

    try:
        stream = TwitterNewsStream(TWITTER_BEARER_TOKEN)

        # 기존 규칙 제거 후 새로 등록
        rules = stream.get_rules().data
        if rules:
            stream.delete_rules([r.id for r in rules])
        stream.add_rules(tweepy.StreamRule(" OR ".join(TWITTER_KEYWORDS) + " lang:en -is:retweet"))

        stream.filter(tweet_fields=["text"])
    except Exception as e:
        logging.error(f"❌ 트위터 스트리밍 실패: {e}")

class TwitterNewsStream(tweepy.StreamingClient):
    def on_tweet(self, tweet):
        global volatility_blocked

        text = tweet.text.lower()
        if any(keyword in text for keyword in TWITTER_KEYWORDS):
            try:
                score = vader.polarity_scores(text)['compound']
            except Exception as e:
                logging.warning(f"❌ 트윗 감정 분석 실패: {e}")
                return

            if score == 0.0:
                return  # 감정 없음 → 무시

            log_sentiment(score)

            sentiment_label = (
                "긍정" if score >= 0.05 else
                "부정" if score <= -0.05 else
                "중립"
            )

            asyncio.create_task(send_telegram_message(
                f"🐦 트윗 감지: {sentiment_label.upper()} ({score:+.3f})\n\n{text[:300]}"
            ))

            if score <= -0.2 or score >= 0.5:  # ⚠️ 기준값은 조정 가능
                logging.warning(f"⚠️ {sentiment_label.upper()} 트윗 감지 → 감정 기반 경고 (전략 영향 가능)")

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return 'positive'
    elif polarity < -0.1:
        return 'negative'
    else:
        return 'neutral'

def fetch_latest_crypto_news():
    url = f"https://cryptopanic.com/api/v1/posts/?auth_token={CRYPTO_PANIC_API_KEY}&currencies=ETH&public=true"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        news_list = []
        for item in data.get('results', []):
            title = item.get('title', '').lower()
            if any(keyword in title for keyword in NEWS_KEYWORDS):
                news_list.append(item)
        return news_list
    except Exception as e:
        logging.error(f"❌ 뉴스 가져오기 실패: {e}")
        return []
    
async def monitor_news_loop():
    global volatility_blocked, latest_news_ids
    logging.info("📰 뉴스 감시 루프 시작됨")

    while True:
        if not check_and_increment_api_calls("news", NEWS_API_LIMIT):
            await send_telegram_message("📛 뉴스 API 호출 한도 초과로 감시 일시 중단됨")
            await asyncio.sleep(600)  # 10분 후 재시도
            continue

        try:
            news_items = fetch_latest_crypto_news()
            new_alerts = []

            for news in news_items:
                news_id = news['id']
                if news_id not in latest_news_ids:
                    latest_news_ids.add(news_id)
                    new_alerts.append(news)

            if new_alerts:
                for news in new_alerts:
                    title = news['title']
                    url = news.get('url', '')
                    content = title + " " + news.get('description', '')

                    try:
                        # VADER 감정 분석
                        score = vader.polarity_scores(content)['compound']
                    except Exception as e:
                        logging.warning(f"❌ 감정 분석 실패: {e}")
                        continue

                    if score == 0.0:
                        continue  # 감정 없음 → 무시

                    log_sentiment(score)

                    sentiment_label = (
                        "긍정" if score >= 0.05 else
                        "부정" if score <= -0.05 else
                        "중립"
                    )

                    # 조건부 경고만 표시
                    if score <= -0.2 or score >= 0.5:
                        await send_telegram_message(
                            f"🚨 ETH 뉴스 감지!\n📰 {title}\n🔗 {url}\n🧠 감정 점수: {sentiment_label.upper()}({score:+.3f})"
                        )
                        logging.warning(f"⚠️ {sentiment_label.upper()} 뉴스 감지 → 감정 기반 경고 (전략 영향 가능)")
                    
        except Exception as e:
            logging.error(f"❌ 뉴스 감시 중 오류: {e}")

        await asyncio.sleep(120)  # 2분 간격 확인

# 바이낸스에서 캔들 데이터 불러오기
def get_klines(symbol='ETHUSDT', interval=TRADING_INTERVAL, limit=1000):
    klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'num_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
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
def get_next_bar_close_time(interval_str='5m', buffer_seconds=5):
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
    df = df.copy()
    df = compute_features(df)  # 핵심 지표 계산 통합 함수로 분리
    df = df.dropna()

    if len(df) < 1:
        return 1, 0.0

    expected_features = [
        'ma_ratio', 'volatility', 'rsi', 'macd', 'macd_signal', 'bb_width',
        'ema_ratio_9_21', 'adx', 'atr', 'stoch_k'
    ]
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

def compute_features(df):
    df = df.copy()
    df['return'] = df['close'].pct_change()
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma_ratio'] = df['ma5'] / df['ma10']
    df['volatility'] = df['return'].rolling(5).std()
    df['rsi'] = compute_rsi(df['close'])

    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()

    ma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['bb_width'] = (2 * std20) / ma20

    df['ema9'] = df['close'].ewm(span=9).mean()
    df['ema21'] = df['close'].ewm(span=21).mean()
    df['ema_ratio_9_21'] = df['ema9'] / df['ema21']

    from ta.trend import ADXIndicator
    from ta.volatility import AverageTrueRange
    from ta.momentum import StochasticOscillator

    adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'])
    df['adx'] = adx.adx()

    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'])
    df['atr'] = atr.average_true_range()

    stoch = StochasticOscillator(close=df['close'], high=df['high'], low=df['low'])
    df['stoch_k'] = stoch.stoch()

    return df

# 진입 전략 학습용 데이터셋 생성
# 출력: features + label (0: 역추세, 1: 추세)
def generate_entry_strategy_dataset(df: pd.DataFrame, trend_model_path: str, future_window: int = 10, support_resistance_margin: float = 0.3) -> pd.DataFrame:
    data = []
    df = df.copy()
    df = compute_features(df)  # ▶️ 핵심 지표 계산 통합 함수로 분리

    # 예외 방지
    if df.isna().sum().sum() > 0:
        df = df.dropna()

    if len(df) < future_window + 30:
        return pd.DataFrame()

    # 지지/저항 계산
    support, resistance = calculate_support_resistance(df)

    # 트렌드 예측 모델 로딩
    model = joblib.load(trend_model_path)

    for i in range(30, len(df) - future_window):
        row = df.iloc[i]
        current_price = row['close']

        # 지지/저항 근접 조건 (±support_resistance_margin%)
        support_dist = abs(current_price - support) / current_price * 100
        resistance_dist = abs(current_price - resistance) / current_price * 100
        if support_dist > support_resistance_margin and resistance_dist > support_resistance_margin:
            continue

        # 트렌드 예측
        feature_cols = [
            'ma_ratio', 'volatility', 'rsi', 'macd', 'macd_signal', 'bb_width',
            'ema_ratio_9_21', 'adx', 'atr', 'stoch_k'
        ]
        features = df[feature_cols].iloc[i:i+1]

        try:
            proba = model.predict_proba(features)[0]
            trend = int(np.argmax(proba))
            confidence = float(proba[trend])
        except Exception as e:
            continue

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

        # 추가: 유의미한 수익률 차이 필터 (0.2% 이하 차이는 제외)
        if abs(pnl_trend - pnl_counter) < 0.2:
            continue

        # 라벨 결정: 누가 더 나은 수익률을 냈는가?
        label = 1 if pnl_trend > pnl_counter else 0

        sentiment_score = get_recent_sentiment_score()

        data.append({
            'ma_ratio': row['ma_ratio'],
            'volatility': row['volatility'],
            'rsi': row['rsi'],
            'macd': row['macd'],
            'macd_signal': row['macd_signal'],
            'bb_width': row['bb_width'],
            'ema_ratio_9_21': row.get('ema_ratio_9_21', np.nan),
            'adx': row.get('adx', np.nan),
            'atr': row.get('atr', np.nan),
            'stoch_k': row.get('stoch_k', np.nan),
            'dist_support': support_dist,
            'dist_resistance': resistance_dist,
            'trend': trend,
            'confidence': confidence,
            'sentiment_score': sentiment_score,
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

    if (now_kst.time() >= target_time and (last_retrain_str is None or last_retrain_str < today_str)):
        await send_telegram_message("매일 Trend 모델 재학습 시작")
        if retrain_model_by_script("train_trend_model_xgb.py"):
            await send_telegram_message("Trend 모델 재학습 완료")
            save_last_retrain_date(today_str)
        else:
            await send_telegram_message("Trend 모델 재학습 실패")

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
    if (now_kst.weekday() == 0 and now_kst.time() >= target_time and (last_entry_str is None or last_entry_str < today_str)):
        intervals = ['15m', '1h']

        for interval in intervals:
            try:
                await send_telegram_message(f"[{interval}] 전략 모델 재학습 시작")

                # 캔들 데이터 가져오기
                limit = get_auto_limit(interval=interval)
                df = get_klines(symbol='ETHUSDT', interval=interval, limit=limit)

                # 학습 데이터셋 생성
                dataset = generate_entry_strategy_dataset(df, trend_model_path=f"trend_model_xgb_{interval}.pkl")

                if dataset.empty:
                    await send_telegram_message(f"[{interval}] 학습 데이터 부족으로 재학습 생략")
                    continue

                # CSV 저장 (선택, 분석용)
                csv_path = f"entry_strategy_dataset_{interval}.csv"
                dataset.to_csv(csv_path, index=False)

                # 모델 재학습 실행
                from train_entry_strategy_model_from_csv import train_entry_strategy_from_csv
                train_entry_strategy_from_csv(csv_path=csv_path, interval=interval)

                await send_telegram_message(f"[{interval}] 전략 모델 재학습 완료")
            except Exception as e:
                await send_telegram_message(f"[{interval}] 전략 모델 재학습 실패: {e}")

        save_last_entry_retrain_date(today_str)

def retrain_model_by_script(script_path="train_trend_model_xgb.py"):
    try:
        # 트렌드 학습용 CSV 먼저 생성
        result_generate = subprocess.run(
            ["python", "generate_trend_training_data.py", TRADING_INTERVAL],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            check=True
        )
        logging.info(f"✅ 트렌드 학습 데이터 생성 완료")

        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',  # 이 한 줄로 해결
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
        symbol='ETHUSDT',
        side='BUY' if side == 'long' else 'SELL',
        type='MARKET',
        quantity=quantity
    )
    return order

def close_position(current_side: str, quantity: float):
    close_side = 'SELL' if current_side == 'long' else 'BUY'
    order = client.futures_create_order(
        symbol='ETHUSDT',
        side=close_side,
        type='MARKET',
        quantity=quantity
    )
    return order

def get_tick_size(symbol='ETHUSDT'):
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
    tick_size = get_tick_size('ETHUSDT')

    tp_price = entry_price * (1 + TP_PERCENT / 100) if side == 'long' else entry_price * (1 - TP_PERCENT / 100)
    sl_price = entry_price * (1 - SL_PERCENT / 100) if side == 'long' else entry_price * (1 + SL_PERCENT / 100)

    tp_price = str(round_to_tick(tp_price, tick_size))
    sl_price = str(round_to_tick(sl_price, tick_size))

    tp_order = client.futures_create_order(
        symbol='ETHUSDT',
        side='SELL' if side == 'long' else 'BUY',
        type='LIMIT',
        price=tp_price,
        quantity=quantity,
        timeInForce='GTC',
        reduceOnly=True
    )

    sl_order = client.futures_create_order(
        symbol='ETHUSDT',
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

def get_current_position(symbol='ETHUSDT'):
    positions = client.futures_position_information(symbol=symbol)
    for p in positions:
        pos_amt = float(p['positionAmt'])
        if abs(pos_amt) > 1e-5:  # 아주 작은 수량 무시
            side = 'long' if pos_amt > 0 else 'short'
            entry_price = float(p['entryPrice'])
            return side, entry_price
    return None, None

def check_existing_tp_sl_orders(symbol='ETHUSDT'):
    open_orders = client.futures_get_open_orders(symbol=symbol)
    tp_exists = any(o['type'] == 'LIMIT' and o['reduceOnly'] for o in open_orders)
    sl_exists = any(o['type'] == 'STOP_MARKET' and o['reduceOnly'] for o in open_orders)
    return tp_exists, sl_exists

async def check_should_exit(symbol: str, interval: str, entry_price: float, strategy: int, position_state: str) -> tuple[bool, str]:
    """
    포지션 유지 조건을 확인 후, 종료 여부 반환
    반환값: (종료 필요 여부, 설명 메시지)
    """
    try:
        df_recent = get_klines(symbol=symbol, interval=interval, limit=get_auto_limit(interval))
        new_trend, new_confidence = predict_trend_with_proba(df_recent)
        entry_model_path = f"entry_strategy_model_{interval}.pkl"
        entry_model = joblib.load(entry_model_path)

        new_features_df = generate_entry_strategy_dataset(df_recent, trend_model_path=f"trend_model_xgb_{interval}.pkl")

        if new_features_df.empty:
            return False, ""

        new_entry_row = new_features_df.iloc[-1]
        new_strategy = int(entry_model.predict(new_entry_row.drop('label', errors='ignore').values.reshape(1, -1))[0])

        expected_trend = 2 if position_state == 'long' else 0
        expected_strategy = strategy

        # 추세 변경 체크 (횡보 무시)
        if new_trend != expected_trend and new_trend != 1:
            # 신뢰도가 낮으면 종료
            if new_confidence < 0.6:
                return True, f"📉 추세 변경 + 신뢰도 낮음 ({new_confidence:.2f}) → 종료"
            else:
                logging.info(f"🔍 추세 변경은 감지됐지만 신뢰도 높음 ({new_confidence:.2f}) → 포지션 유지")
        if new_strategy != expected_strategy:
            return True, f"🔁 전략 변경 감지: {'추세' if expected_strategy else '역추세'} → {'추세' if new_strategy else '역추세'}"

        return False, ""

    except Exception as e:
        logging.error(f"❌ 포지션 유지 조건 확인 실패: {e}")
        return False, ""

async def multi_tf_trading_loop():
    global position_state, entry_price, volatility_blocked, cumulative_pnl, strategy_used_at_entry
    global TP_PERCENT, SL_PERCENT, last_reset_month, tp_order_id, sl_order_id

    # 심볼 선택(추후에는 여러 코인으로 확장)
    symbol = 'ETHUSDT'
    
    support = None
    resistance = None

    # 월별 초기화 / 손실 중단
    current_month = datetime.now().month
    if current_month != last_reset_month:
        last_reset_month = current_month
        cumulative_pnl = 0.0
        await send_telegram_message("🔄 새 달이 시작되어 누적 수익률을 초기화합니다.")

    # 복구 로직(보통 재시작됬을때 정보가져오기용)
    if position_state is None and entry_price is None:
        position_state, entry_price = get_current_position()
        
        # 기존 포지션이 복구된것은 굳이 밑에 로직을 진행할 필요가 없기떄문에 return
        if position_state:
            await send_telegram_message(f"🔁 기존 포지션 복구: {position_state.upper()} @ {entry_price}")
            tp_exists, sl_exists = check_existing_tp_sl_orders()
            if not tp_exists or not sl_exists:
                tp_order_id, sl_order_id = place_tp_sl_orders(entry_price, position_state, quantity)
                await send_telegram_message("🛠️ 누락된 TP/SL 주문을 재설정했습니다.")

            return
        
    # 포지션 종료 체크
    if position_state and entry_price:
        current_price = float(client.futures_mark_price(symbol=symbol)['markPrice'])
        change_pct = (current_price - entry_price) / entry_price * 100
        if position_state == 'short':
            change_pct *= -1

        if change_pct >= TP_PERCENT or change_pct <= -SL_PERCENT:
            label = "🎯 TP 도달" if change_pct >= TP_PERCENT else "⚠️ SL 도달"
            
            # 남아있는 tp, sl 주문 제거
            cancel_order(symbol)

            # 수익률 기록
            cumulative_pnl += change_pct

            # 알림 전송
            await send_telegram_message(
                f"{label}. {position_state.upper()} 종료\n"
                f"PnL: {change_pct:.2f}%\n"
                f"누적 PnL: {cumulative_pnl:.2f}%\n"
                f"📉 포지션 종료 완료"
            )

            # 상태 초기화
            position_state = None
            entry_price = None
            strategy_used_at_entry = None  # 전략 상태도 초기화
            tp_order_id = None
            sl_order_id = None

            await asyncio.sleep(1.5)
            logging.info("✅ 포지션 종료 후 상태 초기화 및 대기 완료")

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
            logging.warning("⚠️ Binance에는 포지션 없지만 상태 남아 있음 → 초기화")
        else:
            logging.info("중복 진입 방지: 이미 포지션이 존재함")
            return  # 이미 포지션 있음 → 진입 불가

    trend_model_path = f"trend_model_xgb_{TRADING_INTERVAL}.pkl"
    entry_model_path = f"entry_strategy_model_{TRADING_INTERVAL}.pkl"
    entry_model = joblib.load(entry_model_path)
    
    df = get_klines(symbol=symbol, interval=TRADING_INTERVAL, limit=get_auto_limit(TRADING_INTERVAL))
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
    if float(confidence) < 0.6:
        logging.info(f"❌ [{TRADING_INTERVAL}] 신뢰도 낮음({confidence * 100:.2f}%) → 진입 회피")
        return
    
    if trend == 1:
        logging.info(f"😐 [{TRADING_INTERVAL}] 횡보 예측 → 진입 회피")
        return
    
    # entry 전략 예측을 위한 feature 생성
    entry_features_df = generate_entry_strategy_dataset(df, trend_model_path=trend_model_path)
    if entry_features_df.empty:
        logging.info(f"🚫 [{TRADING_INTERVAL}] 유효한 진입 포인트 없음 → 회피")
        return

    entry_row = entry_features_df.iloc[-1]
    X_entry = entry_row.drop('label', errors='ignore').values.reshape(1, -1)
    strategy = int(entry_model.predict(X_entry)[0])  # 0 = 역추세, 1 = 추세

    def trend_to_signal(t): return 'long' if t == 2 else 'short' if t == 0 else None
    def reverse_signal(s): return 'short' if s == 'long' else 'long'

    signal = trend_to_signal(trend) if strategy == 1 else reverse_signal(trend_to_signal(trend))
    if signal is None:
        logging.info(f"🚫 [{TRADING_INTERVAL}] 진입 신호 없음 (None)")
        return

    if trend == 2 and signal == 'short':
        await send_telegram_message(f"📈 [{TRADING_INTERVAL}] 상승 추세인데 숏 진입 시도 → 회피")
        return
    
    if trend == 0 and signal == 'long':
        await send_telegram_message(f"📉 [{TRADING_INTERVAL}] 하락 추세인데 롱 진입 시도 → 회피")
        return

    # confidence 기반 TP/SL 조정
    if float(confidence) >= 0.8:
        TP_PERCENT, SL_PERCENT = 1.8, 0.3
    elif float(confidence) >= 0.6:
        TP_PERCENT, SL_PERCENT = 1.0, 0.5
    else:
        TP_PERCENT, SL_PERCENT = 0.7, 0.5

    await send_telegram_message(
        f"📡 [{TRADING_INTERVAL}] 진입 신호 발생\n"
        f"🧠 추세 예측: {predict_trend_text(trend)} / 전략: {'추세' if strategy == 1 else '역추세'}\n"
        f"📊 신뢰도: {confidence * 100:.2f}%\n"
        f"🎯 TP: {TP_PERCENT}%, SL: {SL_PERCENT}%\n"
        f"지지선: {support} / 저항선: {resistance}\n"
        f"📌 진입 방향: {signal.upper()}"
    )

    current_price = float(client.futures_mark_price(symbol=symbol)['markPrice'])

    # 포지션 진입
    try:
        order = place_order(signal, quantity)
    except Exception as e:
        await send_telegram_message(f"❌ 주문 실패: {e}")
        return
    await asyncio.sleep(1.5)  # 체결 대기 (Binance 응답 속도 고려)

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
    strategy_used_at_entry = strategy  # 전략 저장
    
    tp_price = round(entry_price * (1 + TP_PERCENT / 100), 2) if signal == 'long' else round(entry_price * (1 - TP_PERCENT / 100), 2)
    sl_price = round(entry_price * (1 - SL_PERCENT / 100), 2) if signal == 'long' else round(entry_price * (1 + SL_PERCENT / 100), 2)

    await send_telegram_message(
        f"🚀 [{TRADING_INTERVAL}] 진입 완료: {signal.upper()} @ {entry_price}\n"
        f"🎯 TP: {tp_price}\n"
        f"⚠️ SL: {sl_price}"
    )

    logging.info(f"✅ [{TRADING_INTERVAL}] 진입 완료: {signal.upper()} @ {entry_price:.2f} | TP: {tp_price}, SL: {sl_price}")

async def start_bot():
    await send_telegram_message(f"트레이딩봇 시작.")
    logging.info("프로그램 시작됨. 다음 봉 마감까지 대기 중...")

    load_api_usage()

    # 뉴스 감지 루프 및 트레이딩 루프 동시 실행
    await asyncio.gather(    
        monitor_news_loop(),        # 뉴스 API 감시
        monitor_twitter_loop(),     # 트위터 감시
        trading_loop_wrapper()      # 기존 트레이딩 루프
    )

async def trading_loop_wrapper():
    while True:
        await maybe_retrain_daily()             # 기존 trend 모델 재학습
        await maybe_retrain_entry_strategy()    # 새로운 entry 전략 모델 재학습 

        # 다음 봉 마감 시점 계산 (예: 현재 시각이 09:14:53 → 09:15:00 마감까지 7초 남음)
        sleep_sec = get_next_bar_close_time(TRADING_INTERVAL)
        logging.info(f"다음 봉 마감까지 {sleep_sec:.2f}초 대기...")
        await asyncio.sleep(sleep_sec)

        try:
            await multi_tf_trading_loop()
        except SystemExit:
            break
        except Exception as e:
            await send_telegram_message(f"오류 발생: {e}")

def predict_trend_sync(df: pd.DataFrame, model_path=f"trend_model_xgb_{TRADING_INTERVAL}.pkl") -> tuple[int, float]:
    df = df.copy()
    df = compute_features(df)  # 핵심 지표 계산 통합 함수로 분리

    df = df.dropna()

    if len(df) < 1:
        return 1, 0.0

    expected_features = [
        'ma_ratio', 'volatility', 'rsi', 'macd', 'macd_signal', 'bb_width',
        'ema_ratio_9_21', 'adx', 'atr', 'stoch_k'
    ]
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
    intervals = ['15m', '1h']
    summary_results = {}

    for interval in intervals:
        logging.info(f"\n[{interval}] 백테스트 시작\n")
        pnl = await test_backtest_bot(interval=interval)
        summary_results[interval] = pnl

    # 결과 요약 출력
    logging.info("\n전체 백테스트 요약\n")
    for interval, pnl in summary_results.items():
        sign = "+" if pnl >= 0 else ""
        logging.info(f"{interval:>3}  →  누적 PnL: {sign}{pnl:.2f}%")

def predict_entry_strategy_from_row(row: pd.Series, model_path: str):
    import joblib

    model = joblib.load(model_path)

    # 학습 당시 feature 리스트 (고정)
    feature_cols = [
        'ma_ratio', 'volatility', 'rsi', 'macd', 'macd_signal',
        'bb_width', 'ema_ratio_9_21', 'adx', 'atr', 'stoch_k',
        'dist_support', 'dist_resistance', 'trend', 'confidence'
    ]

    features = row[feature_cols].values.reshape(1, -1)
    pred = model.predict(features)
    return int(pred[0])  # 0 = 역추세, 1 = 추세

summary_results = {}

async def backtest_bot(interval='15m', isLogShow=True) -> float:
    import joblib
    global bak_position_state, bak_entry_price, bak_volatility_blocked
    global BAK_TP_PERCENT, BAK_SL_PERCENT

    bak_cumulative_pnl = 0.0
    bak_strategy_used_at_entry = None

    df = get_klines(symbol='ETHUSDT', interval=interval, limit=1000)
    trend_model_path = f"trend_model_xgb_{interval}.pkl"
    entry_model_path = f"entry_strategy_model_{interval}.pkl"
    entry_model = joblib.load(entry_model_path)

    for i in range(100, len(df)):
        sliced_df = df.iloc[:i].copy()
        current_price = sliced_df['close'].iloc[-1]
        timestamp = pd.to_datetime(sliced_df['timestamp'].iloc[-1], unit='ms')

        if bak_cumulative_pnl <= STOP_LOSS_LIMIT:
            if isLogShow:
                logging.info(f"\n🛑 누적 손실 {bak_cumulative_pnl:.2f}%로 자동 종료")
            break

        # 변동성 필터
        volatility = analyze_volatility(sliced_df)
        if volatility >= VOLATILITY_THRESHOLD:
            bak_volatility_blocked = True
            continue
        else:
            bak_volatility_blocked = False

        # 진입 Feature 준비
        trend, confidence = predict_trend_sync(sliced_df, model_path=trend_model_path)
        if trend == 1 or confidence < 0.6:
            continue

        entry_features_df = generate_entry_strategy_dataset(sliced_df, trend_model_path=trend_model_path)
        if entry_features_df.empty:
            continue

        entry_row = entry_features_df.iloc[-1]
        strategy = predict_entry_strategy_from_row(entry_row, model_path=entry_model_path)

        def trend_to_signal(t): return 'long' if t == 2 else 'short' if t == 0 else None
        def reverse_signal(s): return 'short' if s == 'long' else 'long'

        signal = trend_to_signal(trend) if strategy == 1 else reverse_signal(trend_to_signal(trend))
        if signal is None:
            continue

        # TP/SL 설정
        if confidence >= 0.8:
            BAK_TP_PERCENT, BAK_SL_PERCENT = 1.8, 0.3
        elif confidence >= 0.6:
            BAK_TP_PERCENT, BAK_SL_PERCENT = 1.0, 0.5
        else:
            BAK_TP_PERCENT, BAK_SL_PERCENT = 0.7, 0.5

        # 포지션 종료 조건
        if bak_position_state and bak_entry_price:
            change_pct = (current_price - bak_entry_price) / bak_entry_price * 100
            if bak_position_state == 'short':
                change_pct *= -1

            hit_tp = change_pct >= BAK_TP_PERCENT
            hit_sl = change_pct <= -BAK_SL_PERCENT

            # 추세나 전략 변경으로 인한 종료 판단
            new_trend, _ = predict_trend_sync(sliced_df, model_path=trend_model_path)
            new_strategy = predict_entry_strategy_from_row(entry_row, model_path=entry_model_path)

            exit_by_trend = (new_trend != (2 if bak_position_state == 'long' else 0))
            exit_by_strategy = (new_strategy != bak_strategy_used_at_entry)

            if hit_tp or hit_sl or exit_by_trend or exit_by_strategy:
                label = (
                    "🎯 TP" if hit_tp else
                    "⚠️ SL" if hit_sl else
                    "🔁 전략 변경" if exit_by_strategy else
                    "📉 추세 변경"
                )
                bak_cumulative_pnl += change_pct
                if isLogShow:
                    logging.info(
                        f"{label} → {bak_position_state.upper()} 종료 | "
                        f"PnL: {change_pct:.2f}%, 누적: {bak_cumulative_pnl:.2f}%"
                    )
                bak_position_state = None
                bak_entry_price = None
                bak_strategy_used_at_entry = None
                continue

        # 진입 조건
        if not bak_volatility_blocked and bak_position_state is None:
            bak_position_state = signal
            bak_entry_price = current_price
            bak_strategy_used_at_entry = strategy
            if isLogShow:
                logging.info(
                    f"\n{timestamp} | 추세: {trend} / 전략: {'추세' if strategy == 1 else '역추세'} / "
                    f"방향: {signal.upper()} / 신뢰도: {confidence:.2f}"
                )
                logging.info(
                    f"진입 @ {bak_entry_price:.2f} | TP: {BAK_TP_PERCENT}%, SL: {BAK_SL_PERCENT}%"
                )
            continue

    if isLogShow:
        logging.info(f"\n백테스트 종료 → 최종 누적 PnL: {bak_cumulative_pnl:.2f}%\n")

    return bak_cumulative_pnl

async def test_backtest_bot(interval='15m', isLogShow=True) -> float:
    import joblib
    global bak_position_state, bak_entry_price, bak_volatility_blocked
    global BAK_TP_PERCENT, BAK_SL_PERCENT

    bak_cumulative_pnl = 0.0
    bak_strategy_used_at_entry = None
    bak_tp_order_id = None
    bak_sl_order_id = None

    df = get_klines(symbol='ETHUSDT', interval=interval, limit=1000)
    trend_model_path = f"trend_model_xgb_{interval}.pkl"
    entry_model_path = f"entry_strategy_model_{interval}.pkl"
    entry_model = joblib.load(entry_model_path)

    for i in range(100, len(df)):
        sliced_df = df.iloc[:i].copy()
        current_price = sliced_df['close'].iloc[-1]
        timestamp = pd.to_datetime(sliced_df['timestamp'].iloc[-1], unit='ms')

        if bak_cumulative_pnl <= STOP_LOSS_LIMIT:
            if isLogShow:
                logging.info(f"\n🛑 누적 손실 {bak_cumulative_pnl:.2f}%로 자동 종료")
            break

        # 변동성 필터
        volatility = analyze_volatility(sliced_df)
        if volatility >= VOLATILITY_THRESHOLD:
            if not bak_volatility_blocked:
                bak_volatility_blocked = True
            continue
        elif volatility < VOLATILITY_THRESHOLD and bak_volatility_blocked:
            bak_volatility_blocked = False

        support, resistance = calculate_support_resistance(sliced_df)

        trend, confidence = predict_trend_sync(sliced_df, model_path=trend_model_path)
        if trend == 1 or confidence < 0.6:
            continue

        entry_features_df = generate_entry_strategy_dataset(sliced_df, trend_model_path=trend_model_path)
        if entry_features_df.empty:
            continue

        entry_row = entry_features_df.iloc[-1]
        strategy = predict_entry_strategy_from_row(entry_row, model_path=entry_model_path)

        def trend_to_signal(t): return 'long' if t == 2 else 'short' if t == 0 else None
        def reverse_signal(s): return 'short' if s == 'long' else 'long'

        signal = trend_to_signal(trend) if strategy == 1 else reverse_signal(trend_to_signal(trend))
        if signal is None:
            continue

        # confidence 기반 TP/SL 설정
        if confidence >= 0.8:
            BAK_TP_PERCENT, BAK_SL_PERCENT = 1.8, 0.3
        elif confidence >= 0.6:
            BAK_TP_PERCENT, BAK_SL_PERCENT = 1.0, 0.5
        else:
            BAK_TP_PERCENT, BAK_SL_PERCENT = 0.7, 0.5

        # 포지션 종료 조건 체크
        if bak_position_state and bak_entry_price:
            change_pct = (current_price - bak_entry_price) / bak_entry_price * 100
            if bak_position_state == 'short':
                change_pct *= -1

            hit_tp = change_pct >= BAK_TP_PERCENT
            hit_sl = change_pct <= -BAK_SL_PERCENT

            new_trend, new_confidence = predict_trend_sync(sliced_df, model_path=trend_model_path)
            new_entry_features_df = generate_entry_strategy_dataset(sliced_df, trend_model_path=trend_model_path)
            if new_entry_features_df.empty:
                new_strategy = bak_strategy_used_at_entry
            else:
                new_entry_row = new_entry_features_df.iloc[-1]
                new_strategy = predict_entry_strategy_from_row(new_entry_row, model_path=entry_model_path)

            expected_trend = 2 if bak_position_state == 'long' else 0
            expected_strategy = bak_strategy_used_at_entry

            exit_by_trend = (new_trend != expected_trend and new_trend != 1 and new_confidence < 0.6)
            exit_by_strategy = (new_strategy != expected_strategy)

            if hit_tp or hit_sl or exit_by_trend or exit_by_strategy:
                label = (
                    "🎯 TP" if hit_tp else
                    "⚠️ SL" if hit_sl else
                    "🔁 전략 변경" if exit_by_strategy else
                    "📉 추세 변경"
                )
                bak_cumulative_pnl += change_pct
                if isLogShow:
                    logging.info(
                        f"{label} → {bak_position_state.upper()} 종료 | "
                        f"PnL: {change_pct:.2f}%, 누적: {bak_cumulative_pnl:.2f}%"
                    )
                bak_position_state = None
                bak_entry_price = None
                bak_strategy_used_at_entry = None
                continue

        # 진입 조건 체크
        if not bak_volatility_blocked and bak_position_state is None:
            bak_position_state = signal
            bak_entry_price = current_price
            bak_strategy_used_at_entry = strategy
            if isLogShow:
                logging.info(
                    f"[{interval}] {timestamp} | 추세: {trend} / 전략: {'추세' if strategy == 1 else '역추세'} / "
                    f"방향: {signal.upper()} / 신뢰도: {confidence:.2f}"
                )
                logging.info(
                    f"진입 @ {bak_entry_price:.2f} | TP: {BAK_TP_PERCENT}%, SL: {BAK_SL_PERCENT}%"
                )
            continue

    if isLogShow:
        logging.info(f"\n백테스트 종료 → 최종 누적 PnL: {bak_cumulative_pnl:.2f}%\n")

    return bak_cumulative_pnl

if __name__ == "__main__":
    mode = input("실행 모드 선택 (live / backtest / all_backtest): ").strip()
    if mode == "live":
        asyncio.run(start_bot())
    elif mode == "backtest":
        asyncio.run(backtest_bot(interval=TRADING_INTERVAL))
    elif mode == "all_backtest":
        asyncio.run(run_all_backtests())