import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import joblib
import os
import sys
from datetime import datetime

from telegram import Bot

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

bot = Bot(token=TELEGRAM_BOT_TOKEN)

sys.stdout.reconfigure(encoding='utf-8')

def compute_rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['return'] = df['close'].pct_change()
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ma_ratio'] = df['ma5'] / df['ma10']
    df['volatility'] = df['return'].rolling(window=5).std()
    df['rsi'] = compute_rsi(df['close'], 14)

    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()

    # Bollinger Band Width
    ma20 = df['close'].rolling(window=20).mean()
    std20 = df['close'].rolling(window=20).std()
    df['bb_width'] = (2 * std20) / ma20

    df = df.dropna()
    return df

def label_trend(df: pd.DataFrame, forward_window: int = 5, threshold: float = 0.002) -> pd.DataFrame:
    future_price = df['close'].shift(-forward_window)
    df['future_return'] = (future_price - df['close']) / df['close']

    def categorize_trend(r):
        if r > threshold:
            return 2   # 상승
        elif r < -threshold:
            return 0   # 하락
        else:
            return 1   # 횡보

    df['label'] = df['future_return'].apply(categorize_trend)
    df = df.dropna()
    return df

def train_trend_model(df: pd.DataFrame, model_path='trend_model_xgb.pkl'):
    print("🚀 모델 학습 시작...")
    df = extract_features(df)
    df = label_trend(df)

    features = ['ma_ratio', 'volatility', 'rsi', 'macd', 'macd_signal', 'bb_width']
    X = df[features]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    new_model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss'
    )
    new_model.fit(X_train, y_train)
    new_pred = new_model.predict(X_test)

    new_f1 = f1_score(y_test, new_pred, average='macro')
    print(f"📈 새 모델 F1-score (macro): {new_f1:.4f}")
    print("📊 새 모델 분류 리포트:")
    print(classification_report(y_test, new_pred))

    try:
        if os.path.exists(model_path):
            old_model = joblib.load(model_path)
            old_pred = old_model.predict(X_test)
            old_f1 = f1_score(y_test, old_pred, average='macro')
            print(f"📉 기존 모델 F1-score (macro): {old_f1:.4f}")
        else:
            raise FileNotFoundError
    except Exception as e:
        print(f"⚠️ 기존 모델 로딩 실패 또는 없음 → 새 모델 사용\n{e}")
        old_f1 = 0.0

    if new_f1 >= old_f1:
        joblib.dump(new_model, model_path)
        print(f"✅ 새 모델이 더 우수하여 교체 완료 → 저장됨: {model_path}")
        message = (
            "📈 [모델 업데이트 완료]\n\n"
            "🆕 새로운 모델이 기존보다 성능이 더 우수하여 교체되었습니다.\n\n"
            f"🔹 기존 F1 (macro): {old_f1:.4f}\n"
            f"🔹 새 모델 F1 (macro): {new_f1:.4f}\n"
            f"📁 {model_path}로 저장 완료"
        )
    else:
        print("❌ 새 모델의 성능이 낮아 교체하지 않음")
        message = (
            "📉 [모델 업데이트 스킵]\n\n"
            "❌ 새 모델의 성능이 기존보다 낮아 저장하지 않았습니다.\n\n"
            f"🔹 기존 F1 (macro): {old_f1:.4f}\n"
            f"🔹 새 모델 F1 (macro): {new_f1:.4f}\n"
            f"📁 기존 모델 유지됨"
        )
    
    send_telegram_message_sync(message)

# 텔레그램 메시지 전송용 (비동기 필요 없음)
async def send_telegram_message_sync(message: str):
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
    except Exception as e:
        print(f"❌ 텔레그램 전송 실패: {e}")

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

# Binance에서 데이터 받아서 학습 실행
if __name__ == '__main__':
    from binance.client import Client
    from config import BINANCE_API_KEY, BINANCE_API_SECRET

    client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

    def get_klines(symbol='BTCUSDT', interval='5m', limit=1000):
        klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'num_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        df['close'] = df['close'].astype(float)
        return df

    timeframes = ['1m', '5m', '15m', '1h']

    for tf in timeframes:
        print(f"\n⏳ {tf} 타임프레임 모델 학습 시작...")
        limit = get_auto_limit(tf)
        df = get_klines(interval=tf, limit=limit)
        model_path = f"trend_model_xgb_{tf}.pkl"
        train_trend_model(df, model_path=model_path)