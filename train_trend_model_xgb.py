import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

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
    df = extract_features(df)
    df = label_trend(df)

    features = ['ma_ratio', 'volatility', 'rsi']
    X = df[features]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("📊 Classification Report (XGBoost):")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, model_path)
    print(f"✅ XGBoost 모델 저장 완료: {model_path}")

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

    df = get_klines()
    train_trend_model(df)
