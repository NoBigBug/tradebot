import pandas as pd
from binance.client import Client
from config import BINANCE_API_KEY, BINANCE_API_SECRET
from new_tradeBot import (
    generate_entry_strategy_dataset
)

# Binance API client
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

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

def get_klines(symbol='BTCUSDT', interval='5m', limit=1000):
    klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'num_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    return df

def save_entry_strategy_dataset_csv(
    symbol='BTCUSDT',
    interval='5m',
    trend_model_path=None,
    output_csv=None
):
    if trend_model_path is None:
        trend_model_path = f"trend_model_xgb_{interval}.pkl"
    if output_csv is None:
        output_csv = f"entry_strategy_dataset_{interval}.csv"

    print(f"📦 {symbol} {interval} 데이터 가져오는 중...")

    limit = get_auto_limit(interval=interval)
    df = get_klines(symbol=symbol, interval=interval, limit=limit)

    if df is None or df.empty:
        print("❌ 데이터 로딩 실패")
        return

    print("🧠 학습용 데이터셋 생성 중...")
    dataset = generate_entry_strategy_dataset(df, trend_model_path=trend_model_path)

    if dataset.empty:
        print("⚠️ 유효한 진입 포인트가 없어 저장되지 않았습니다.")
        return

    dataset.to_csv(output_csv, index=False)
    print(f"✅ 저장 완료 → {output_csv} (총 {len(dataset)}개 샘플)")

if __name__ == "__main__":
    intervals = ['5m', '15m', '1h']
    for interval in intervals:
        print(f"\n==============================")
        print(f"📁 [{interval}] CSV 생성 시작")
        print(f"==============================\n")
        save_entry_strategy_dataset_csv(interval=interval)