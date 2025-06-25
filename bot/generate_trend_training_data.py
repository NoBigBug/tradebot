import pandas as pd
from binance.client import Client
from config import BINANCE_API_KEY, BINANCE_API_SECRET, TRADING_INTERVAL

# Binance 클라이언트 연결
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

# 데이터 수집 함수
def get_klines(symbol='BTCUSDT', interval=TRADING_INTERVAL, limit=1000):
    klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'num_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

# 저장 함수
def save_training_data(interval=TRADING_INTERVAL, symbol='BTCUSDT', limit=1000):
    df = get_klines(symbol=symbol, interval=interval, limit=limit)
    filename = f"trend_training_data_{interval}.csv"
    df.to_csv(filename, index=False)
    print(f"학습용 데이터 저장 완료 → {filename}")

if __name__ == "__main__":
    intervals = ['15m', '1h']

    for interval in intervals:
        save_training_data(interval=interval, symbol='BTCUSDT', limit=1000)