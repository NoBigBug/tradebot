import pandas as pd
from binance.client import Client
from config import BINANCE_API_KEY, BINANCE_API_SECRET
from new_tradeBot import (
    get_klines,
    generate_entry_strategy_dataset
)

def save_entry_strategy_dataset_csv(
    symbol='BTCUSDT',
    interval='5m',
    trend_model_path='trend_model_xgb_5m.pkl',
    output_csv='entry_strategy_dataset.csv'
):
    # Binance API client
    client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

    print(f"📦 {symbol} {interval} 데이터 가져오는 중...")
    df = get_klines(symbol=symbol, interval=interval, limit=1500)

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
    save_entry_strategy_dataset_csv()