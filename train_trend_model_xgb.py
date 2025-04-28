import pandas as pd
import joblib
import xgboost as xgb
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange
from ta.momentum import StochasticOscillator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from config import TRADING_INTERVAL # 사용 중인 인터벌(예: '15m') 불러오기

# 기술 지표 기반 피처 생성
def compute_features(df: pd.DataFrame):
    df = df.copy()
    df['return'] = df['close'].pct_change()
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ma_ratio'] = df['ma5'] / df['ma10']
    df['volatility'] = df['return'].rolling(window=5).std()
    df['rsi'] = compute_rsi(df['close'], 14)

    # MACD 및 시그널
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()

    # Bollinger Band Width
    ma20 = df['close'].rolling(window=20).mean()
    std20 = df['close'].rolling(window=20).std()
    df['bb_width'] = (2 * std20) / ma20

    # EMA 간격 비율
    df['ema9'] = df['close'].ewm(span=9).mean()
    df['ema21'] = df['close'].ewm(span=21).mean()
    df['ema_ratio_9_21'] = df['ema9'] / df['ema21']

    # ADX (추세 강도 지표)
    adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['adx'] = adx.adx()

    # ATR (평균 진폭 범위, 변동성 지표)
    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['atr'] = atr.average_true_range()

    # Stochastic RSI (%K)
    stoch = StochasticOscillator(close=df['close'], high=df['high'], low=df['low'], window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()

    return df.dropna()

# RSI 계산 함수
def compute_rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# 추세 라벨링 함수
def label_trend(df: pd.DataFrame, future_window=10, threshold=0.8):
    df = df.copy()

    # 미래 수익률 계산 (10봉 후 기준)
    df['future_return'] = df['close'].pct_change(periods=future_window).shift(-future_window)

    # 라벨 부여 (상승: 2 / 횡보: 1 / 하락: 0)
    df['trend'] = df['future_return'].apply(
        lambda x: 2 if x > threshold / 100 else (0 if x < -threshold / 100 else 1)
    )
    return df.dropna()

# 스마트 횡보 판별 기반 트렌드 라벨링
def label_trend_smart(df: pd.DataFrame, future_window=10, threshold=0.8) -> pd.DataFrame:
    df = df.copy()

    # 미래 수익률 계산
    df['future_return'] = df['close'].pct_change(periods=future_window).shift(-future_window)

    # 기본 상승/하락/횡보 구분
    df['basic_trend'] = df['future_return'].apply(
        lambda x: 2 if x > threshold / 100 else (0 if x < -threshold / 100 else 1)
    )

    # 추가적인 스마트 횡보 판별
    # 볼린저 밴드 폭
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['std20'] = df['close'].rolling(window=20).std()
    df['bb_width'] = (2 * df['std20']) / df['ma20']

    # 단기/장기 이평 간 거리
    df['ema9'] = df['close'].ewm(span=9).mean()
    df['ema21'] = df['close'].ewm(span=21).mean()
    df['ema_distance'] = abs(df['ema9'] - df['ema21']) / df['close']

    # ADX 추세 강도
    adx_indicator = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['adx'] = adx_indicator.adx()

    # 스마트 횡보 조건
    volatility_condition = df['bb_width'] < 0.01    # 볼린저 밴드 폭 1% 이내
    ema_condition = df['ema_distance'] < 0.003       # 이평선 간 거리 0.3% 이내
    adx_condition = df['adx'] < 20                   # ADX 20 이하 (추세 약함)

    smart_consolidation = volatility_condition & ema_condition & adx_condition

    # 최종 트렌드 결정
    df['trend'] = df.apply(
        lambda row: 1 if (row['basic_trend'] == 1 or smart_consolidation.loc[row.name]) else row['basic_trend'],
        axis=1
    )

    return df.dropna()

# 모델 학습 함수
def train_model(interval='15m'):
    # 학습용 CSV 파일 로드
    df = pd.read_csv(f"trend_training_data_{interval}.csv")

    # 피처 및 라벨 생성
    df = compute_features(df)
    df = label_trend_smart(df)

    # 사용 피처 정의
    features = [
        'ma_ratio', 'volatility', 'rsi', 'macd', 'macd_signal', 'bb_width',
        'ema_ratio_9_21', 'adx', 'atr', 'stoch_k'
    ]

    X = df[features]
    y = df['trend']  # 0=하락, 1=횡보, 2=상승

    # 학습/검증 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 다중 클래스 XGBoost 모델 정의
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        verbosity=0,
        objective='multi:softprob',  # 다중 클래스 확률 출력
        num_class=3,
        tree_method='hist',
        scale_pos_weight=1,
    )

    # 모델 학습
    model.fit(X_train, y_train)

    # 평가 결과 출력
    y_pred = model.predict(X_test)
    print("\n[모델 평가 결과]")
    print(classification_report(y_test, y_pred, digits=3))

    # 모델 저장
    joblib.dump(model, f"trend_model_xgb_{interval}.pkl")
    print(f"\n모델 저장 완료 → trend_model_xgb_{interval}.pkl")

# Binance에서 데이터 받아서 학습 실행
if __name__ == '__main__':
    intervals = ['15m', '1h']

    for interval in intervals:
        train_model(interval)