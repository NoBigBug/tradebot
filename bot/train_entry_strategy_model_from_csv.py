import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
from config import TRADING_INTERVAL

def train_entry_strategy_from_csv(
    csv_path=f'entry_strategy_dataset_{TRADING_INTERVAL}.csv',
    interval=TRADING_INTERVAL,
    test_size=0.2,
    random_state=42
):
    print(f"CSV 로딩 중: {csv_path}")
    df = pd.read_csv(csv_path)

    if df.empty:
        print("CSV 파일에 데이터가 없습니다.")
        return

    print(f"총 샘플 수: {len(df)}")

    # 라벨 확인
    if 'label' not in df.columns:
        print("'label' 컬럼이 존재하지 않습니다.")
        return

    # 레이블 분포 체크
    label_counts = df['label'].value_counts()
    if label_counts.min() < 2:
        print(f"레이블 불균형 (0: {label_counts.get(0,0)}개, 1: {label_counts.get(1,0)}개) → 학습 스킵")
        return
    
    # 사용할 피처 명시적으로 정의 (감정 포함)
    feature_cols = [
        'ma_ratio', 'volatility', 'rsi', 'macd', 'macd_signal',
        'bb_width', 'ema_ratio_9_21', 'adx', 'atr', 'stoch_k',
        'dist_support', 'dist_resistance', 'trend', 'confidence',
        'sentiment_score'
    ]

    # 누락된 피처 확인
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ 다음 필수 컬럼이 누락됨: {missing_cols}")
        return

    X = df.drop(columns=['label'])
    y = df['label']

    # 레이블 분포 확인
    print(f"레이블 분포:\n{y.value_counts()}\n")

    if len(y.unique()) < 2:
        print("⚠️ 레이블이 하나의 클래스만 포함되어 있어 학습을 건너뜁니다.")
        return

    # 학습/검증 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"학습 데이터: {len(X_train)}개 | 테스트 데이터: {len(X_test)}개")

    # 불균형 데이터 대응용 샘플 가중치 계산
    sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)

    # 모델 정의
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=random_state
    )

    # 학습
    model.fit(X_train, y_train, sample_weight=sample_weight)

    # 검증 예측 및 리포트
    y_pred = model.predict(X_test)
    print("\n 검증 결과:")
    print(classification_report(y_test, y_pred, digits=4))

    # 교차검증
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"\n 교차검증 평균 정확도: {scores.mean():.4f} / 표준편차: {scores.std():.4f}")

    # 모델 저장
    model_path = f"entry_strategy_model_{interval}.pkl"
    joblib.dump(model, model_path)
    print(f"\n 모델 저장 완료 → {model_path}")

if __name__ == "__main__":
    intervals = ['15m', '1h']

    for interval in intervals:
        print(f"\n==============================")
        print(f"[{interval}] 진입 전략 모델 학습 시작")
        print(f"==============================\n")
        train_entry_strategy_from_csv(csv_path=f'entry_strategy_dataset_{interval}.csv', interval=interval)