import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_entry_strategy_from_csv(
    csv_path='entry_strategy_dataset.csv',
    model_path='entry_strategy_model.pkl',
    test_size=0.2,
    random_state=42
):
    print(f"📂 CSV 로딩 중: {csv_path}")
    df = pd.read_csv(csv_path)

    if df.empty:
        print("❌ CSV 파일에 데이터가 없습니다.")
        return

    print(f"📊 총 샘플 수: {len(df)}")

    # 입력 feature와 label 분리
    X = df.drop(columns=['label'])
    y = df['label']

    # 학습/검증 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"🧠 학습 데이터: {len(X_train)}개 | 테스트 데이터: {len(X_test)}개")

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=random_state
    )
    model.fit(X_train, y_train)

    # 검증 결과 출력
    y_pred = model.predict(X_test)
    print("\n📈 검증 결과:")
    print(classification_report(y_test, y_pred, digits=4))

    # 모델 저장
    joblib.dump(model, model_path)
    print(f"\n✅ 모델 저장 완료 → {model_path}")

if __name__ == "__main__":
    train_entry_strategy_from_csv()