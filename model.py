from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from enum import IntEnum

'''
데이터셋 분할하는 함수
'''
def split(train_data, label):
    X_train, X_test, y_train, y_test = train_test_split(train_data, label, test_size=0.3, shuffle=True, random_state=42)

    print(f'X_train shape : {X_train.shape}, y_train shape : {y_train.shape}')
    print(f'X_test shape : {X_test.shape}, y_test shape : {y_test.shape}')

    return X_train, y_train, X_test, y_test


'''
고정값 -> 열거형 Enum 처리
'''

class RfEnum(IntEnum):
    RANDOM_STATE = 42
    N_JOBS = -1 # 컴퓨터 모든 코어를 사용함.
    VERBOSE = 2 # 상세 + 훈련수 체크

'''
best_fit : 최적의 값을 찾아서 훈련시켜주는 함수임.
'''
def best_fit(param_grid, cv, scoring, X_train, y_train):
    rf = RandomForestClassifier(random_state=RfEnum.RANDOM_STATE, n_jobs=RfEnum.N_JOBS)
    grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring=scoring, refit='roc_auc_score', verbose=RfEnum.VERBOSE)
    grid_search.fit(X_train, y_train)
    print(f'Best parameters : {grid_search.best_params_}')  # 최적 파라미터 출력

    rfModel = RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'],
                                     max_depth=grid_search.best_params_['max_depth'],
                                     min_samples_split=grid_search.best_params_['min_samples_split'],
                                     min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
                                     random_state=RfEnum.RANDOM_STATE,
                                     n_jobs=RfEnum.N_JOBS, verbose=RfEnum.VERBOSE)
    rfModel.fit(X_train, y_train)
    return rfModel

'''
eval : 평가 함수하고 pred, pred_binary, fpr, tpr, roc_auc 리턴함
'''
def eval(model, X_test, y_test):
    pred = model.predict_proba(X_test)[:, 1]
    pred_binary = model.predict(X_test)

    # ROC-AUC 점수로 모델 평가
    fpr, tpr, _ = roc_curve(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred)
    f1 = f1_score(pred_binary, y_test)

    print(f'ROC-AUC Score : {roc_auc}')
    print(f'F1 score : {f1}')
    return pred, pred_binary, fpr, tpr, roc_auc


'''
eval_visualize : 평가를 시각화해주는 함수
'''
def eval_visualize(X_test, y_test, pred, fpr, tpr, roc_auc, model):
    threshold = 0.5
    pred_binary = (pred > threshold).astype(int)

    conf_matrix = confusion_matrix(y_test, pred_binary)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 14})
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Tuned ROC Curve')
    plt.legend()
    plt.show()

    # 시리즈로 만들어 인덱스를 붙인다
    fi = pd.Series(model.feature_importances_, index=X_test.columns)

    # 내림차순 정렬을 이용한다
    desc_cols = fi.sort_values(ascending=False)
    print(desc_cols)

    plt.figure(figsize=(8, 6))
    plt.title('Feature Importances')
    sns.barplot(x=desc_cols, y=desc_cols.index)
    plt.show()


