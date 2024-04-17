from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score, \
    precision_score, recall_score, roc_auc_score, roc_curve, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

'''
데이터셋 분할하는 함수
'''
def split(train_data, label):
    X_train, y_train, X_test, y_test = train_test_split(train_data, label, test_size=0.3, shuffle=True, random_state=42)

    print(f'X_train shape : {X_train.shape}, y_train shape : {y_train.shape}')
    print(f'X_test shape : {X_test.shape}, y_test shape : {y_test.shape}')

    return X_train, y_train, X_test, y_test


'''
RandomForestClassifier Class 
1) __init__ : Rf 클래스 생성하면 기본적으로 random_state, n_jobs 값은 고정됨
2) best_fit : 최적의 값을 찾아서 훈련시켜주는 함수임.
3) eval : 평가 함수
'''
class Rf:
    def __init__(self):
        self.random_state= 42 # 랜덤값 고정
        self.n_jobs = -1 # 컴퓨터의 모든 코어 사용


    def best_fit(self, param_grid, cv, scoring, X_train, y_train):

        rf = RandomForestClassifier(random_state=self.random_state, n_jobs=self.n_jobs)
        grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring=scoring,refit='roc_auc_score')
        grid_search.fit(X_train, y_train)
        print(f'Best parameters : {grid_search.best_params_}') # 최적 파라미터 출력

        rfModel = RandomForestClassifier(n_estimators=grid_search['n_estimators'],
                                         max_depth=grid_search['max_depth'],
                                         min_samples_split=grid_search['min_samples_split'],
                                         min_samples_leaf=grid_search['min_samples_leaf'],
                                         random_state=self.random_state,
                                         n_jobs=self.n_jobs)
        rfModel.fit(X_train, y_train)
        return rfModel


    def eval(self, model, X_test, y_test):
        pred = model.predict_proba(X_test)[:, 1]
        pred_binary = model.preidct(X_test)

        # ROC-AUC 점수로 모델 평가
        fpr, tpr, _ = roc_curve(y_test, pred)
        roc_auc = roc_auc_score(y_test, pred)
        f1 = f1_score(pred_binary, y_test)

        print(f'ROC-AUC Score : {roc_auc}')
        print(f'F1 score : {f1}')
        return pred, fpr, tpr, roc_auc

    def eval_visualize(self, X_test, y_test, pred, fpr, tpr, roc_auc, model):
        threshold = 0.5
        pred_binary = (pred > threshold).astype(int)

        conf_matrix = confusion_matrix(y_test, pred)

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