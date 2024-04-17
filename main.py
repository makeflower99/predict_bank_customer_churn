import eda_analysis
from model import split, best_fit, eval, eval_visualize
from preprocessing import *
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

if __name__ == '__main__':
    # 전처리 데이터 불러옴
    train_X, train_y, test_X = preprocess()

    # 아래모델 적용 해주세요!
    X_train, y_train, X_test, y_test = split(train_X, train_y)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score),
               'recall': make_scorer(recall_score),
               'f1_score': make_scorer(f1_score),
               'roc_auc_score': make_scorer(roc_auc_score)
    }

    best_rf = best_fit(param_grid, 5, scoring, X_train, y_train)
    pred, pred_binary, fpr, tpr, roc_auc = eval(best_rf, X_test, y_test)
    eval_visualize(X_test, y_test, pred, fpr, tpr, roc_auc, best_rf)