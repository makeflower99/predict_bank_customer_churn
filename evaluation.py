def mm_sclaer(train, test):
    features = ['CreditScore', 'Balance', 'EstimatedSalary']
    mm = mm = MinMaxScaler()
    data_scale = train[features]
    test_scale = test[features]
    train[features] = mm.fit_transform(data_scale)
    test[features] = mm.transform(test_scale)

    return train, test
def evaluation(model, X_train, X_test, y_train, y_test, test):
    pred = model.predict_proba(X_test)[:, 1]
    pred_binary = model.preidct(X_test)

    # ROC-AUC 점수로 모델 평가
    fpr, tpr, _ = roc_curve(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred)
    f1 = f1_score(pred_binary, y_test)

    print(f'ROC-AUC Score : {roc_auc}')
    print(f'F1 score : {f1}')
    return pred, pred_binary, fpr, tpr, roc_auc

def sub_result(model, test, submission):
    submission["Exited"] = prediction
    submission.to_csv("bank_churn_clf.csv")
    return submission

