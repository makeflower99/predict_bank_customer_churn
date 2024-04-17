from sklearn.preprocessing import MinMaxScaler


def mm_sclaer(train, test):
    features = ['CreditScore', 'Balance', 'EstimatedSalary']
    mm = mm = MinMaxScaler()
    data_scale = train[features]
    test_scale = test[features]
    train[features] = mm.fit_transform(data_scale)
    test[features] = mm.transform(test_scale)

    return train, test

def sub_result(model, test, submission):
    submission["Exited"] = prediction
    submission.to_csv("bank_churn_clf.csv")
    return submission

