import pandas as pd
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()


def preprocess ():
    # 파일 로드
    data = pd.read_csv('./dataset/train.csv')
    test = pd.read_csv('./dataset/test.csv')
    # id 열 제거
    data = data.iloc[:,1:]
    ids = test['id'] # 최종 제출용 id 
    test = test.iloc[:,1:]
    # 나이 파생변수
    data["Status"] = pd.cut(data["Age"], bins = [0, 22, 55, data["Age"].max()+1],
                            labels= ["Student","Employee","Retired"])
    test["Status"] = pd.cut(test["Age"], bins = [0, 22, 55, test["Age"].max()+1],
                            labels= ["Student","Employee","Retired"])
    # 라벨 인코딩
    data['Gender'] = le.fit_transform(data['Gender'])

    #One hot Encoding
    geo_ohe = onehot('Geography', data, test)
    stat_ohe = onehot('Status', data, test)

    data = pd.concat([data, geo_ohe, stat_ohe], axis=1)
    data.drop(['Geography', 'Status','CustomerId'], axis=1, inplace=True)
    test.drop(['Geography', 'Status','CustomerId'], axis=1, inplace=True)
    # 성 분류 변수
    frequency_surname = data.Surname.value_counts() 
    sorted_surnames = frequency_surname.index
    frequency_encoding = {surname: i + 1 for i, surname in enumerate(sorted_surnames)}
    frequency_surname_test = test.Surname.value_counts() 
    sorted_surnames_test = frequency_surname_test.index
    frequency_encoding_test = {surname: i + 1 for i, surname in enumerate(sorted_surnames_test)}
    data['Surname'] = data['Surname'].map(frequency_encoding)
    test['Surname'] = test['Surname'].map(frequency_encoding_test)
    # 중복제거
    data.duplicated().sum()
    data.drop_duplicates(inplace=True)
    # 스케일링
    num_col = ['CreditScore', 'Balance', 'EstimatedSalary','Surname']
    df_scale = data[num_col]
    data[num_col] = mm.fit_transform(df_scale)
    df_scale_test = test[num_col]
    test[num_col] = mm.transform(df_scale_test)

    train_X = data.drop("Exited", axis=1)
    train_y = data.Exited
    test_X = test

    print(f'Train X shape : {train_X.shape}')
    print(f'train_y shape : {train_y .shape}')
    print(f'test_X  shape : {test_X.shape}')
    
    return train_X, train_y, test_X

# 원-핫 인코딩
def onehot(col, data, test):
    encoder.fit(data[[col]])
    onehot = encoder.transform(data[[col]]).toarray()  # 데이터 fit & 변환
    onehot = pd.DataFrame(onehot)
    onehot.columns = encoder.get_feature_names_out()  # array형태의 출력을 DF로 변환 & 컬럼명 지정

    test_trans = encoder.transform(test[[col]])
    test_ohe = pd.DataFrame(test_trans.toarray())
    test_ohe.columns = encoder.get_feature_names_out()
    test = pd.concat([test, test_ohe], axis=1)

    return onehot