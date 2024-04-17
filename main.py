import eda_analysis
import model
from preprocessing import *

if __name__ == '__main__':
    # 전처리 데이터 불러옴
    train_X, train_y, test_X = preprocess()

    # 아래모델 적용 해주세요!
    X_train, y_train, X_test, y_test = split(train_X, train_y)