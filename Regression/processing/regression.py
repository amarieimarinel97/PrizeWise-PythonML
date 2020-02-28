import quandl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import processing.stock_utils as stock_utils
import matplotlib.pyplot as plt


no_of_days_to_predict = 30


def predict_stock(df, days):
    df = df[['open']]
    df['Prediction'] = df[['open']].shift(-days)
    data_set1 = np.array(df.drop(['Prediction'], 1))
    data_set1 = data_set1[:-days]

    data_set2 = np.array(df['Prediction'])
    data_set2 = data_set2[:-days]

    set1_train, set1_test, set2_train, set2_test = train_test_split(
        data_set1, data_set2, test_size=0.3)

    my_svr = SVR(kernel='rbf', C=1e3, gamma='auto')
    my_svr.fit(set1_train, set2_train)
    my_svr_conf = my_svr.score(set1_test, set2_test)
    print("my_svr conf:", my_svr_conf)

    my_linreg = LinearRegression()
    my_linreg.fit(set1_train, set2_train)
    my_linreg_conf = my_linreg.score(set1_test, set2_test)
    print("my_linreg confg:", my_linreg_conf)

    stock_prediction = np.array(df.drop(['Prediction'], 1))[:days]
    print(stock_prediction)

    linreg_prediction = my_linreg.predict(stock_prediction)
    print(linreg_prediction)

    svr_prediction = my_svr.predict(stock_prediction)
    print(svr_prediction)

    return (linreg_prediction, svr_prediction, stock_prediction)


if __name__ == "__main__":
    url = stock_utils.build_url_with_symbol('AMZN&outputsize=full')
    stock_json = stock_utils.get_json_from_url(url)
    df = stock_utils.process_json_to_pd(stock_json)
    df = df[:2000]
    lr_pred, svr_pred, last_pred = predict_stock(df, no_of_days_to_predict)
    stock_pred = list(last_pred)
    stock_pred.extend(lr_pred)
    # plt.plot(stock_pred)

    plt.plot(lr_pred)
    plt.ylabel('USD')
    plt.xlabel('Next days')
    plt.axvline(x=30, ymin=0, ymax=2000)
    plt.show()
