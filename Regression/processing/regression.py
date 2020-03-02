import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, ARDRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ElasticNet
import processing.stock_utils as stock_utils

no_of_days_to_predict = 30


class Predictions_Manager:
    _predictions_models = []
    _predictions_array = []
    _predictions_conf = []

    def __init__(self):
        self._predictions_models = []
        self._predictions_array = []
        self._predictions_conf = []

    def add_prediction(self, values, model_name, confidence):
        self._predictions_models.append(model_name)
        self._predictions_array.append(values)
        self._predictions_conf.append(confidence)

    def get_average_prediction(self):
        data = np.array(self._predictions_array)
        data = np.average(data, axis=0)
        return data

    def plot_all_predictions(self, days, history=None, include_average=True, include_present_day=True):
        colors = ['red', 'blue', 'green', 'pink', 'purple', 'orange', 'black']

        caption_text = ""
        for i in range(len(self._predictions_models)):
            if history is not None:
                prediction_with_history = list(history)
                prediction_with_history.extend(self._predictions_array[i])
                plt.plot(prediction_with_history, colors[i % len(colors)])
            else:
                plt.plot(self._predictions_array[i], colors[i % len(colors)])
            caption_text += (self._predictions_models[i] + " - " + colors[i % len(self._predictions_models)] + "\n")

        if include_average:
            if history is not None:
                prediction_with_history = list(history)
                prediction_with_history.extend(self.get_average_prediction())
                plt.plot(prediction_with_history, 'black')
            else:
                plt.plot(self.get_average_prediction(), 'black')
            caption_text += ('Average - black\n')
        if include_present_day:
            plt.axvline(x=days - 1, ymin=0, ymax=2500)
        stock_utils.header("Plot legend")
        print(caption_text)
        plt.xlabel('Next days')
        plt.ylabel('USD')
        plt.show()


def predict_stock_with_multiple_regressors(df, days):
    pred_manager = Predictions_Manager()
    df = df.drop(['timestamp'], 1)
    df['Prediction'] = df[['close']].shift(-days)
    X = np.array(df.drop(['Prediction', 'close'], 1))
    X = X[:-days]

    y = np.array(df['Prediction'])
    y = y[:-days]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)

    stock_history_for_prediction = np.flip(np.array(df.drop(['Prediction', 'close'], 1))[:days], 0)

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    lin_reg_conf = lin_reg.score(X_test, y_test)
    lr_prediction = lin_reg.predict(stock_history_for_prediction)
    pred_manager.add_prediction(lr_prediction, "Linear regression", lin_reg_conf)

    br = BayesianRidge()
    br.fit(X_train, y_train)
    br_conf = br.score(X_test, y_test)
    br_prediction = br.predict(stock_history_for_prediction)
    pred_manager.add_prediction(br_prediction, "Bayesian Ridge", br_conf)

    en = ElasticNet()
    en.fit(X_train, y_train)
    en_conf = en.score(X_test, y_test)
    en_prediction = en.predict(stock_history_for_prediction)
    pred_manager.add_prediction(en_prediction, "Elastic Net", en_conf)

    ard = ARDRegression()
    ard.fit(X_train, y_train)
    ard_conf = ard.score(X_test, y_test)
    ard_prediction = ard.predict(stock_history_for_prediction)
    pred_manager.add_prediction(ard_prediction, "ARD Regression", ard_conf)

    stock_history_to_plot = np.flip(pd.to_numeric(df['close'][:days], errors='coerce'), 0)

    # pred_manager.plot_all_predictions(days=days,history=stock_history_to_plot)
    return pred_manager.get_average_prediction(), stock_history_to_plot


if __name__ == "__main__":
    url = stock_utils.build_url_with_symbol('AMZN')
    stock_json = stock_utils.get_json_from_url(url)
    df = stock_utils.process_json_to_pd(stock_json)
    df = df[:1000]

    avg_pred, stock_history = predict_stock_with_multiple_regressors(df, no_of_days_to_predict)
