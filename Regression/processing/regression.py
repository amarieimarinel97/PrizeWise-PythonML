import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression, ARDRegression
from sklearn.model_selection import train_test_split

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
        return data.tolist()

    def plot_all_predictions(self, days, history=None, include_average=True, include_present_day=True,
                             median_line_points=None):
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
            average_prediction = self.get_average_prediction()
            if history is not None:
                prediction_with_history = list(history)
                prediction_with_history.extend(average_prediction)
                plt.plot(prediction_with_history, 'black')
            else:
                plt.plot(average_prediction, 'black')

            caption_text += 'Average - black\n'
            if median_line_points is not None:
                x_values = [median_line_points[0][0], median_line_points[1][0]]
                y_values = [median_line_points[0][1], median_line_points[1][1]]
                # print(x_values, y_values)
                plt.plot(x_values, y_values)

        if include_present_day:
            plt.axvline(x=days - 1, ymin=0, ymax=2500)

        bottom_lim, top_lim = plt.ylim()
        middle = (bottom_lim + top_lim) / 2
        bottom_lim -= 1.5 * (middle - bottom_lim)
        top_lim += 1.5 * (top_lim - middle)
        plt.ylim(bottom_lim, top_lim)

        plt.xlabel('Next days')
        plt.ylabel('USD')
        # plt.show()


def predict_stock_with_multiple_regressors(df, days):
    global no_of_days_to_predict
    no_of_days_to_predict = days
    pred_manager = Predictions_Manager()
    df = df.drop(['timestamp'], 1)
    df['Prediction'] = df[['close']].shift(-days)
    X = np.array(df.drop(['Prediction', 'close'], 1))
    X = X[:-days]

    y = np.array(df['Prediction'])
    y = y[:-days]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)

    stock_history_for_prediction = np.array(df.drop(['Prediction', 'close'], 1))[:days * 10]

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    lin_reg_conf = lin_reg.score(X_test, y_test)
    lr_prediction = lin_reg.predict(stock_history_for_prediction[:days])
    pred_manager.add_prediction(lr_prediction, "Linear regression", lin_reg_conf)

    # br = BayesianRidge()
    # br.fit(X_train, y_train)
    # br_conf = br.score(X_test, y_test)
    # br_prediction = br.predict(stock_history_for_prediction[:days])
    # pred_manager.add_prediction(br_prediction, "Bayesian Ridge", br_conf)
    #
    # ard = ARDRegression()
    # ard.fit(X_train, y_train)
    # ard_conf = ard.score(X_test, y_test)
    # ard_prediction = ard.predict(stock_history_for_prediction[:days])
    # pred_manager.add_prediction(ard_prediction, "ARD Regression", ard_conf)

    stock_history_to_plot = np.flip(pd.to_numeric(df['close'][:days*3], errors='coerce'), 0)

    average_prediction = pred_manager.get_average_prediction()

    # stock_with_history = list(stock_history_to_plot)
    # stock_with_history.extend(average_prediction)
    # median_line = get_median_line(stock_with_history)
    # pred_manager.plot_all_predictions(days=days,
    #                                   history=stock_history_to_plot,
    #                                   median_line_points=median_line
    #                                   )

    return average_prediction, stock_history_to_plot


def get_vertical_projection_of_point_on_line(p1, p2, point):
    yA = p1[1]
    yB = p2[1]
    xA = p1[0]
    xB = p2[0]
    xC = point[0]
    m = (yB - yA) / (xB - xA)
    yC = m * (xC - xA) + yA
    return yC


def compute_vertical_deviation(line, input_array):
    p1 = line[0]
    p2 = line[1]

    result = []
    for i in range(0, len(input_array)):
        projection = get_vertical_projection_of_point_on_line(p1, p2, [i, input_array[i]])
        absolute_deviation = input_array[i] - projection
        percent_deviation = absolute_deviation * 100 / projection
        result.append(percent_deviation)
    return result

def compute_percentage_changes(input_array):
    result = [0]
    for i in range(1, len(input_array)):
        result.append((input_array[i]-input_array[i-1])/input_array[i-1]*100)
    return result


def get_median_line(input_array):
    no_of_elem = len(input_array)

    start_point = [0, input_array[0]]
    end_point = [len(input_array) - 1, input_array[len(input_array) - 1]]
    deviation = compute_vertical_deviation([start_point, end_point], input_array)
    current_sign_of_elements = sum(deviation)

    if current_sign_of_elements < 0:
        while current_sign_of_elements < 0:
            start_point[1] -= 0.01 * start_point[1]
            end_point[1] -= 0.01 * end_point[1]
            deviation = compute_vertical_deviation([start_point, end_point], input_array)
            current_sign_of_elements = sum(deviation)

    else:
        while current_sign_of_elements > 0:
            start_point[1] += 0.01 * start_point[1]
            end_point[1] += 0.01 * end_point[1]
            deviation = compute_vertical_deviation([start_point, end_point], input_array)
            current_sign_of_elements = sum(deviation)
    return [start_point, end_point]


def get_start_and_end_point(input_array):
    start_point = [0, input_array[0]]
    end_point = [len(input_array) - 1, input_array[len(input_array) - 1]]
    return [start_point, end_point]


if __name__ == "__main__":
    url = stock_utils.build_url_with_symbol('AMZN')
    stock_json = stock_utils.get_json_from_url(url)
    df = stock_utils.process_json_to_pd_with_limit(stock_json, 1000)

    avg_pred, stock_history = predict_stock_with_multiple_regressors(df, no_of_days_to_predict)
    pred_with_present_day = [stock_history[0]] + avg_pred
    print(pred_with_present_day)
    print(compute_vertical_deviation(get_start_and_end_point(pred_with_present_day), pred_with_present_day))
    # deviation = compute_vertical_deviation(get_median_line(stock_with_history), stock_with_history)
    # print(deviation[len(deviation)//2:])

    plt.show()
