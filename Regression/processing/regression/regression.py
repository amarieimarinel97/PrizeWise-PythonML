import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression, ARDRegression
from sklearn.model_selection import train_test_split

import processing.utils.stock_utils as stock_utils


class Regressor:
    no_of_days_to_predict = 3
    main_stocks = [
        "DIA", "SPY", "QQQ", "AMZN", "MSFT", "GOOG", "DAVA", "AAPL", "FB", "TSLA", "BABA", "NFLX", "DIS"
    ]
    main_stocks_models = {}

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

    def predict_stock(self, model_info, days):
        pred_manager = self.Predictions_Manager()
        df = model_info["df"]

        stock_history_for_prediction = np.array(df.drop(['prediction', 'close'], 1))[:days * 10]

        prediction = model_info["model"].predict(stock_history_for_prediction[:days])
        pred_manager.add_prediction(prediction, model_info["name"], model_info["confidence"])

        stock_history_to_plot = np.flip(pd.to_numeric(df['close'][:days * 3 + 1], errors='coerce'), 0)
        average_prediction = pred_manager.get_average_prediction()

        return average_prediction, stock_history_to_plot

    def model_fit_symbol(self, df, no_of_days=None, model_name=None):
        if no_of_days is None:
            no_of_days = self.no_of_days_to_predict

        model = None
        models_switcher = {
            "lr": LinearRegression(),  # linear regression
            "br": BayesianRidge(),  # bayesian ridge
            "ard": ARDRegression(),  # ard regression
        }
        if model_name is None or model_name not in models_switcher:
            model_name = "br"

        model = models_switcher[model_name]

        df['prediction'] = df['close'].shift(-1)  # TODO: check again here
        with pd.option_context('display.max_rows', None, 'display.max_columns',
                               None):  # more options can be specified also
            print(df)
        df = df[:-1]
        X = np.array(df.drop(['prediction', 'close'], 1))
        y = np.array(df['prediction'])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)

        model.fit(X_train, y_train)
        model_conf = model.score(X_test, y_test)

        return {"model": model,
                "confidence": model_conf,
                "name": model_name,
                "df": df,
                "last_updated": datetime.datetime.now()}

    def get_vertical_projection_of_point_on_line(self, p1, p2, point):
        yA = p1[1]
        yB = p2[1]
        xA = p1[0]
        xB = p2[0]
        xC = point[0]
        m = (yB - yA) / (xB - xA)
        yC = m * (xC - xA) + yA
        return yC

    def compute_vertical_deviation(self, line, input_array):
        p1 = line[0]
        p2 = line[1]

        result = []
        for i in range(0, len(input_array)):
            projection = self.get_vertical_projection_of_point_on_line(p1, p2, [i, input_array[i]])
            absolute_deviation = input_array[i] - projection
            percent_deviation = absolute_deviation * 100 / projection
            result.append(percent_deviation)
        return result

    def compute_percentage_changes(self, input_array):
        result = [0]
        for i in range(1, len(input_array)):
            result.append((input_array[i] - input_array[i - 1]) / input_array[i - 1] * 100)
        return result

    def get_median_line(self, input_array):
        no_of_elem = len(input_array)

        start_point = [0, input_array[0]]
        end_point = [len(input_array) - 1, input_array[len(input_array) - 1]]
        deviation = self.compute_vertical_deviation([start_point, end_point], input_array)
        current_sign_of_elements = sum(deviation)

        if current_sign_of_elements < 0:
            while current_sign_of_elements < 0:
                start_point[1] -= 0.01 * start_point[1]
                end_point[1] -= 0.01 * end_point[1]
                deviation = self.compute_vertical_deviation([start_point, end_point], input_array)
                current_sign_of_elements = sum(deviation)

        else:
            while current_sign_of_elements > 0:
                start_point[1] += 0.01 * start_point[1]
                end_point[1] += 0.01 * end_point[1]
                deviation = self.compute_vertical_deviation([start_point, end_point], input_array)
                current_sign_of_elements = sum(deviation)
        return [start_point, end_point]

    def get_start_and_end_point(self, input_array):
        start_point = [0, input_array[0]]
        end_point = [len(input_array) - 1, input_array[len(input_array) - 1]]
        return [start_point, end_point]

    def init_module(self, stocks_no=None):
        if stocks_no is not None:
            stocks_no = min(stocks_no, len(self.main_stocks))
            for i in range(stocks_no):
                self.generate_regression_model(self.main_stocks[i])
            print("Generated %d models." % stocks_no)

        print("Regression process initialized")

    def generate_regression_model(self, symbol):
        model_info = None
        must_generate_new_model = False

        if symbol not in self.main_stocks_models:
            must_generate_new_model = True
        else:
            diff = datetime.datetime.now() - self.main_stocks_models[symbol]["last_updated"]
            days, seconds = diff.days, diff.seconds
            hours = days * 24 + seconds // 3600
            if hours > 24:
                must_generate_new_model = True

        if must_generate_new_model:
            url = stock_utils.build_url_with_symbol(symbol, True)
            stock_json = stock_utils.get_json_from_url(url)
            df, history_days = stock_utils.process_json_to_pd_with_limit(stock_json, 1000)
            df = df.drop(['timestamp'], 1)
            model_info = self.model_fit_symbol(df)
            model_info["history_days"] = history_days
            self.main_stocks_models[symbol] = model_info
        else:
            model_info = self.main_stocks_models[symbol]

        return model_info


if __name__ == "__main__":
    regressor = Regressor()
    regressor.init_module(5)
    symbol = "AMZN"
    model_info = regressor.generate_regression_model(symbol)
    avg_pred, stock_history = regressor.predict_stock(model_info, regressor.no_of_days_to_predict)

    pred_with_present_day = [stock_history[0]] + avg_pred
    print(pred_with_present_day)
    print(regressor.compute_vertical_deviation(regressor.get_start_and_end_point(pred_with_present_day),
                                               pred_with_present_day))
    # deviation = compute_vertical_deviation(get_median_line(stock_with_history), stock_with_history)
    # print(deviation[len(deviation)//2:])

    plt.show()
