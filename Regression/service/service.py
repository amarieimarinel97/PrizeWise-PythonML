import cherrypy
import tensorflow as tf
import processing.regression.regression as regr
import processing.utils.stock_utils as su
import processing.sentiment_analysis.sentiment_analysis as sa
import numpy


class RegressionService(object):

    @cherrypy.expose("context")
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def get_stock_info(self):
        request_body = cherrypy.request.json
        days = 5
        try:
            if request_body['days'] is not None:
                days = request_body['days']
        except KeyError:
            print("Number of days not inserted")

        model_info = regr.generate_regression_model(request_body['symbol'])
        avg_pred, stock_history = regr.predict_stock(model_info, days)

        history_days = model_info["history_days"][:len(stock_history)]
        history_days.reverse()

        pred_with_present_day = [stock_history[0]] + avg_pred
        deviation = regr.compute_vertical_deviation(regr.get_start_and_end_point(pred_with_present_day),
                                                    pred_with_present_day)
        response = {
            "predictedEvolution": pred_with_present_day,
            "pastEvolution": stock_history.tolist(),
            "percentageChanges": regr.compute_percentage_changes(pred_with_present_day),
            "deviation": deviation,
            "pastDays": history_days
        }
        return response

    @cherrypy.expose("textual")
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def get_sentiment_analysis(self):
        request_body = cherrypy.request.json
        print("Received this: ", request_body)
        if isinstance(request_body['text'], list):
            result = []
            text = request_body['text']
            for ind in tf.range(len(text)):
                prediction = sa.pad_predict_sample(text[ind], True)
                result.append(prediction)
            response = {
                "sentiment_analysis": result
            }
        else:
            result = []
            text = request_body['text'].split("|")
            for ind in tf.range(len(text)):
                prediction = sa.pad_predict_sample(text[ind], True)
                result.append(prediction)
            result = numpy.mean(result)
            response = {
                "sentiment_analysis": result
            }

        return response


if __name__ == '__main__':
    regr.init_module(0)
    sa.init_module()
    cherrypy.quickstart(RegressionService(), '/', {'global': {'server.socket_host': '0.0.0.0', 'server.socket_port': 8081}})

