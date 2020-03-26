import cherrypy

import json
import processing.regression as regr
import processing.stock_utils as su
import processing.sentiment_analysis as sa


class RegressionService(object):

    @cherrypy.expose("stock_regr")
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def get_stock_info(self):
        request_body = cherrypy.request.json
        url = su.build_url_with_symbol(request_body['symbol'])
        days = 10
        try:
            if request_body['days'] is not None:
                days = request_body['days']
        except KeyError:
            print("Number of days not inserted")

        stock_json = su.get_json_from_url(url)
        df = su.process_json_to_pd_with_limit(stock_json, 1000)

        avg_pred, stock_history = regr.predict_stock_with_multiple_regressors(df, days)

        pred_with_present_day = [stock_history[0]] + avg_pred
        deviation = regr.compute_vertical_deviation(regr.get_start_and_end_point(pred_with_present_day),
                                                    pred_with_present_day)
        response = {
            "prediction": pred_with_present_day,
            "changes": regr.compute_percentage_changes(pred_with_present_day),
            "deviation": deviation
        }
        return response

    @cherrypy.expose("sent_analysis")
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def get_sentiment_analysis(self):
        request_body = cherrypy.request.json
        if isinstance(request_body['text'], list):
            result = []
            # for pred in sa.model.predict(request_body['text']):
            #     result.append(float(pred[0]))
            for text in request_body['text']:
                result.append(  sa.pad_predict_sample(text, False)  )
            response ={
                "sentiment_analysis": result
            }
        else:
            result = sa.predict_sample(request_body['text'])
            response = {
                "sentiment_analysis": float(result[0][0])
            }

        return response


if __name__ == '__main__':
    sa.init_module()
    cherrypy.server.socket_host = '127.0.0.2'
    cherrypy.quickstart(RegressionService())