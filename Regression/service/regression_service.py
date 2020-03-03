import json

from json import JSONEncoder
import cherrypy
import processing.stock_utils as stock_utils
import processing.regression as regression

import numpy


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


class RegressionService(object):

    @cherrypy.expose("stock_regr")
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def stock_info(self):
        request_body = cherrypy.request.json
        url = stock_utils.build_url_with_symbol(request_body['symbol'])
        days = 10
        try:
            if request_body['days'] is not None:
                days = request_body['days']
        except KeyError:
            print("Number of days not inserted")

        stock_json = stock_utils.get_json_from_url(url)
        # df = stock_utils.process_json_to_pd(stock_json)
        # df = df[:1000]
        df = stock_utils.process_json_to_pd_with_limit(stock_json, 1000)
        avg_pred, stock_history = regression.predict_stock_with_multiple_regressors(df, days)

        stock_with_history = list(stock_history)
        stock_with_history.extend(avg_pred)
        deviation = regression.compute_vertical_deviation(regression.get_median_line(stock_with_history),
                                                          stock_with_history)
        print(deviation[len(deviation) // 2:])
        print(avg_pred)
        response = {
            'prediction': avg_pred.tolist(),
            'deviation': deviation
        }
        return json.dumps(response, indent=2)


if __name__ == '__main__':
    cherrypy.server.socket_host = '127.0.0.3'
    cherrypy.quickstart(RegressionService())
