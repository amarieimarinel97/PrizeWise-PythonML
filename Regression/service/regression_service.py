import cherrypy
import processing.stock_utils as stock_utils


class RegressionService(object):

    @cherrypy.expose("stock")
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def stock_info(self):
        request_body = cherrypy.request.json
        url = stock_utils.build_url_with_symbol(request_body['symbol'])
        stock_json = stock_utils.get_json_from_url(url)
        df = stock_utils.process_json_to_pd(stock_json)
        return df.to_json()


if __name__ == '__main__':
    cherrypy.server.socket_host='127.0.0.2'
    cherrypy.quickstart(RegressionService())
