import json
import pandas as pd
import urllib.request


def get_json_from_file(filename):
    with open('stock.json') as json_file:
        data = json.load(json_file)
        json_file.close()
        return data


def process_json_to_pd(json_data):
    info_list = list()
    for time_reg in json_data['Time Series (Daily)']:
        curr_row = list()
        curr_row.append(time_reg)
        for stock_reg in json_data['Time Series (Daily)'][time_reg]:
            curr_row.append(json_data['Time Series (Daily)'][time_reg][stock_reg])
        info_list.extend([curr_row])
    index = [i for i in range(0, len(info_list))]
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'vol']
    return pd.DataFrame(info_list, index, columns)


def header(msg):
    print('-' * 50)
    print('[ ' + msg + ' ]')


def get_json_from_url(url_input):
    with urllib.request.urlopen(url_input) as url:
        return json.load(url)


def build_url_with_symbol(symbol):
    return "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s" \
           "&apikey=5ZZAGO8AS4V9XCXD&outputsize=full" % symbol


if __name__ == "__main__":
    json_data = get_json_from_url("https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY"
                                  "&symbol=DAVA&interval=60min&apikey=5ZZAGO8AS4V9XCXD")
    print(process_json_to_pd(json_data))
