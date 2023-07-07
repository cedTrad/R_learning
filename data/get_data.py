import pandas as pd
import numpy as np
import sqlalchemy


path = "C:/Users/cc/Desktop/CedAlgo/database/"

def Get_data(asset , interval="1d"):
    engine = sqlalchemy.create_engine('sqlite:///'+path+'database_{}.db'.format(interval))
    
    data = pd.read_sql(asset+'USDT' ,engine)
    data.set_index('time' , inplace=True)
    data['volume'] = pd.to_numeric(data['volume'])
    data = data[['open', 'high', 'low' , 'close' , 'volume', 'symbol']]
    return data


def data_m(assets, interval="1d"):
    data = pd.DataFrame().reindex_like(Get_data("BTC", interval))['open']
    assets_to_drop = []
    for asset in assets:
        try:
            df = Get_data(asset, interval)['close'].rename(asset)
            data = pd.concat([data, df], axis=1)
        except:
            assets_to_drop.append(asset)
            continue
    data.drop(columns='open', inplace=True)
    return data, assets_to_drop


def get_data(assets, interval = "1d", start = "2017", end = "2023"):
    data, assets_to_drop = data_m(interval = interval, assets=assets)
    for asset in assets_to_drop:
        assets.remove(asset)
    data_r = data.pct_change().loc[start : end, assets]
    return data_r