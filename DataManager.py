"""
    AlphaNet: Data preparing
    Step:
        In this part of code we first get the exactly transaction date data from the database, then for each stock get it ipo
        time and delist time which is a single record. After that, for each stock we get its multi-dimensional value data and
        calculate return5/return10 and also filter the "-" type data. Finally, use numpy.lib.stride_tricks.as_strided to split
        the data picture and save it into the folder `ori_data` by using os library. From then on, we can make merge the needed
        training and testing data;
"""
import os
import json
import copy
import sys
import joblib
import requests
import datetime
import numpy as np
import pandas as pd
from typing import Dict
from multiprocessing import Pool
from urllib.parse import parse_qs, urlparse
from sklearn.model_selection import train_test_split
from numpy.lib.stride_tricks import as_strided as stride


host = '192.168.1.114'
port = 30300


def get_data(params: Dict, url=f'http://{host}:{port}/tqmain/equity_daily', res=None) -> pd.DataFrame:
    """
    主要获取DB中的数据
    :param params: GET方法需要的query string，字典参数
    :param url: 资源的url地址
    :param res: 循环获取分页时会用到
    :return:
    """
    if not res:
        res = []
    assert not urlparse(url).params, 'url中不要放入query string'

    r = requests.get(
        url,
        params=params)
    assert r.status_code == 200, f'请求失败： {r.status_code} {r.text}'
    j = json.loads(r.text)

    res.extend(j['results'])

    if j['next']:  # 如果有下一页，递归获取直至无下一页
        query = urlparse(j['next']).query
        page = parse_qs(query).get('page')[0]
        params['page'] = page
        return get_data(
            url=url,
            res=res,
            params=params
        )

    res = pd.DataFrame(res)
    return res


def get_tick(params: Dict,  url: str) -> pd.DataFrame:  # url 默认获取股票日度表
    """
    主要获取以文件形式存放的tick数据
    :param params: GET方法需要的query string，字典参数
    :param url: 资源的url地址
    :return:
    """
    assert not urlparse(url).params, 'url中不要放入query string'

    r = requests.get(
        url,
        params=params)
    assert r.status_code == 200, f'请求失败： {r.status_code} {r.text}'
    j = json.loads(r.text)
    columns = j['columns'].split(',')
    if not j['data']:
        return pd.DataFrame()
    data = j['data'].split('\n')
    data = [i.split(',') for i in data]

    res = pd.DataFrame(data, columns=columns)
    return res


def get_token(usr: str, pwd: str) -> str:
    """
    获取用户token
    """
    url = f'http://{host}:{port}/tq_user/login'
    r = requests.post(url, data={
        "username": usr,
        "password": pwd
    })
    j = json.loads(r.text)
    assert 'error' not in j, f"error: {j['error']}"
    assert 'token' in j, f"error: {j}"
    return j['token']


def is_trading_date(dt='2021-01-01', exchange_cn='上海证券交易所') -> bool:
    url = f"http://{host}:{port}/tqmain/trading_date_info/"
    params = {
        "nature_date": dt,
        "exchange_cn": exchange_cn,
    }
    r = requests.get(url, params=params)
    j = json.loads(r.text)
    return j['results'][0]['is_trading_date']


def get_authenticate_data(token: str) -> pd.DataFrame:
    """
    用token访问tq_model数据
    """
    url = f'http://{host}:{port}/tq_model/lgt_factor/?trading_date=2021-04-08'
    header = {'Authorization': f'Token {token}'}  # 把token放入header中
    r = requests.get(url, headers=header)
    assert r.status_code == 200, f'error: {r.status_code}; {r.text}'
    j = json.loads(r.text)
    df = pd.DataFrame(j['results'])
    return df


def get_user_id_by_token(token: str) -> Dict:
    url = f'http://{host}:{port}/tq_user/validate_token'
    data = {'token': token}
    r = requests.post(url, data=data)
    assert r.status_code == 200, f'error: {r.status_code}; {r.text}'
    j = json.loads(r.text)
    return j


class DataMgr:
    def __init__(self, pool_num):
        """
        create the save path
        """
        self.pool_num = pool_num

        # for original daily.csv and _factor.pkl, _ret5.pkl, _ret10.pkl
        self.save_path = os.path.dirname(os.path.abspath(__file__)) + '/ori_data/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # for creating factor_path_dict
        self.data_processing_save_path = os.path.dirname(os.path.abspath(__file__)) + '/data_processing/'
        if not os.path.exists(self.data_processing_save_path):
            os.makedirs(self.data_processing_save_path)

        # for creating factor_data
        self.factor_data_save_path = self.data_processing_save_path + 'factor_data/'
        if not os.path.exists(self.factor_data_save_path):
            os.makedirs(self.factor_data_save_path)

        # for creating training_data
        self.training_save_path = os.path.dirname(os.path.abspath(__file__)) + '/train_data/'
        if not os.path.exists(self.training_save_path):
            os.makedirs(self.training_save_path)

    def get_date_list(self, start_date='2011-01-31'):
        doc = self.save_path + 'date_list.csv'
        if os.path.exists(doc):
            date = pd.read_csv(doc)
            return date
        url = f"http://{host}:{port}/tqmain/trading_date_info/"
        params = {'exchange_cn': '上海证券交易所', 'start_date': start_date}
        r = requests.get(url, params=params)
        j = json.loads(r.text)
        df = pd.DataFrame(j['results'])
        df = df[df['is_trading_date']]
        df.rename(columns={'nature_date': 'trading_date'}, inplace=True)
        df.sort_values(by='trading_date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df = df[['trading_date', 'is_trading_date']]
        df.to_csv(doc, index=None)
        return df

    def get_stock_list(self, start_date='2011-01-31'):
        """
        target:
            主要解决的问题是，股票状态会更新，一个股票可能在数据库里进行多次 upload，最终需要改成一条包括 ipo_date 和 delist_date 的数据
            在计算因子的时候，再根据 delist 截断或者进行后续处理
        step:
            1) 首先对所有股票根据 delist_date 分类：delist_date 有在"1900-01-01"之前记录的证明有一段时间是没有停牌的，标记为1，有确定停牌时间的就标记为0
            （这部分股票的记录是唯一一条的，获取为：股票代码，上市日期，摘牌日期）
            2) 对于后续标记为1的股票中，有部分股票是有多条记录的，只需要选取退市前最新的数据即可
        param:
            :param start_date: 数据开始日期
        database:
            ipo_date: 首发上市日期
            delist_date: 摘牌日期
            upload_time: 更新时间，对于单只股票，其在数据库里的数据只到upload_time.max()为止
            mkt: 上市板
            exchange_en: 交易所名称
        """
        doc = self.save_path + 'stock_list.csv'
        if os.path.exists(doc):
            data = pd.read_csv(doc)
            return data.to_dict('records')
        dic = {'fields': 'wind_code,ipo_date,delist_date,upload_time,mkt,stock_name,exchange_en'}
        url = f'http://{host}:{port}/tqmain/equity_header'
        data = get_data(dic, url)
        data = data[data['exchange_en'] != 'BSE']
        data['upload_time'] = data['upload_time'].apply(lambda x: datetime.datetime.strptime(x.split('T')[0], '%Y-%m-%d'))
        data['list_date'] = data['ipo_date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
        data['delist_date'] = data['delist_date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
        delist_stock_list = data[data['delist_date'] < '1900-01-01']['wind_code'].unique()
        data['to_del'] = data['wind_code'].apply(lambda x: 1 if x in delist_stock_list else 0)
        st_stock = copy.deepcopy(data[data['to_del'] < 1])
        for stock in delist_stock_list:
            stock_data = copy.deepcopy(data[data['wind_code'] == stock])
            stock_data = stock_data.sort_values(by='upload_time')
            stock_data['to_del'] = stock_data['stock_name'].apply(lambda x: 1 if '退' in x else 0)
            delist_date = stock_data['delist_date'].max()
            stock_data['delist_date'] = stock_data['upload_time'].shift(-1).fillna(delist_date)
            stock_data = stock_data[stock_data['to_del'] < 1]
            st_stock = st_stock.append(stock_data)
        st_stock = st_stock.sort_values(by='upload_time')
        st_stock = st_stock.drop_duplicates(subset=['wind_code'], keep='last')
        st_stock = st_stock[(st_stock['delist_date'] < '1900-01-01') | (st_stock['delist_date'] > start_date)]
        st_stock = st_stock.sort_values(by='wind_code')
        st_stock.reset_index(drop=True, inplace=True)
        st_stock[['wind_code', 'list_date', 'delist_date']].to_csv(doc, index=None)
        return st_stock[['wind_code', 'list_date', 'delist_date']].to_dict('records')

    def get_stock_daily(self, stock_dict, stock_path, start_date='2011-01-31'):
        """
        target:
            五日/十日收益率：停牌的交易日不算
            开仓不能成交删除，平仓不能成交填充，接近现实情况
            在计算y的时候，可以采用如下方法，以 000001.SZ 为例：
                2011-04-29 的未来五日收益率应为：(634.451 - 653.481) / 653.481 * 100 = -2.912%
                2011-05-03 的未来五日收益率应为：(615.780 - 654.199) / 654.199 * 100 = -5.873%
                2011-05-04 的未来五日收益率应为：(641.273 - 641.273) / 641.273 * 100 = 0.0%
                2011-05-05 的未来五日收益率应为：(650.249 - 636.605) / 636.605 * 100 = 2.143%
                2011-05-06 的未来五日收益率应为：(643.786 - 629.783) / 629.783 * 100 = 2.223%
                2011-05-09/10/11/12 的未来五日收益率应为：停牌，开仓不能成交，删除
                2011-05-13 的未来五日收益率应为：(658.508 - 634.451) / 634.451 * 100 = 3.792%
        step:
            1) 对于数据先进行 data = data[data['volume'] > 0]， 计算五日/十日收益率
            2) 再 merge 回去缺失的交易日
        param:
            :param stock_dict: {'wind_code': '000001.SZ', 'list_date': '1991-04-03', 'delist_date': '1899-12-30'}
            :param stock_path: self.path + '000001.SZ/'
            :param start_date: 数据开始日期
        database:
            vwap: 均价
            pct_change: 涨跌幅
            adj_factor: 复权因子
            turn: 换手率
            free_turn: 换手率（基准为自由流通股本）
            trade_status: 交易状态
            is_st: 是否为st股票
        """
        doc = stock_path + 'daily.csv'
        if os.path.exists(doc):
            data = pd.read_csv(doc)
            return data
        dic = {'wind_code': stock_dict['wind_code'], 'start_date': start_date, 'fields': 'trading_date,wind_code,open_price,high_price,low_price,close_price,vwap,volume,pct_change,adj_factor,turn,free_turn,trade_status,is_st'}
        data = get_data(dic)
        data.sort_values(by='trading_date', inplace=True)
        data.reset_index(drop=True, inplace=True)
        data.rename(columns={'open_price': 'open', 'high_price': 'high', 'low_price': 'low', 'close_price': 'close', 'pct_change': 'pct_chg'}, inplace=True)
        for col in ['open', 'high', 'low', 'close', 'vwap']:
            data[col] = round(data[col] * data['adj_factor'], 3)
        for col in ['pct_chg', 'adj_factor', 'turn', 'free_turn']:
            data[col] = round(data[col], 3)
        data = data[data['volume'] > 0]
        data['close/free_turn'] = data['close'] / data['free_turn']
        data['open/turn'] = data['open'] / data['turn']
        data['volume/low'] = data['volume'] / data['low']
        data['vwap/high'] = data['vwap'] / data['high']
        data['low/high'] = data['low'] / data['high']
        data['vwap/close'] = data['vwap'] / data['close']
        data.to_csv(doc, index=None)
        return data

    def cal_ret(self, data):
        data['ret_5'] = data['close'].shift(-5) / data['close'] - 1
        data['ret_10'] = data['close'].shift(-10) / data['close'] - 1
        return data

    def filter_daily(self, data):
        """
        target:
            过滤了一字涨停和跌停的股票(盘中涨停的股票，实盘也已经交易部分了), 剔除每个截面期下一交易日停牌的股票
            回测一般默认以 vwap(成交量加权平均价) 成交, 不会冲击市场, 有公司做算法交易, bench就是vwap
            拆单: 根据历史规律, 预测成交量的时间分布, 然后分配不通的权重, 后面就根据还差多少没成交, 适当调整下单量, 如果研究盘口订单簿的数据, 可能预测很短期的方向, 就可以做的更精细
            'trade_status' 所有可能的状态为:
                '交易',
                '停牌一天': 重大事项 -> data = data[data['volume'] > 0] 可以过滤
                '盘中停牌': 盘中成交价较开盘价首次上涨达到或超过10%, 股票价格异常波动 -> data = data[data['volume'] > 0] 不可以过滤
                '下午停牌': 重要事项未公告, 拟披露重大事项, 重大事项 -> data = data[data['volume'] > 0] 不可以过滤
                '上午停牌': 未能及时披露年度报告, 公共媒体报道告, 拟披露重大事项, 重大事项 -> data = data[data['volume'] > 0] 不可以过滤
                '暂停上市': -> data = data[data['volume'] > 0] 可以过滤
        """
        data['next_status'] = data['trade_status'].shift(-1)
        data['next_status'] = data['next_status'].fillna('交易')  # 假定未来一天为交易
        data['diff'] = data['high'] - data['low']  # 剔除一字涨停
        data['next_diff'] = data['diff'].shift(-1)
        data['next_diff'] = data['next_diff'].fillna(100)
        data = data[(data['trade_status'] == '交易') & (data['next_status'] == '交易') & (data['is_st'] == 0) & (data['next_diff'] > 0.001)]
        data = data.reset_index(drop=True)
        data = data.dropna()
        data = data.reset_index(drop=True)
        return data

    def cal_one_stock(self, stock_dict, date_df):
        window = 30
        if '1900-01-01' < stock_dict['delist_date'] < '2011-01-31':
            return
        stock = stock_dict['wind_code']
        stock_path = self.save_path + stock + '/'
        if not os.path.exists(stock_path):
            os.makedirs(stock_path)
        daily_data = self.get_stock_daily(stock_dict, stock_path)
        daily_data = self.cal_ret(daily_data)
        daily_data = self.filter_daily(daily_data)
        daily_data = pd.merge(daily_data, date_df, how='right')
        daily_data.sort_values(by='trading_date', inplace=True)
        daily_data = daily_data.reset_index(drop=True)
        daily_data['wind_code'] = daily_data['wind_code'].fillna(method='bfill')
        daily_data = daily_data.dropna(subset=['wind_code'])
        daily_data = daily_data.reset_index(drop=True)
        daily_data['open'] = daily_data['open'].fillna(-1)
        col_list = ['open', 'high', 'low', 'close', 'vwap', 'volume', 'pct_chg', 'turn', 'free_turn']
        stride_values = self.roll_ar(daily_data[col_list], window)
        date_series = daily_data['trading_date']
        ret_5_series = daily_data['ret_5']
        ret_10_series = daily_data['ret_10']
        for i in range(window - 1, len(daily_data)):
            item = stride_values[i - (window - 1)]
            if -1 in item[0]:
                continue
            joblib.dump(item, stock_path + date_series[i].replace('-', '') + '_factor.pkl')  # 文件名保存的为数据图片的结束日期
            joblib.dump(np.array([[ret_5_series[i]]]), stock_path + date_series[i].replace('-', '') + '_ret5.pkl')
            joblib.dump(np.array([[ret_10_series[i]]]), stock_path + date_series[i].replace('-', '') + '_ret10.pkl')

    def roll_ar(self, data, window):
        """
        target:
            input: data, window
            output: (n, 9, 30)
        step:
            v.strides: strides是numpy数组对象的一个属性, 官方手册给出的解释是跨越数组各个维度所需要经过的字节数(bytes), float = 8, int32 = 4
            而对于 as_strided: 返回的是X的View, 所以strides的理解是, 在矩阵X的基础上，按照给定的strides, 来切割出一个符合给定shape的视图
            对于4维的划分则为:
                A[i, j, k, x], A[i, j, k, x+1] 对应X中移动几个字节
                A[i, j, k, x], A[i, j, k+1, x] 对应X中移动几个字节
                A[i, j, k, x], A[i, j+1, k, x] 对应X中移动几个字节
                A[i, j, k, x], A[i+1, j, k, x] 对应X中移动几个字节
        """
        v = data.reset_index(drop=True).values
        # 竖着滚动
        # dim0, dim1 = v.shape
        # stride0, stride1 = v.strides
        # stride_values = stride(v, (dim0 - (window - 1), window, dim1), (stride0, stride0, stride1))

        # 横着滚动
        v = v.T
        dim1, dim0 = v.shape
        stride1, stride0 = v.strides
        stride_values = stride(v, (dim0 - (window - 1), dim1, window), (stride0, stride1, stride0))  # frequency = 1
        # stride_values = stride(v, ((dim0 - (window - 1) + 1) // 2, dim1, window), (2 * stride0, stride1, stride0))  # frequency = 2
        return stride_values

    def cal_stock(self, stock_list, date_df):
        """
        step:
            注意不能让方法为静态方法, 否则无法调动进程池, 同时不能使用 Python Console, 需要使用 Run / Terminal
        """
        if self.pool_num > 0:
            pool = Pool(self.pool_num)
            for stock_dict in stock_list:
                pool.apply_async(self.cal_one_stock, args=(stock_dict, date_df, ))
            pool.close()
            pool.join()
        else:
            for stock_dict in stock_list:
                self.cal_one_stock(stock_dict, date_df)

    def run(self):
        date_df = self.get_date_list()
        stock_list = self.get_stock_list()
        self.cal_stock(stock_list, date_df)

    def get_factor_path_dict(self):
        """
        target:
            factor_path_dict -> 日期: str - 因子路径: list
        step:
            os.walk(top[, topdown=True[, onerror=None[, followlinks=False]]]) -> (root, dirs, files):
                root 所指的是当前正在遍历的这个文件夹的本身的地址
                dirs 是一个 list , 内容是该文件夹中所有的目录的名字(不包括子目录)
                files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
                topdown 可选，为 True, 则优先遍历 top 目录, 否则优先遍历 top 的子目录(默认为开启)。如果 topdown 参数为 True, walk 会遍历top文件夹, 与 top 文件夹中每一个子目录。
        """
        doc = self.data_processing_save_path + 'factor_data_dict.pkl'
        if os.path.exists(doc):
            path_dict = joblib.load(doc)
            return path_dict
        path_dict = {}
        for root, dirs, files in os.walk(self.save_path):
            for name in files:
                if '_factor.pkl' in name:
                    date = name.split('_')[0]
                    path = root + '/' + name
                    if date in path_dict:
                        path_dict[date].append(path)
                    else:
                        path_dict[date] = [path]
                else:
                    continue
        joblib.dump(path_dict, doc)
        return path_dict

    def merge_one_date(self, date, path_list, save_path):
        factor_data = np.concatenate([np.array([[joblib.load(path)]]) for path in sorted(path_list)], axis=0)
        target_data_ret5 = np.concatenate([joblib.load(path.replace('_factor', '_ret5'))[0] for path in sorted(path_list)], axis=0)
        target_data_ret5_norm = (target_data_ret5 - target_data_ret5.mean()) / target_data_ret5.std()
        target_data_ret10 = np.concatenate([joblib.load(path.replace('_factor', '_ret10'))[0] for path in sorted(path_list)], axis=0)
        target_data_ret10_norm = (target_data_ret10 - target_data_ret10.mean()) / target_data_ret10.std()
        stock_data = np.array([[x.split('/')[-2], date] for x in path_list])
        joblib.dump(factor_data, save_path + date + '_factor.pkl')
        joblib.dump(target_data_ret5, save_path + date + '_ret5.pkl')
        joblib.dump(target_data_ret5_norm, save_path + date + '_ret5_norm.pkl')
        joblib.dump(target_data_ret10, save_path + date + '_ret10.pkl')
        joblib.dump(target_data_ret10_norm, save_path + date + '_ret10_norm.pkl')
        joblib.dump(stock_data, save_path + date + '_stock.pkl')

    def merge_stock(self):
        factor_path_dict = self.get_factor_path_dict()
        if self.pool_num > 0:
            pool = Pool(self.pool_num)
            for date, path_list in factor_path_dict.items():
                pool.apply_async(self.merge_one_date, args=(date, path_list, self.factor_data_save_path, ))
            pool.close()
            pool.join()
        else:
            for date, path_list in factor_path_dict.items():
                self.merge_one_date(date, path_list, self.factor_data_save_path)

    def merge_train_data_base(self, date_list, save_path, factor_path, suffix, num, NORM):
        path_list = [factor_path + x.replace('-', '') + '_factor.pkl' for x in date_list]

        factor_data = np.concatenate([joblib.load(path) for path in sorted(path_list)], axis=0)
        if NORM:
            target_data = np.concatenate([joblib.load(path.replace('_factor', '_ret{}_norm'.format(num))) for path in sorted(path_list)], axis=0)
        else:
            target_data = np.concatenate([joblib.load(path.replace('_factor', '_ret{}'.format(num))) for path in sorted(path_list)], axis=0)
        stock_data = np.concatenate([joblib.load(path.replace('_factor', '_stock')) for path in sorted(path_list)], axis=0)

        joblib.dump(factor_data, save_path + suffix + '_factor.pkl')
        if NORM:
            joblib.dump(target_data, save_path + suffix + '_ret{}_norm.pkl'.format(num))
        else:
            joblib.dump(target_data, save_path + suffix + '_ret{}.pkl'.format(num))
        joblib.dump(stock_data, save_path + suffix + '_stock.pkl')

    def merge_train_data(self, NUM, TRAIN_SIZE, NORM=False):
        """
        :param NUM: it decide whether to calculate _ret5 or _ret10
        :param TRAIN_SIZE: it decide the length of training part and validation part
        :param NORM: it decide whether use normalization(Z-score) data
        """
        date_df = self.get_date_list()
        date_list = sorted([x for x in list(date_df['trading_date']) if x[0:4] == '2020' and x[5:7] < '07'])
        train_date_list, valid_date_list = train_test_split(date_list, train_size=TRAIN_SIZE, shuffle=False)
        valid_date_list = valid_date_list[NUM:]
        self.merge_train_data_base(train_date_list, self.training_save_path, self.factor_data_save_path, 'train', NUM, NORM)
        self.merge_train_data_base(valid_date_list, self.training_save_path, self.factor_data_save_path, 'valid', NUM, NORM)


if __name__ == '__main__':
    GetData = DataMgr(4)
    # GetData.run()
    # GetData.merge_stock()
    GetData.merge_train_data(5, 0.8)


