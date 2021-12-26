import os
import pandas as pd
import csv
import codecs
import numpy as np


def get_mean_value(row):
    arr = np.array(row)
    return np.mean(arr)


def get_std(row):
    arr = np.array(row)
    return np.std(arr)


def get_max_value(row):
    return max(row)


def get_min_value(row):
    return min(row)


def get_first_value(row):
    return row[0]


def get_last_value(row):
    return row[-1]


# 可能出错，100*(row[1] - row[0])/row[0]
def get_open_5min_price_rate(row):
    return 100*(row[2] - row[1])/row[1]


def get_last_5min_price_rate(row):
    return 100*(row[-1] - row[-2])/row[-1]


# 可能出错，100*(row[1] - row[0])/row[0]
def get_open_5min_share_rate(row):
    return 100*(row[2] - row[1])/row[-1]


def get_last_5min_share_rate(row):
    return 100*(row[-1] - row[-2])/row[-1]


def add_new_feature(day_info_df):
    # 1.当天的开盘价，五分钟的开盘价的均值和标准差
    day_info_df['open_day'] = day_info_df.day_open_price.apply(get_first_value)
    day_info_df['open_mean'] = day_info_df.day_open_price.apply(get_mean_value)
    day_info_df['open_std'] = day_info_df.day_open_price.apply(get_std)
    # 2.当天的收盘价，五分钟的收盘价的均值和标准差
    day_info_df['last_day'] = day_info_df.day_last_price.apply(get_last_value)
    day_info_df['last_mean'] = day_info_df.day_last_price.apply(get_mean_value)
    day_info_df['last_std'] = day_info_df.day_last_price.apply(get_std)
    # 3.当天的最高价，五分钟的最高价的均值和标准差
    day_info_df['high_day'] = day_info_df.day_high_price.apply(get_max_value)
    day_info_df['high_mean'] = day_info_df.day_high_price.apply(get_mean_value)
    day_info_df['high_std'] = day_info_df.day_high_price.apply(get_std)
    # 4.当天的最低价，五分钟的最低价的均值和标准差
    day_info_df['low_day'] = day_info_df.day_low_price.apply(get_min_value)
    day_info_df['low_mean'] = day_info_df.day_low_price.apply(get_mean_value)
    day_info_df['low_std'] = day_info_df.day_low_price.apply(get_std)
    # 5.5分钟内成交量（titradeshare）的均值，标准差
    day_info_df['titradeshare_mean'] = day_info_df.day_titradeshare.apply(get_mean_value)
    day_info_df['titradeshare_std'] = day_info_df.day_titradeshare.apply(get_std)
    # 6.5分钟内成交额（titradevalue）的均值，标准差
    day_info_df['titradevalue_mean'] = day_info_df.day_titradevalue.apply(get_mean_value)
    day_info_df['titradevalue_std'] = day_info_df.day_titradevalue.apply(get_std)
    # 7.当天的总成交量(totaltradeshare)的最大值、均值、方差
    day_info_df['totaltradeshare_day'] = day_info_df.day_totaltradeshare.apply(get_max_value)
    day_info_df['totaltradeshare_mean'] = day_info_df.day_totaltradeshare.apply(get_mean_value)
    day_info_df['totaltradeshare_std'] = day_info_df.day_totaltradeshare.apply(get_std)
    # 8.当天的总成交额(totaltradevalue)的最大值、均值、方差
    day_info_df['totaltradevalue_day'] = day_info_df.day_totaltradevalue.apply(get_max_value)
    day_info_df['totaltradevalue_mean'] = day_info_df.day_totaltradevalue.apply(get_mean_value)
    day_info_df['totaltradevalue_std'] = day_info_df.day_totaltradevalue.apply(get_std)
    # 9.当天的平均价，totaltradevalue/totaltradeshare，及其均值和标准差
    day_info_df['price_mean'] = day_info_df['totaltradevalue_day']/day_info_df['totaltradeshare_day']
    # 10.std的均值
    day_info_df['std_mean'] = day_info_df.day_std.apply(get_mean_value)
    # 11.Vwap的均值、标准差
    day_info_df['vwap_mean'] = day_info_df.day_vwap.apply(get_mean_value)
    day_info_df['vwap_std'] = day_info_df.day_vwap.apply(get_std)
    # 12.Twap的均值、标准差
    day_info_df['twap_mean'] = day_info_df.day_twap.apply(get_mean_value)
    day_info_df['twap_std'] = day_info_df.day_twap.apply(get_std)
    # 13.abspread的均值、标准差
    day_info_df['abspread_mean'] = day_info_df.day_abspread.apply(get_mean_value)
    day_info_df['abspread_std'] = day_info_df.day_abspread.apply(get_std)
    # 14.相比开盘价，收盘价的涨幅
    day_info_df['price_rate'] = 100*(day_info_df['last_day'] - day_info_df['open_day'])/day_info_df['open_day']
    # 15.相比最低价，最高价的涨幅
    day_info_df['range_rate'] = 100*(day_info_df['high_day'] - day_info_df['low_day']) / day_info_df['high_day']
    # 16.开盘五分钟的股价涨幅、成交量占比
    day_info_df['open_5min_price_rate'] = day_info_df.day_open_price.apply(get_open_5min_price_rate)
    day_info_df['open_5min_share_rate'] = day_info_df.day_totaltradeshare.apply(get_open_5min_share_rate)
    # 17.收盘五分钟的股价涨幅、成交量占比
    day_info_df['last_5min_price_rate'] = day_info_df.day_last_price.apply(get_last_5min_price_rate)
    day_info_df['last_5min_share_rate'] = day_info_df.day_totaltradeshare.apply(get_last_5min_share_rate)
    # 18.集合竞价的成交量占比，开盘股价涨幅
    # 19.处理空值
    day_info_df.fillna(0, inplace=True)
    return day_info_df


def aggregate_day_info(csv_reader):
    day_info_df = csv_reader.groupby("ticker").agg(
        day_time=("time", list),
        day_open_price=("open", list),
        day_last_price=("last", list),
        day_high_price=("high", list),
        day_low_price=("low", list),
        day_titradeshare=("titradeshare", list),
        day_titradevalue=("titradevalue", list),
        day_totaltradeshare=("totaltradeshare", list),
        day_totaltradevalue=("totaltradevalue", list),
        day_askavgprice=("askavgprice", list),
        day_bidavgprice=("bidavgprice", list),
        day_asktotalshare=("asktotalshare", list),
        day_bidtotalshare=("bidtotalshare", list),
        day_std=("std", list),
        day_vwap=("vwap", list),
        day_twap=("twap", list),
        day_ask1price=("ask1price", list),
        day_ask1share=("ask1share", list),
        day_bid1price=("bid1price", list),
        day_bid1share=("bid1share", list),
        day_abspread=("abspread", list)
    ).reset_index()

    # day_info_df = day_info_df.drop(day_info_df[len(day_info_df.day_time) < 10].index)
    # csv_reader = csv_reader.drop(csv_reader[csv_reader.ticker == 'string'].index)
    return day_info_df


def feature_engineer(file_path, cur_date):
    csv_reader = pd.read_csv(file_path)
    # 1.丢弃一行
    csv_reader = csv_reader.drop(csv_reader[csv_reader.ticker == 'string'].index)

    # 2.转化数值类型
    csv_reader = csv_reader.apply(pd.to_numeric)
    csv_reader = csv_reader.drop(csv_reader[csv_reader.open == 0.0].index)
    # csv_reader = csv_reader.apply(pd.to_numeric, errors='ignore')
    # csv_reader[['ticker', 'time', '']] = csv_reader[['ticker', 'time']].astype(int)

    # 3.按照股票代码和时间，从小到大的排序
    csv_reader = csv_reader.sort_values(by=['ticker', 'time'], ascending=True, inplace=False)

    # 4.按照股票代码，将股价信息聚合在一起，形成列表
    day_info_df = aggregate_day_info(csv_reader)
    drop_cols = list(day_info_df.columns)
    drop_cols.remove('ticker')
    day_info_df['date'] = cur_date

    # 5.构建特征
    day_info_df = add_new_feature(day_info_df)

    # 6.丢掉一些特征
    day_info_df = day_info_df.drop(drop_cols, axis=1)

    return day_info_df


if __name__ == "__main__":
    data_dir_part1 = '/Users/kunjin/Downloads/5min2019'
    data_dir_part2 = '/Users/kunjin/Downloads/5min2020_2021'

    print('step1: 读取并处理 5min2019 文件夹下的数据')
    i = 0
    stock_info_df = pd.DataFrame()
    for csv_file in os.listdir(data_dir_part1):
        i = i + 1
        cur_date = int(csv_file.replace('.csv', ''))
        file_path = os.path.join(data_dir_part1, csv_file)
        day_info_df = feature_engineer(file_path, cur_date)
        if stock_info_df.empty:
            stock_info_df = day_info_df
        else:
            stock_info_df = pd.concat([stock_info_df, day_info_df], axis = 0)

        print('Done:', csv_file, str(i) + '/' + str(len(os.listdir(data_dir_part1))))

    print('step2: 读取并处理 5min2020_2021 文件夹下的数据')
    i = 0
    for csv_file in os.listdir(data_dir_part2):
        i = i + 1
        cur_date = int(csv_file.replace('.csv', ''))
        file_path = os.path.join(data_dir_part2, csv_file, )
        day_info_df = feature_engineer(file_path, cur_date)
        if stock_info_df.empty:
            stock_info_df = day_info_df
        else:
            stock_info_df = pd.concat([stock_info_df, day_info_df], axis=0)
        print('Done:', csv_file, str(i) + '/' + str(len(os.listdir(data_dir_part2))))

    # step2: 保存CSV文件
    out_file = '../data/1_feature_engineering_all.csv'
    stock_info_df.to_csv(out_file, sep="\t", index=False)
