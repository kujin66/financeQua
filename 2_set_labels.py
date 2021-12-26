import os
import pandas as pd
import csv
import codecs
import numpy as np


def round_digits(row):
    return [round(e, 3) for e in row]


def get_date_cnt(row):
    return len(row)


def get_is_valid(row):
    if row[0] == 'nan':
        print(row)
        return 0
    if row[0] == 0:
        print(row)
        return 0

    if len(row) < 20:
        print(row)
        return 0
    return 1


def normalize_by_mean_value(row):
    mean_value = sum(row)/len(row)
    return [round(e/mean_value, 3) for e in row]


def compute_1day_price_rate(row):
    new_row = []
    for i in range(len(row) - 1):
        price_rate = round(100*(row[i+1] - row[i])/row[i], 3)
        new_row.append(price_rate)
    new_row = new_row + [-1]
    return new_row


def compute_3day_price_rate(row):
    new_row = []
    for i in range(len(row) - 3):
        price_rate = round(100*(row[i+3] - row[i])/row[i], 3)
        new_row.append(price_rate)
    new_row = new_row + [-1, -1, -1]
    return new_row


def compute_5day_price_rate(row):
    new_row = []
    for i in range(len(row) - 5):
        price_rate = round(100*(row[i+5] - row[i])/row[i], 3)
        new_row.append(price_rate)
    new_row = new_row + [-1] * 5
    return new_row


def compute_10day_price_rate(row):
    new_row = []
    for i in range(len(row) - 10):
        price_rate = round(100*(row[i+10] - row[i])/row[i], 3)
        new_row.append(price_rate)
    new_row = new_row + [-1] * 10
    return new_row


def add_new_feature(day_info_df):

    # 需要丢弃的列
    # drop_cols = []
    # cols = list(day_info_df.columns)
    # for col in cols:
    #     if '_std' not in col:
    #         if col not in ['ticker', 'ticker_price_rate', 'ticker_range_rate']:
    #             drop_cols.append(col)
    # print('drop_cols:', drop_cols)

    # 删除统计天数在20以内的数据
    day_info_df['ticker_date_cnt'] = day_info_df.ticker_date.apply(get_date_cnt)
    day_info_df['ticker_is_valid'] = day_info_df.ticker_price_mean.apply(get_is_valid)
    day_info_df = day_info_df.drop(day_info_df[day_info_df.ticker_is_valid == 0].index)

    # 1.当天的开盘价
    day_info_df['ticker_open_day'] = day_info_df.ticker_open_day.apply(normalize_by_mean_value)
    day_info_df['ticker_open_mean'] = day_info_df.ticker_open_mean.apply(normalize_by_mean_value)
    day_info_df['ticker_open_std'] = day_info_df.ticker_open_std.apply(normalize_by_mean_value)
    # 2.当天的收盘价
    day_info_df['ticker_last_day'] = day_info_df.ticker_last_day.apply(normalize_by_mean_value)
    day_info_df['ticker_last_mean'] = day_info_df.ticker_last_mean.apply(normalize_by_mean_value)
    day_info_df['ticker_last_std'] = day_info_df.ticker_last_std.apply(normalize_by_mean_value)
    # 3.当天的最高价
    day_info_df['ticker_high_day'] = day_info_df.ticker_high_day.apply(normalize_by_mean_value)
    day_info_df['ticker_high_mean'] = day_info_df.ticker_high_mean.apply(normalize_by_mean_value)
    day_info_df['ticker_high_std'] = day_info_df.ticker_high_std.apply(normalize_by_mean_value)
    # 4.当天的最低价
    day_info_df['ticker_low_day'] = day_info_df.ticker_low_day.apply(normalize_by_mean_value)
    day_info_df['ticker_low_mean'] = day_info_df.ticker_low_mean.apply(normalize_by_mean_value)
    day_info_df['ticker_low_std'] = day_info_df.ticker_low_std.apply(normalize_by_mean_value)
    # 5.5分钟内成交量
    day_info_df['ticker_titradeshare_mean'] = day_info_df.ticker_titradeshare_mean.apply(normalize_by_mean_value)
    day_info_df['ticker_titradeshare_std'] = day_info_df.ticker_titradeshare_std.apply(normalize_by_mean_value)
    # 6.5分钟内成交额（titradevalue）
    day_info_df['ticker_titradevalue_mean'] = day_info_df.ticker_titradevalue_mean.apply(normalize_by_mean_value)
    day_info_df['ticker_titradevalue_std'] = day_info_df.ticker_titradevalue_std.apply(normalize_by_mean_value)
    # 7.当天的总成交量(totaltradeshare)
    day_info_df['ticker_totaltradeshare_day'] = day_info_df.ticker_totaltradeshare_day.apply(normalize_by_mean_value)
    day_info_df['ticker_totaltradeshare_mean'] = day_info_df.ticker_totaltradeshare_mean.apply(normalize_by_mean_value)
    day_info_df['ticker_totaltradeshare_std'] = day_info_df.ticker_totaltradeshare_std.apply(normalize_by_mean_value)
    # 8.当天的总成交额(totaltradevalue)
    day_info_df['ticker_totaltradevalue_day'] = day_info_df.ticker_totaltradevalue_day.apply(normalize_by_mean_value)
    day_info_df['ticker_totaltradevalue_mean'] = day_info_df.ticker_totaltradevalue_mean.apply(normalize_by_mean_value)
    day_info_df['ticker_totaltradevalue_std'] = day_info_df.ticker_totaltradevalue_std.apply(normalize_by_mean_value)
    # 9.当天的平均价
    day_info_df['ticker_price_mean'] = day_info_df.ticker_price_mean.apply(round_digits)
    # 10.std的均值
    day_info_df['ticker_std_mean'] = day_info_df.ticker_std_mean.apply(normalize_by_mean_value)
    # 11.Vwap
    day_info_df['ticker_vwap_mean'] = day_info_df.ticker_vwap_mean.apply(normalize_by_mean_value)
    day_info_df['ticker_vwap_std'] = day_info_df.ticker_vwap_std.apply(normalize_by_mean_value)
    # 12.Twap的均值、标准差
    day_info_df['ticker_twap_mean'] = day_info_df.ticker_twap_mean.apply(normalize_by_mean_value)
    day_info_df['ticker_twap_std'] = day_info_df.ticker_twap_std.apply(normalize_by_mean_value)
    # 13.abspread的均值、标准差
    day_info_df['ticker_abspread_mean'] = day_info_df.ticker_abspread_mean.apply(normalize_by_mean_value)
    day_info_df['ticker_abspread_std'] = day_info_df.ticker_abspread_std.apply(normalize_by_mean_value)
    # 14.相比开盘价，收盘价的涨幅
    day_info_df['ticker_price_rate'] = day_info_df.ticker_price_rate.apply(normalize_by_mean_value)
    # 15.相比最低价，最高价的涨幅
    day_info_df['ticker_range_rate'] = day_info_df.ticker_range_rate.apply(round_digits)
    # 16.开盘五分钟的股价涨幅、成交量占比
    day_info_df['ticker_open_5min_price_rate'] = day_info_df.ticker_open_5min_price_rate.apply(round_digits)
    day_info_df['ticker_open_5min_share_rate'] = day_info_df.ticker_open_5min_share_rate.apply(round_digits)
    # 17.收盘五分钟的股价涨幅、成交量占比
    day_info_df['ticker_last_5min_price_rate'] = day_info_df.ticker_last_5min_price_rate.apply(round_digits)
    day_info_df['ticker_last_5min_share_rate'] = day_info_df.ticker_last_5min_share_rate.apply(round_digits)
    # 18.未来一天的均价的涨幅
    day_info_df['ticker_1day_price_rate'] = day_info_df.ticker_price_mean.apply(compute_1day_price_rate)
    # 19.未来三天的均价涨幅
    day_info_df['ticker_3day_price_rate'] = day_info_df.ticker_price_mean.apply(compute_3day_price_rate)
    # 19.未来五天的均价涨幅
    day_info_df['ticker_5day_price_rate'] = day_info_df.ticker_price_mean.apply(compute_5day_price_rate)
    # 20.未来十天的均价涨幅
    day_info_df['ticker_10day_price_rate'] = day_info_df.ticker_price_mean.apply(compute_10day_price_rate)

    # 19.处理空值
    day_info_df.fillna(0, inplace=True)
    return day_info_df


def aggregate_day_info(csv_reader):
    day_info_df = csv_reader.groupby("ticker").agg(
        ticker_date=("date", list),
        ticker_open_day=("open_day", list),
        ticker_open_mean=("open_mean", list),
        ticker_open_std=("open_std", list),
        ticker_last_day=("last_day", list),
        ticker_last_mean=("last_mean", list),
        ticker_last_std=("last_std", list),
        ticker_high_day=("high_day", list),
        ticker_high_mean=("high_mean", list),
        ticker_high_std=("high_std", list),
        ticker_low_day=("low_day", list),
        ticker_low_mean=("low_mean", list),
        ticker_low_std=("low_std", list),
        ticker_titradeshare_mean=("titradeshare_mean", list),
        ticker_titradeshare_std=("titradeshare_std", list),
        ticker_titradevalue_mean=("titradevalue_mean", list),
        ticker_titradevalue_std=("titradevalue_std", list),
        ticker_totaltradeshare_day=("totaltradeshare_day", list),
        ticker_totaltradeshare_mean=("totaltradeshare_mean", list),
        ticker_totaltradeshare_std=("totaltradeshare_std", list),
        ticker_totaltradevalue_day=("totaltradevalue_day", list),
        ticker_totaltradevalue_mean=("totaltradevalue_mean", list),
        ticker_totaltradevalue_std=("totaltradevalue_std", list),
        ticker_price_mean=("price_mean", list),
        ticker_std_mean=("std_mean", list),
        ticker_vwap_mean=("vwap_mean", list),
        ticker_vwap_std=("vwap_std", list),
        ticker_twap_mean=("twap_mean", list),
        ticker_twap_std=("twap_std", list),
        ticker_abspread_mean=("abspread_mean", list),
        ticker_abspread_std=("abspread_std", list),
        ticker_price_rate=("price_rate", list),
        ticker_range_rate=("range_rate", list),
        ticker_open_5min_price_rate=("open_5min_price_rate", list),
        ticker_open_5min_share_rate=("open_5min_share_rate", list),
        ticker_last_5min_price_rate=("last_5min_price_rate", list),
        ticker_last_5min_share_rate=("last_5min_share_rate", list),
    ).reset_index()
    return day_info_df


if __name__ == "__main__":
    feature_file = '../data/1_feature_engineering_all.csv'
    csv_reader = pd.read_csv(feature_file, sep="\t")

    # 1.按照股票代码和时间，从小到大的排序
    csv_reader = csv_reader.sort_values(by=['ticker', 'date'], ascending=True, inplace=False)
    # print(csv_reader.columns)

    # 2.按照股票代码，将股价信息聚合在一起，形成列表
    day_info_df = aggregate_day_info(csv_reader)

    # 3.标准化特征、添加新特征、添加label
    day_info_df = add_new_feature(day_info_df)

    # 4.保存文件
    out_file = '../data/2_feature_engineering_for_train.csv'
    day_info_df.to_csv(out_file, sep="\t", index=False)




