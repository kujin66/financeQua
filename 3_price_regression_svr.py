import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd
import json


def convert_to_matrix(data):
    # 1.当天的开盘价
    data["ticker_open_day"] = data.ticker_open_day.apply(lambda x: json.loads(x))
    data["ticker_open_mean"] = data.ticker_open_mean.apply(lambda x: json.loads(x))
    data["ticker_open_std"] = data.ticker_open_std.apply(lambda x: json.loads(x))
    # 2.当天的收盘价
    data["ticker_last_day"] = data.ticker_last_day.apply(lambda x: json.loads(x))
    data["ticker_last_mean"] = data.ticker_last_mean.apply(lambda x: json.loads(x))
    data["ticker_last_std"] = data.ticker_last_std.apply(lambda x: json.loads(x))
    # 3.当天的最高价
    data["ticker_high_day"] = data.ticker_high_day.apply(lambda x: json.loads(x))
    data["ticker_high_mean"] = data.ticker_high_mean.apply(lambda x: json.loads(x))
    data["ticker_high_std"] = data.ticker_high_std.apply(lambda x: json.loads(x))
    # 4.当天的最低价
    data["ticker_low_day"] = data.ticker_low_day.apply(lambda x: json.loads(x))
    data["ticker_low_mean"] = data.ticker_low_mean.apply(lambda x: json.loads(x))
    data["ticker_low_std"] = data.ticker_low_std.apply(lambda x: json.loads(x))
    # 5.5分钟内成交量
    data["ticker_titradeshare_mean"] = data.ticker_titradeshare_mean.apply(lambda x: json.loads(x))
    data["ticker_titradeshare_std"] = data.ticker_titradeshare_std.apply(lambda x: json.loads(x))
    # 6.5分钟内成交额（titradevalue）
    data["ticker_titradevalue_mean"] = data.ticker_titradevalue_mean.apply(lambda x: json.loads(x))
    data["ticker_titradevalue_std"] = data.ticker_titradevalue_std.apply(lambda x: json.loads(x))
    # 7.当天的总成交量(totaltradeshare)
    data["ticker_totaltradeshare_day"] = data.ticker_totaltradeshare_day.apply(lambda x: json.loads(x))
    data["ticker_totaltradeshare_mean"] = data.ticker_totaltradeshare_mean.apply(lambda x: json.loads(x))
    data["ticker_totaltradeshare_std"] = data.ticker_totaltradeshare_std.apply(lambda x: json.loads(x))
    # 8.当天的总成交额(totaltradevalue)
    data["ticker_totaltradevalue_day"] = data.ticker_totaltradevalue_day.apply(lambda x: json.loads(x))
    data["ticker_totaltradevalue_mean"] = data.ticker_totaltradevalue_mean.apply(lambda x: json.loads(x))
    data["ticker_totaltradevalue_std"] = data.ticker_totaltradevalue_std.apply(lambda x: json.loads(x))
    # 9.当天的平均价
    data["ticker_price_mean"] = data.ticker_price_mean.apply(lambda x: json.loads(x))
    data["ticker_std_mean"] = data.ticker_std_mean.apply(lambda x: json.loads(x))
    # 10.std的均值
    # data["ticker_std_mean"] = data.ticker_std_mean.apply(lambda x: json.loads(x))
    # 11.Vwap
    data["ticker_vwap_mean"] = data.ticker_vwap_mean.apply(lambda x: json.loads(x))
    data["ticker_vwap_std"] = data.ticker_vwap_std.apply(lambda x: json.loads(x))
    # 12.Twap的均值、标准差
    data["ticker_twap_mean"] = data.ticker_twap_mean.apply(lambda x: json.loads(x))
    data["ticker_twap_std"] = data.ticker_twap_std.apply(lambda x: json.loads(x))
    # 13.abspread的均值、标准差
    data["ticker_abspread_mean"] = data.ticker_abspread_mean.apply(lambda x: json.loads(x))
    data["ticker_abspread_std"] = data.ticker_abspread_std.apply(lambda x: json.loads(x))
    # 14.相比开盘价，收盘价的涨幅
    data["ticker_price_rate"] = data.ticker_price_rate.apply(lambda x: json.loads(x))
    # 15.相比最低价，最高价的涨幅
    data["ticker_range_rate"] = data.ticker_range_rate.apply(lambda x: json.loads(x))
    # 16.开盘五分钟的股价涨幅、成交量占比
    data["ticker_open_5min_price_rate"] = data.ticker_open_5min_price_rate.apply(lambda x: json.loads(x))
    data["ticker_open_5min_share_rate"] = data.ticker_open_5min_share_rate.apply(lambda x: json.loads(x))
    # 17.收盘五分钟的股价涨幅、成交量占比
    data["ticker_last_5min_price_rate"] = data.ticker_last_5min_price_rate.apply(lambda x: json.loads(x))
    data["ticker_last_5min_share_rate"] = data.ticker_last_5min_share_rate.apply(lambda x: json.loads(x))
    # 18.未来一天的均价的涨幅
    data["ticker_1day_price_rate"] = data.ticker_1day_price_rate.apply(lambda x: json.loads(x))
    # 19.未来三天的均价涨幅
    data["ticker_3day_price_rate"] = data.ticker_3day_price_rate.apply(lambda x: json.loads(x))
    # 19.未来五天的均价涨幅
    data["ticker_5day_price_rate"] = data.ticker_5day_price_rate.apply(lambda x: json.loads(x))
    # 20.未来十天的均价涨幅
    data["ticker_10day_price_rate"] = data.ticker_10day_price_rate.apply(lambda x: json.loads(x))
    # 21.统计信息的天数
    # data["ticker_date_cnt"] = data.ticker_date_cnt.apply(lambda x: json.loads(x))
    return data


def train_val_features(csv_reader, feat_day=5, pred_day=1, feature_name='ticker_price_mean'):
    train_features = []
    train_labels = []
    val_features = []
    val_labels = []
    num = 0
    for i in range(len(csv_reader)):
        price_mean_list = csv_reader[feature_name][i]
        for j in range(len(price_mean_list) - 6):
            feature_list = []
            for k in range(feat_day):
                feature_list.append(price_mean_list[j+k])
            label = price_mean_list[j+feat_day+pred_day-1]
            if label != -1:
                num = num + 1
                if j > len(price_mean_list) - 10:
                    val_features.append(feature_list)
                    val_labels.append(label)
                    break
                else:
                    train_features.append(feature_list)
                    train_labels.append(label)
    return np.array(train_features), np.array(train_labels), np.array(val_features), np.array(val_labels)


if __name__ == "__main__":
    feature_file = '../data/2_feature_engineering_for_train.csv'
    csv_reader = pd.read_csv(feature_file, sep="\t")

    # 采用第method_index个技术方案，用前feat_day的时间做为feature，预测未来pred_day的股价
    method_index = 1
    feat_day = 5
    pred_day = 1

    # step1: 存储的时候，每个特征被存为json格式，需要转化为数值表
    csv_reader = convert_to_matrix(csv_reader)

    # step2: 获取训练集的特征和标签
    if method_index == 0:
        # 思路1：股价预测
        feature_name = 'ticker_price_mean'
        train_features, train_labels, val_features, val_labels = train_val_features(csv_reader, feat_day, pred_day, feature_name)
    elif method_index == 1:
        # 思路2：股价涨幅预测
        feature_name = 'ticker_1day_price_rate'
        train_features, train_labels, val_features, val_labels = train_val_features(csv_reader, feat_day, pred_day, feature_name)
    elif method_index == 2:
        # 思路3：多因子预测
        pass

    # step3: 构建模型
    svr_rbf = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
    svr_lin = SVR(kernel="linear", C=100, gamma="auto")
    svr_poly = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)

    # step4: 预测结果
    svr_rbf_pred = svr_rbf.fit(train_features, train_labels).predict(val_features)
    svr_lin_pred = svr_lin.fit(train_features, train_labels).predict(val_features)
    svr_poly_pred = svr_poly.fit(train_features, train_labels).predict(val_features)
