from math import radians, cos, sin, asin, sqrt
import os
import pandas as pd
import numpy as np
import datetime
from dateutil.parser import parse
from tqdm import tqdm
import time
import logging
import pysnooper


def geodistance(lng1, lat1, lng2, lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
    dlon = lng2-lng1
    dlat = lat2-lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance = 2 * asin(sqrt(a)) * 6371.393 * 1000
    distance = round(distance, 5)
    return distance


def judge_direct(lng1, lat1, lng2, lat2, label):
    if label==1:
        return label
    return 1 if geodistance(lng1, lat1, lng2, lat2) <= 10 else 0


def judge_indirect(lng1, lat1, lng2, lat2, label):
    if label==1 or label==2:
        return label
    return 2 if geodistance(lng1, lat1, lng2, lat2) <= 10 else 0


def statistics(data):
    if len(data['label'].value_counts()) == 3:
        normal, direct, indirect = data['label'].value_counts()
    elif len(data['label'].value_counts()) == 2:
        normal, direct, indirect= data['label'].value_counts()[0], data['label'].value_counts()[1], 0
    elif len(data['label'].value_counts()) == 1:
        normal, direct, indirect = data['label'].value_counts()[0], 0, 0
    logging.info("共有 {} 个样本, 其中无任何接触人员 {} 个, 其中直接密接人员 {} 个, 间接密接人员 {} 个".format(normal+direct+indirect, normal, direct, indirect))


# @pysnooper.snoop('./log/file.log')
def first_engage(df, column='usetime'):
    if 1 in df['label'].unique():
        # 找出最早接触的时间
        earliest = min(df[df['label']==1]['usetime'])

        # 所有之后的时间全部标记为1
        df_later = df[(df['usetime']-earliest) >= datetime.timedelta(seconds=0)]
        df.loc[df_later.index, 'label'] = 1
    return df


def combine(df):
    id = df['id'].unique()
    l = df['label'].unique()
    label = 1 if 1 in l else 2 if 2 in l else 0
    return pd.DataFrame({'id': id, 'label':label})


if __name__ == '__main__':

    # 记录程序运行开始时间, 来给生成文件命名
    now = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(time.time()))
    output_dir = './results/{}'.format(now)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # log文件存放位置
    log_file = os.path.join(output_dir, 'search.log'.format(now))
    logging.basicConfig(level = logging.DEBUG, #控制台打印的日志级别
                        filename = log_file,
                        filemode = 'w', # w就是写模式，每次都会重新写日志，覆盖之前的日志, a是追加模式，默认如果不写的话，就是追加模式
                        format = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'#日志格式
    )

    logging.info('开始读取数据')
    confirmed = pd.read_csv('./data/确诊患者亮码记录.csv', parse_dates=['亮码时间'])
    df_travel = pd.read_csv('./data/df_travel.csv', parse_dates=['usetime'])
    df_travel['label'] = 0
    df_travel = df_travel[:]

    # dict_1 = {
    #     'id':[1,1,2,3,3],
    #     'usetime':[parse('Nov 17, 2020 7:09 AM'),parse('Nov 17, 2020 7:12 AM'),parse('Nov 29, 2020 7:05 AM'),parse('Nov 17, 2020 7:13 AM'),parse('Nov 17, 2020 7:09 AM')],
    #     'lat':[36.20256,36.20256,36.20256,36.20256,36.20256],
    #     'lng':[117.20285,157.20285,117.20285,117.20285,117.20285]
    # }
    # df_travel = pd.DataFrame(dict_1)
    # df_travel['label'] = 0
    # df_travel = df_travel[:]

    DELAY = datetime.timedelta(seconds=300)

    logging.info('开始寻找直接接触人员')
    for i in tqdm(range(confirmed.shape[0])):
        lng1, lat1, time1 = confirmed.loc[i, 'lng'], confirmed.loc[i, 'lat'], confirmed.loc[i, '亮码时间']
        df = df_travel[abs(df_travel['usetime']-time1)<=DELAY]
        df_travel.loc[df.index, 'label'] = df.apply(lambda x: judge_direct(lng1, lat1, x.lng, x.lat, x.label), axis = 1)

    # 按id分组
    grouped = df_travel.groupby(df_travel['id'])

    # 将第一次直接接触后的所有出行记录label都标注为1
    middle = grouped.apply(first_engage)
    middle.to_csv(os.path.join(output_dir, 'middle.csv'), index=False, encoding="utf-8")

    direct = middle[middle['label']==1]

    logging.info('共有 {} 次直接接触记录'.format(direct.shape[0]))

    logging.info('开始寻找间接接触人员')
    # for row in tqdm(direct.itertuples(index=True, name='Pandas')):
    #     lng1, lat1, time1 = getattr(row, "lng"), getattr(row, "lat"), getattr(row, "usetime")
    for i in tqdm(range(direct.shape[0])):
        lng1, lat1, time1 = direct.iloc[i, 3], direct.iloc[i, 2], direct.iloc[i, 1]
        # 先分离出所有在时间上有可能是间接接触的样本, 再按照距离去寻找
        df = middle[abs(middle['usetime']-time1)<=DELAY]
        middle.loc[df.index, 'label'] = df.apply(lambda x: judge_indirect(lng1, lat1, x.lng, x.lat, x.label), axis = 1)

        # 直接查找所有样本, 相比上面两句, apply的效率较低
        # middle['label'] = middle.apply(lambda x: judge_indirect(lng1, lat1, time1, x.lng, x.lat, x.usetime, x.label), axis = 1）

    middle.to_csv(os.path.join(output_dir, 'final.csv'), index=False, encoding="utf-8")


    logging.info('共有 {} 次间接接触记录'.format(middle['label'].value_counts()[2]))

    logging.info('开始生成提交文件')
    grouped = middle[['id','label']].groupby(middle['id'])
    submit = grouped.apply(combine)
    statistics(submit)
    submit.to_csv(os.path.join(output_dir, 'submit.csv'), index=False, encoding="utf-8")
