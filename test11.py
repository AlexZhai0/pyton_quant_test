
import os
import datetime

import threading
threadLock = threading.Lock()

import requests
import re

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib

import pyecharts.options as opts
from pyecharts.charts import Page, Line

#指定默认字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family']='sans-serif'
#解决负号'-'显示为方块的问题
matplotlib.rcParams['axes.unicode_minus'] = False


# 获取上个工作日日期
def get_last_work_date():
    date = datetime.datetime.today()
    w = date.weekday() + 1
    format = '%Y-%m-%d'
    if w==1:
        last_work_date = (date + datetime.timedelta(days = -3)).strftime(format)
    elif 1<w<7:
        last_work_date = (date + datetime.timedelta(days = -1)).strftime(format)
    elif w==7:
        last_work_date = (date + datetime.timedelta(days = -2)).strftime(format)
    # print(f'今天{date.strftime("%Y-%m-%d")}周{w}, 上个工作日是{last_work_date}周{w-1}')
    return last_work_date


# 根据当前交易日获取下一个交易日
def get_next_work_date(current_trade_date=''):
    format = '%Y-%m-%d'
    current_date = datetime.datetime.strptime(current_trade_date, format)
    w = current_date.weekday() + 1
    if 1<=w<5 or w==7:
        next_work_date = (current_date + datetime.timedelta(days = +1)).strftime(format)
    elif w==5:
        next_work_date = (current_date + datetime.timedelta(days = +3)).strftime(format)
    elif w==6:
        next_work_date = (current_date + datetime.timedelta(days = +2)).strftime(format)
    return next_work_date


def get_url(url, params=None):
    rsp = requests.get(url, params=params)
    # rsp.encoding = 'utf-8'
    # rsp.raise_for_status()
    return rsp.text

def get_fund_data(code, per=20, sdate='', edate=''):
    url = 'http://fund.eastmoney.com/f10/F10DataApi.aspx'
    # url = 'https://fundf10.eastmoney.com/F10DataApi.aspx?type=lsjz&code=110022&page=1&per=20'
    params = {'type': 'lsjz', 'code': code, 'page':1,'per': per, 'sdate': sdate, 'edate': edate}
    html = get_url(url, params)
    
    # 获取总页数
    pattern = re.compile(r'pages:(.*),')
    result = re.search(pattern, html).group(1)
    pages = int(result)
    print('总页数：%r' %(pages))

    # 从第1页开始抓取所有页面数据
    records = []
    page = 1
    while page <= pages:
        params = {'type': 'lsjz', 'code': code, 'page': page ,'per': per, 'sdate': sdate, 'edate': edate}
        html = get_url(url, params)
        bs = BeautifulSoup(html, 'html.parser')
        for row in bs.findAll('tbody')[0].findAll('tr'):
            row_records = []
            for record in row.findAll('td'):
                val = record.contents
                if val == []:
                    row_records.append(np.nan)
                else:
                    row_records.append(val[0])
            records.append(row_records)
        page += 1

    # 获取表头
    bs = BeautifulSoup(html, 'html.parser')
    heads = []
    for head in bs.findAll('th'):
        heads.append(head.contents[0])
    
    # 数据整理到dataframe
    np_records = np.array(records)
    data = pd.DataFrame()
    for col, col_name in enumerate(heads):
        data[col_name] = np_records[:,col]
    print(f'网络获取数据成功,code:{code},数据:')
    print(data)
    return data

code300 = '110020' # 沪深300指数
path300 = '/Users/alex/Desktop/A_%s.xlsx' % code300
code = '161725' # 白酒指数
# code = '519674' # 银河创新成长混合A
path = '/Users/alex/Desktop/A_%s.xlsx' % code
# print(path)

# 把DataFrame插入到另一个DataFrame中
def insert(df, i, df_add):
    # 指定第i行插入一行数据
    df1 = df.iloc[:i, :]
    df2 = df.iloc[i:, :]
    df_new = pd.concat([df1, df_add, df2], ignore_index=False)
    # 删除自动生成的Unnamed:0列
    df_new.drop(df_new.filter(regex="Unname"),axis=1, inplace=True)
    return df_new

# 从本地Excel表中获取数据，如果没有就从网络请求，请求成功后存储到Excel中
def get_file_fund_data(code,path=''):
    threadLock.acquire()
    data = pd.DataFrame()
    # 当前最新交易日，上一天
    last_work_date = get_last_work_date()
    if os.path.exists(path):
        data = pd.read_excel(path)
        print('本地有数据(excel)')
        # 是否更新最新数据
        last_table_data = data.loc[0]
        # 表中的最新日期
        last_table_date = last_table_data['净值日期']
        # 表中的下一个交易日
        next_table_date = get_next_work_date(last_table_date)
        # 如果表中的最新日期小于实际上个交易日，就重新获取最新的数据，插入之前数据后再次存储到本地
        if last_table_date<last_work_date:
            print('本地不是最新数据,需要更新')
            new_data = get_fund_data(code,per=49,sdate=next_table_date,edate=last_work_date)
            print(f'新数据:{new_data}')
            # 插入到dataframe中，再重新写入到文件中
            data = insert(data, 0, new_data).reset_index(drop=True)
            # data = data.reset_index(drop=True)
            data.to_excel(path, sheet_name='data', header=1)
            print('已经存入到本地的最新数据:')
            print(data)
    else :
        print('本地没有数据')
        data = get_fund_data(code,per=49,sdate='2000-01-01',edate=last_work_date)
        data.to_excel(path, sheet_name='data', header=1)
    threadLock.release()
    return data


if __name__ == '__main__':    
    # 获取当前基金数据
    data = get_file_fund_data(code, path)
    # 获取沪深300基金数据
    data300 = get_file_fund_data(code300, path300)
    net_asset_value300 = data300['单位净值']
    net_asset_value300_total = data300['累计净值']
    # 把沪深300的「单位净值」数据插入到当前基金数据中
    data.insert(data.shape[1], '单位净值300', net_asset_value300)
    data.insert(data.shape[1], '累计净值300', net_asset_value300_total)
    
    data['净值日期'] = pd.to_datetime(data['净值日期'], format='%Y/%m/%d')
    data['单位净值'] = data['单位净值'].astype(float)
    # data = data.sort_values(by='净值日期', axis=0, ascending=True).reset_index(drop=True)
    data = data.iloc[::-1] # 正序排列
    print('正序排列:')
    print(data)

    

    # 过滤时间
    start='2020-04-27'
    end='2023-04-27'
    data = data[(data['净值日期'] <= end) & (data['净值日期'] >= start)]
    print('过滤一定时间段:')
    print(data)
    # data = data.tail(100)

    
    data2 = data[['净值日期', '累计净值', '累计净值300']].set_index(['净值日期'], drop=True)
    print('重置源数据：')
    print(data2)
    # print(data2[data2['累计净值'] >= 3.1])
    data3 = (1 + data2.pct_change()).cumprod()
    print('增长率：')
    print(data3)
    data3.loc[:,['累计净值','累计净值300']].plot(figsize=(10,5), grid=True)
    plt.show()


    # data['相对净值'] = (data['单位净值'].astype(float) - float(data.iloc[0]['单位净值'])) / float(data.iloc[0]['单位净值'])
    # data['相对净值300'] = (data['单位净值300'].astype(float) - float(data.iloc[0]['单位净值300'])) / float(data.iloc[0]['单位净值300'])
    # print('相对净值:')
    # print(data)
    

    
    # # 使用matplotlib.pyplot显示图片
    # net_value_date = data['净值日期']
    # net_asset_value = data['相对净值']
    # net_asset_value300 = data['相对净值300']
    # fig = plt.figure()
    # ax1 = fig.add_subplot()
    # ax1.plot(net_value_date, net_asset_value)
    # ax1.plot(net_value_date, net_asset_value300)
    # ax1.set_ylabel('净值数据')
    # ax1.set_xlabel('日期')
    # plt.title('基金净值走势图')
    # plt.legend(loc='upper left')
    # plt.show()



    # # 使用HTML显示
    # page = Page()
    # x_data = list(data['净值日期'])
    # y_data = list(data['相对净值'])
    # y_data300 = list(data['相对净值300'])
    # line = Line(init_opts=opts.InitOpts(width="1800px"))
    # line.set_global_opts(
    #     tooltip_opts=opts.TooltipOpts(is_show=True, trigger="axis"),
    #     # legend_opts=opts.LegendOpts(is_show=True,),
    #     title_opts=opts.TitleOpts(
    #         title='走势图',
    #         title_textstyle_opts=opts.TextStyleOpts(
    #             font_family='KaiTi',
    #             font_size=24,
    #             color='#FF1493'
    #         )
    #     ),
    #     xaxis_opts=opts.AxisOpts(
    #         type_="time",
    #         name='净值日期',
    #         # axispointer_opts=opts.AxisPointerOpts(is_show=True)
    #     ),
    #     yaxis_opts=opts.AxisOpts(
    #         type_="value",
    #         name='单位净值',
    #         min_ = min(y_data), #np.minimum(min(y_data), min(y_data300))
    #         max_= np.maximum(max(y_data), max(y_data300)),
    #         # axistick_opts=opts.AxisTickOpts(is_show=True),
    #         # axispointer_opts=opts.AxisPointerOpts(is_show=True, type_="line"),
    #     )
    # )
    # line.add_xaxis(xaxis_data=x_data)
    # line.add_yaxis(
    #     series_name="白酒折线",
    #     y_axis=y_data,
    #     symbol="circle",
    #     is_symbol_show=True,
    #     label_opts=opts.LabelOpts(is_show=False),
    # ) 

    # line.add_yaxis(
    #     series_name="沪深300折线",
    #     y_axis=y_data300,
    #     symbol="circle", #emptyCircle
    #     is_symbol_show=True,
    #     label_opts=opts.LabelOpts(is_show=False),
    # )

    # page.add(line)
    # table_name = '走势图'
    # page.render(f'/Users/alex/Desktop/{table_name}.html')
    # print('已存入 %s', table_name)
    # page.render_embed()

