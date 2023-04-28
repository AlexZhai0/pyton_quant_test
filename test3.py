
import numpy as np
import scipy.stats as stats
# import scipy.optimize as opt

from scipy import optimize

from pandas import Series, DataFrame

import pandas as pd

import os

import datetime


# rv_unif = stats.uniform.rvs(size=10)
# stats.rvs_ratio_uniforms()

# result = optimize.curve_fit()


# a = np.arange(20).reshape(4, 5)
# print(a)
# loc = np.where(a==11) 
# print(loc)
# b = loc[0][0]
# c = loc[1][0]
# print(b)
# print()


# a = np.random.rand(2,2) 
# print(a)
# b = np.random.rand(2,2)
# print(b)
# c = np.hstack([a,b]) 
# print(c)
# d = np.vstack([a,b])
# print(d)

# a = np.random.randn(3)
# print(a)
# s = Series(a)
# print(s)

# s = Series(np.random.randn(5),index=['a', 'b', 'c', 'd', 'e']) 
# print(s)
# print(s[0])
# print(s[:2])
# print(s[[2,0,4]])
# print(s[['b', 'd']])
# print(s[s > 0.5])
# print('e' in s)

# d = {'one': Series([1., 2.], index=['a', 'b']), 'two': Series([1., 2., 3.], index=['a', 'b', 'c'])} 
# df = DataFrame(d)
# print(df)

# d = {'one': [1., 2., 3.], 'two': [4., 3., 2.]} 
# df = DataFrame(d, index=['a', 'b', 'c'])
# print(df)

# d = {'one': Series([1., 2.], index=['a', 'b']), 'two': Series([1., 2., 3.], index=['a', 'b', 'c'])} 
# df = DataFrame(d, index=['d', 'b', 'a'], columns=['two', 'three' ])
# print(df)

# a = Series(range(5))
# # print(a)
# b = Series(np.linspace(4, 20, 5))
# # print(b)
# df = pd.concat([a, b], axis=1)
# print(df)
# print([a, b])

# # df1 = DataFrame()
# df2 = DataFrame([Series(range(5))], index=['a'])
# df3 = DataFrame([np.linspace(4, 20, 5)], index=['b'])
# df2 = pd.concat([df2, df3], axis=0)
# print(df2)

# s = Series(range(5))
# print(s)
# print([s])

# df = DataFrame()
# index = ['alpha', 'beta', 'gamma', 'delta', 'eta'] 
# # for i in range(1): 
# a = DataFrame([np.linspace(0, 0, 5)], index=[index[0]]) 
# df = pd.concat([df, a], axis=0) 
# print(df)


df = DataFrame() 
index = ['alpha', 'beta', 'gamma', 'delta', 'eta'] 
for i in range(5): 
    a = DataFrame([np.linspace(i, 5*i, 5)], index=[index[i]]) 
    df = pd.concat([df, a], axis=0) 
# print(df)

# print(df[1]) #获取对应列（没列名时）df[1]。单独获取的列，类型都是Series
df.columns = ['a', 'b', 'c', 'd', 'e']
# print(df) # 测试数据
# print(df['b']) #获取对应列df['b']，有列名之后就不能用 df[1]
# print(df.b) #获取对应列df.b
# print(df[['a', 'd']]) #获取其中两列，类型DataFrame

# df = df.iloc[1:]
# print(df)
# df = df.pct_change()
# print(df)
# print(1+df)
# df = (1+df).cumprod()
# print(df)

# N = 2
# s = Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# # 计算出当前值和前第N个值的增加(减少)比例
# s_shift = s.shift() / s.shift(N+1) - 1
# # [nan, nan, nan, 2.0, 1.0, 0.67, 0.5, 0.39, 0.33, 0.28]
# result3 = [1 if e>0.5 else 0 if e<=0.5 else np.nan for e in s_shift]
# # 结果：[nan, nan, nan, 1, 1, 1, 0, 0, 0, 0]

s = Series([0.8, 1.12, 1.09, 0.98, 0.88, 0.99, 1.11, 1.21, 1.33])
#s = Series([300, 321, 303, 311, 305, 289, 288, 299, 310, 330])
# result1 = s.pct_change()
# print(result1)
# result2 = (1+result1).cumprod()
# print(result2)

# s2 = Series([1, 1, 0, 0, 0, 1, 1, 1, 1])
# result3 = (1+(s2.shift(1) * s).sum(axis=1)).cumprod()
# print(result3)

# code300 = '110020' # 沪深300指数
# path300 = '/Users/alex/Desktop/A_%s.xlsx' % code300
# #源数据
# data = pd.read_excel(path300)
# #截取部分日期的源数据
# data1 = data[(data['净值日期'] >= '2023-01-01') & (data['净值日期'] <= '2023-01-31')]
# print('部分日期的源数据：')
# print(data1)
# #只选取两列，并逆序排列
# data2 = data1[['净值日期', '单位净值']].iloc[::-1].reset_index(drop=True)
# #以日期为index
# data3 = data2.set_index(["净值日期"], drop=True)
# print('目标数据：')
# print(data3)
# #计算这段时间内的增长率
# data4 = (1 + data3.pct_change()).cumprod()
# print('增长率：')
# print(data4)

# #以目标源数据为基础，设置对应的策略
# data31 = DataFrame(data=0, index=data3.index, columns=data3.columns)
# #假设使用了日历策略，每月前五个交易日满仓，其余交易日空仓
# data31.iloc[0, 0] = 1
# data31.iloc[1, 0] = 1
# data31.iloc[2, 0] = 1
# data31.iloc[3, 0] = 1
# data31.iloc[4, 0] = 1
# print('策略数据：')
# print(data31)
# data32 = (1 + data31.shift(1) * data3.pct_change()).cumprod()
# print('策略增长率：')
# print(data32)


# print(df['b'][2])
# print(df['c']['delta'])

# print(df.iloc[2])
# print(df.loc['beta'])

# print(df[1:4])
# bool_vec = [True, False, True, False, False]
# print(df[bool_vec])

# print(df[['b', 'd']].iloc[[1, 3]])
# print(df.iloc[[1, 3]][['b', 'd']])
# print(df[['b', 'd']].loc[['beta', 'delta']])
# print(df.loc[['beta', 'delta']][['b', 'd']])

# print(df.iat[2, 3])
# print(df.at['gamma', 'd'])

# print(df.head(3))
# print(df.tail(3))

# dates = pd.date_range('20150101', periods=5)
# df = pd.DataFrame(np.random.randn(5,4),index=dates,columns=list('ABCD'))
# print(df.describe)


# 获取当前脚本文件的绝对路径
# script_path = os.path.abspath(__file__)
# path1 = os.getcwd()
# print(path1)
# path = os.path.dirname(__file__)
# print(path)
# print(script_path)
# dir_path = os.path.dirname(script_path)
# print(script_path)


# time = datetime.timedelta(31)
# print(time)
# time2 = datetime.timedelta(-1)
# print(time2)
# time_now = datetime.datetime.now()
# print(time_now)













