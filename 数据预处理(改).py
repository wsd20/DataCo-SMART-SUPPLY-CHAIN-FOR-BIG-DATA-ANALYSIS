# 导入所有需要安装的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
import datetime as dt
import calendar,warnings,itertools,matplotlib,keras,shutil
import tensorflow as tf
import statsmodels.api as sm
from datetime import datetime
from IPython.core import display as ICD
# from tensorflow_core.estimator import inputs

#导入数据集
dataset=pd.read_csv("./项目/数据分析/DataCode智能供应链物流数据集分析/数据集/DataCoSupplyChainDataset.csv",header= 0,encoding= 'unicode_escape')
dataset.head(5)# 查看数据前5条

# 整个数据集由180519条记录和53列组成
dataset.shape
dataset.columns
dataset.apply(lambda x: sum(x.isnull())) # 查看缺失值
'''
数据包括Customer Lname(客户名称)、产品描述、订单Zipcode和Customer Zipcode中的一些丢失值，在继续分析之前，
这些值应该被删除或替换。另外，由于有可能不同的客户可能具有相同的名字或相同的姓氏，因此创建一个新的列，
其中包含“customer full name”，以避免任何歧义。
'''

# 将名字和姓氏添加到一起以创建新列
dataset['Customer Full Name'] = dataset['Customer Fname'].astype(str)+dataset['Customer Lname'].astype(str)

# print(type(dataset.columns))
#
for i in dataset.columns:
    print(i)

# 为了便于分析，删除一些不重要的列
data=dataset.drop(['Customer Email','Product Status','Customer Password',
                   'Customer Street','Customer Fname','Customer Lname',
                   'Latitude','Longitude','Product Description','Product Image',
                   'Order Zipcode','shipping date (DateOrders)'],axis=1)

data.shape
data.columns
# Customer Zipcode列中缺少3个值。由于缺少的值只是邮政编码，并不十分重要，因此在继续进行数据分析之前，这些值将被替换为零
data['Customer Zipcode']=data['Customer Zipcode'].fillna(0)#使用0来填充缺失值

# 为了找到重要的参数，进行数据关联。

# 进行数据可视化
fig, ax = plt.subplots(figsize=(24,24))
sns.heatmap(data.corr(),annot=True,linewidths=.5,fmt='.1g',cmap= 'coolwarm') # 相关矩阵热图
plt.show()

'''
我们可以看到，产品价格与销售、订单项目总数有着很高的相关性。
由于分析中使用的数据与供应链相关，因此找到哪个地区的销售最多是有意义的
可以使用groupby方法找到，该方法将相似的市场区域分开，并使用“sum”函数添加该特定区域的所有销售。
'''

plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
market = data.groupby('Market') #通过Market聚合
region = data.groupby('Order Region')
plt.figure(1)
market['Sales per customer'].sum().sort_values(ascending=False).plot.bar(figsize=(12,6), title="所有市场的总销售额")
plt.xticks(rotation=45)
plt.show()
plt.figure(2)
region['Sales per customer'].sum().sort_values(ascending=False).plot.bar(figsize=(12,8), title="所有地区的总销售额")
plt.xticks(rotation=60)
plt.show()

'''
从图表中可以看出，欧洲市场的销售额最多，而非洲市场的销售额最少，在这些市场中，
西欧地区和中美洲的销售额最高
这里可以用同样的方法查看销售额最高的产品类别
'''

data.groupby('Category Name').count()


#按照类别分组聚合
cat = data.groupby('Category Name')

cat['Sales per customer'].sum()


plt.figure(1)
# 所有类别的总销售额
cat['Sales per customer'].sum().sort_values(ascending=False).plot.bar(figsize=(12,6), title="所有类别的总销售额")
plt.show()
plt.figure(2)
# 所有类别的平均销售额
cat['Sales per customer'].mean().sort_values(ascending=False).plot.bar(figsize=(12,6), title="所有类别的平均销售额")
plt.show()
plt.figure(3)
# 所有类别的平均价格
cat['Product Price'].mean().sort_values(ascending=False).plot.bar(figsize=(12,6), title="所有类别的平均价格")
plt.show()

'''
从图1中我们可以看出，捕鱼类产品的销售量最大，其次是夹板。
平均价格最高的前7名产品是平均销售量最大的产品，
尽管价格为1500美元，但计算机的销售量几乎为1350台。由于价格和销售额之间的相关性很高，
因此了解价格如何影响所有产品的销售额是很有意义的，以了解趋势。
'''

data.plot(x='Product Price', y='Sales per customer',linestyle='dotted',
          markerfacecolor='blue', markersize=12)
plt.title('产品价格与每位客户的销售额')
plt.xlabel('产品价格')
plt.ylabel('每个客户的销售额')
plt.show()

'''
可以看出，价格与销售额呈线性关系。哪个季度的销售额最高
将订货时间分为年、月、周、日、小时，可以更好地观察趋势。
'''

data['order_year']= pd.DatetimeIndex(data['order date (DateOrders)']).year
data['order_month'] = pd.DatetimeIndex(data['order date (DateOrders)']).month
data['order_week_day'] = pd.DatetimeIndex(data['order date (DateOrders)']).weekday
data['order_hour'] = pd.DatetimeIndex(data['order date (DateOrders)']).hour
data['order_month_year'] = pd.to_datetime(data['order date (DateOrders)']).dt.to_period('M')


quater= data.groupby('order_month_year')
quartersales=quater['Sales'].sum().resample('Q').mean().plot(figsize=(15,6))
plt.title('年月销售额')
plt.show()

'''
从上图可以看出，从2015年第1季度到2017年第3季度，
销售额一直保持稳定，到2018年第1季度，销售额突然下降。
查看周、日、时、月的购买趋势如何
'''

plt.figure(figsize=(10,12))
plt.subplot(4, 2, 1)
quater= data.groupby('order_year')
quater['Sales'].mean().plot(figsize=(12,12),title='三年平均销售额')
plt.subplot(4, 2, 2)
days=data.groupby("order_week_day")
days['Sales'].mean().plot(figsize=(12,12),title='每周平均销售额（天）')
plt.subplot(4, 2, 3)
hrs=data.groupby("order_hour")
hrs['Sales'].mean().plot(figsize=(12,12),title='每天平均销售额（小时）')
plt.subplot(4, 2, 4)
mnth=data.groupby("order_month")
mnth['Sales'].mean().plot(figsize=(12,12),title='年平均销售额（月）')
plt.tight_layout()
plt.show()



'''
价格是如何影响销售的，什么时候，哪些产品有更多的销售，
订单数量最多的是10月份，其次是11月份，所有其他月份的订单都是一致的。
2017年客户的订单数量最多。周六的平均销量最高，周三的销量最少。
无论时间长短，平均销售额始终保持一致，标准差为3。
同样重要的是要知道什么样的支付方式是首选的人购买所有这些产品在所有地区
可以使用.unique（）方法查看不同的付款方式。
'''

data['Type'].unique()
# 支付方式为'DEBIT', 'TRANSFER', 'CASH', 'PAYMENT'('借记'、'转账'、'现金'、'付款')

# 研究发现有四种支付方式，不同地区的人最喜欢哪种支付方式？

#xyz = data.groupby('Type')
xyz1 = data[(data['Type'] == 'TRANSFER')]
xyz2= data[(data['Type'] == 'CASH')]
xyz3= data[(data['Type'] == 'PAYMENT')]
xyz4= data[(data['Type'] == 'DEBIT')]
count1=xyz1['Order Region'].value_counts()
count2=xyz2['Order Region'].value_counts()
count3=xyz3['Order Region'].value_counts()
count4=xyz4['Order Region'].value_counts()
names=data['Order Region'].value_counts().keys()
n_groups=23
fig,ax = plt.subplots(figsize=(20,8))
index=np.arange(n_groups)
bar_width=0.2
opacity=0.6
type1=plt.bar(index,count1,bar_width,alpha=opacity,color='b',label='Transfer')
type2=plt.bar(index+bar_width,count2,bar_width,alpha=opacity,color='r',label='Cash')
type3=plt.bar(index+bar_width+bar_width,count3,bar_width,alpha=opacity,color='g',label='Payment')
type4=plt.bar(index+bar_width+bar_width+bar_width,count4,bar_width,alpha=opacity,color='y',label='Debit')
plt.xlabel('订单区域')
plt.ylabel('付款次数')
plt.title('所有地区使用不同类型的付款')
plt.legend()
plt.xticks(index+bar_width,names,rotation=90)
plt.tight_layout()
plt.show()

'''
借方类型是所有地区人们最喜欢的支付方式，现金支付是最不喜欢的方式。
有些产品在每个订单上都有负收益，这表明订单给公司带来了收入损失。查看这些是什么产品
'''


loss = data[(data['Benefit per order']<0)]
#显示损失最大的十大产品
plt.figure(1)
loss['Category Name'].value_counts().nlargest(10).plot.bar(figsize=(20,8), title="损失最大的产品")
plt.show()
plt.figure(2)
loss['Order Region'].value_counts().nlargest(10).plot.bar(figsize=(20,8), title="损失最大的地区")
#损失的总销售额的总和
plt.show()
print('订单损失总收入',loss['Benefit per order'].sum())

'''
损失总额约390万，这是一个巨大的数额。可以看出，鞋类是损失销售最多的类别，其次是男式鞋。
大多数损失销售发生在中美洲和西欧地区。这种损失的销售可能是由于涉嫌欺诈或延迟交货而发生的。
找出使用哪种支付方式进行欺诈，可以有助于防止欺诈在未来发生
'''

#产品价格检查用于进行除转账以外的欺诈的付款类型
xyz = data[(data['Type'] != 'TRANSFER')&(data['Order Status'] == 'SUSPECTED_FRAUD')]
xyz['Order Region'].value_counts()

'''
可以清楚地看到，没有欺诈进行借记，现金，付款方式，
所以所有可疑的欺诈订单是通过电汇可能从国外。
哪个地区和什么产品最容易被怀疑欺诈
'''

high_fraud = data[(data['Order Status'] == 'SUSPECTED_FRAUD') & (data['Type'] == 'TRANSFER')]#分离涉嫌欺诈的订单
#绘制有关订单区域的饼图
fraud=high_fraud['Order Region'].value_counts().plot.pie(figsize=(24,12),
                                                         startangle=180, explode=(0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),autopct='%.1f',shadow=True,)
plt.title("欺诈最高的地区",size=15,color='y')
plt.ylabel(" ")
fraud.axis('equal')
plt.show()

'''
可以观察到，涉嫌欺诈的订单数量最多的是西欧，约占订单总数的17.4%，
其次是中美洲，占15.5%。查看哪种产品被怀疑欺诈最多
'''
high_fraud1 = data[(data['Order Status'] == 'SUSPECTED_FRAUD')] #
high_fraud2 = data[(data['Order Status'] == 'SUSPECTED_FRAUD') &(data['Order Region'] == 'Western Europe')]
#绘制各地区十大最可疑欺诈部门条形图
fraud1=high_fraud1['Category Name'].value_counts().nlargest(10).plot.bar(figsize=(20,8), title="欺诈类别",color='orange')
#绘制西欧十大最可疑欺诈部门条形图
fraud2=high_fraud2['Category Name'].value_counts().nlargest(10).plot.bar(figsize=(20,8), title="西欧欺诈产品",color='green')
plt.legend(["所有地区", "西欧"])
plt.title("欺诈检测率最高的十大产品", size=15)
plt.xlabel("产品", size=13)
plt.ylim(0,600)
plt.show()

'''
鞋钉部门被怀疑欺诈最多，其次是在所有地区和西欧的男鞋。
查看哪些客户在进行所有这些欺诈
'''


#过滤出可疑的FRAUD订单
cus = data[(data['Order Status'] == 'SUSPECTED_FRAUD')]
#为十大最可疑欺诈客户绘制条形图
cus['Customer Full Name'].value_counts().nlargest(10).plot.bar(figsize=(20,8), title="十大欺诈客户")
plt.show()

'''
一个叫玛丽·史密斯的顾客就要为自己的欺诈行为负责528次，
查看她到底做了多少欺诈订单
'''


#过滤涉嫌欺诈的玛丽·史密斯的订单
amount = data[(data['Customer Full Name'] == 'MarySmith')&(data['Order Status'] == 'SUSPECTED_FRAUD')]
#为十大最可疑欺诈客户绘制条形图
amount['Sales'].sum()

'''
总金额近102k，金额非常巨大，由于每次下单时玛丽都使用不同的地址，
每次都会发出新的客户id，这使得客户很难识别并禁止。
为了提高欺诈检测算法的识别精度，应考虑到这些参数。
'''

'''
*******************************************************************
*******************************************************************
*******************************************************************
*******************************************************************
'''

'''
对于供应链公司来说，按时向客户交付产品而不延迟交付是另一个重要方面，
因为如果产品不能按时交付，客户将不会满意。哪一类产品交货最晚？
'''
#筛选具有延迟交付状态的列
late_delivery = data[(data['Delivery Status'] == 'Late delivery')]
#最迟交货的前10个产品
late_delivery['Category Name'].value_counts().nlargest(10).plot.bar(figsize=(201,8), title="最迟交货的前10个产品")
plt.show()
'''
可以看出，鞋钉部的订单延迟最多，其次是男鞋，
对于一些订单，数据中给出了延迟交货的风险，
并将延迟交货风险的产品与延迟交货的产品进行了比较。
'''

#过滤具有延迟交货风险的订单
xyz1 = data[(data['Late_delivery_risk'] == 1)]
#筛选延迟交货订单
xyz2 = data[(data['Delivery Status'] == 'Late delivery')]
count1=xyz1['Order Region'].value_counts()
count2=xyz2['Order Region'].value_counts()
#索引名称
names=data['Order Region'].value_counts().keys()
n_groups=23
fig,ax = plt.subplots(figsize=(20,8))
index=np.arange(n_groups)
bar_width=0.2
opacity=0.8
type1=plt.bar(index,count1,bar_width,alpha=opacity,color='r',label='延迟交货的风险')
type2=plt.bar(index+bar_width,count2,bar_width,alpha=opacity,color='y',label='迟交')
plt.xlabel('订单区域')
plt.ylabel('装运数量')
plt.title('所有地区使用的延迟交付产品')
plt.legend()
plt.xticks(index+bar_width,names,rotation=90)
plt.tight_layout()
plt.show()

'''
因此，可以得出结论，对于所有存在延迟交付风险的产品，无论产品实际交付的地区是什么，
为了避免延迟交付，公司可以使用更好的装运方法更快地装运产品，或者安排更多的装运天数，
以便客户提前知道产品何时到达。这将是有趣的，看看迟交订单的数量不同类型的装运方式在所有地区。
'''


#使用标准类装运筛选延迟交货订单
xyz1 = data[(data['Delivery Status'] == 'Late delivery') & (data['Shipping Mode'] == 'Standard Class')]
#使用头等舱装运筛选延迟交货订单
xyz2 = data[(data['Delivery Status'] == 'Late delivery') & (data['Shipping Mode'] == 'First Class')]
#使用二级装运筛选延迟交货订单
xyz3 = data[(data['Delivery Status'] == 'Late delivery') & (data['Shipping Mode'] == 'Second Class')]
#过滤当天发货的延迟交货订单
xyz4 = data[(data['Delivery Status'] == 'Late delivery') & (data['Shipping Mode'] == 'Same Day')]
#计算总值
count1=xyz1['Order Region'].value_counts()
count2=xyz2['Order Region'].value_counts()
count3=xyz3['Order Region'].value_counts()
count4=xyz4['Order Region'].value_counts()
#索引名称
names=data['Order Region'].value_counts().keys()
n_groups=23
fig,ax = plt.subplots(figsize=(20,8))
index=np.arange(n_groups)
bar_width=0.2
opacity=0.6
type1=plt.bar(index,count1,bar_width,alpha=opacity,color='b',label='标准等级')
type2=plt.bar(index+bar_width,count2,bar_width,alpha=opacity,color='r',label='头等舱')
type3=plt.bar(index+bar_width+bar_width,count3,bar_width,alpha=opacity,color='g',label='二等舱')
type4=plt.bar(index+bar_width+bar_width+bar_width,count4,bar_width,alpha=opacity,color='y',label='同一天')
plt.xlabel('订单区域')
plt.ylabel('装运数量')
plt.title('各地区采用不同类型的运输方式')
plt.legend()
plt.xticks(index+bar_width,names,rotation=90)
plt.tight_layout()
plt.show()


'''
************************************************************************
************************************************************************
************************************************************************
************************************************************************
'''

'''
了解客户需求，根据客户需求确定特定的客户群，
是供应链企业增加客户数量、获取更多利润的一种途径，
由于数据集中已有客户的采购历史，因此可以利用RFM分析进行客户细分。
利用数值来显示客户最近度、频率和货币价值
'''

#计算每个订单的总价
data['TotalPrice'] = data['Order Item Quantity'] * data['Order Item Total']# 项目价格*订单数量

data['order date (DateOrders)'].max() # 计算最后一个订单何时来检查最近情况

'''
数据集中的最后一个订单是在2018年1月31日完成的。因此当前时间的设置略高于最后一个订单时间，以提高最近值的准确性。
'''

#现在的日期定在最后一个订单的第二天。i、 2018年2月1日
present = dt.datetime(2018,2,1)
data['order date (DateOrders)'] = pd.to_datetime(data['order date (DateOrders)'])


# 将所有值分组到名为customer segmentation的新数据框架中
Customer_seg = data.groupby('Order Customer Id').\
    agg({'order date (DateOrders)': lambda x: (present - x.max()).days,
         'Order Id': lambda x: len(x), 'TotalPrice': lambda x: x.sum()})
#将订单日期更改为int格式
Customer_seg['order date (DateOrders)'] = Customer_seg['order date (DateOrders)'].astype(int)
# 将列值重命名巍为 R_Value,F_Value,M_Value
Customer_seg.rename(columns={'order date (DateOrders)': 'R_Value',
                             'Order Id': 'F_Value',
                             'TotalPrice': 'M_Value'}, inplace=True)
Customer_seg.head()

'''
R_Value（receignity）表示客户上次订购后所用时间。
F_Value（Frequency）表示客户订购的次数。
M_Value（Monetary value）告诉我们客户花了多少钱购买物品。
'''


plt.figure(figsize=(12,10)) # Figure size
plt.subplot(3, 1, 1)
sns.distplot(Customer_seg['R_Value'])# 画出R_Value值的曲线分布
plt.subplot(3, 1, 2)
sns.distplot(Customer_seg['F_Value'])# 画出F_Value值的曲线分布
plt.subplot(3, 1, 3)
sns.distplot(Customer_seg['M_Value'])# 画出M_Value值的曲线分布
plt.show()

quantiles = Customer_seg.quantile(q=[0.25,0.5,0.75]) #Dividing RFM data into four quartiles
quantiles = quantiles.to_dict()

'''
总数据分为4个分位数。R_Value值应较低，因为它表示最近的客户活动和F_Value值，
M_Value值应较高，因为它们表示购买的频率和总价值。函数定义为将分位数表示为数值。
'''


# R_Value分数应为最小值，因此第一个分位数设置为1。
def R_Score(a,b,c):
    if a <= c[b][0.25]:
        return 1
    elif a <= c[b][0.50]:
        return 2
    elif a <= c[b][0.75]:
        return 3
    else:
        return 4
# F_Value得分越高，M_Score得分越好，因此第一块设置为4。
def FM_Score(x,y,z):
    if x <= z[y][0.25]:
        return 4
    elif x <= z[y][0.50]:
        return 3
    elif x <= z[y][0.75]:
        return 2
    else:
        return 1


# 新的R_Value分数列表示1到4之间的数值分数。
Customer_seg['R_Score'] = Customer_seg['R_Value'].apply(R_Score, args=('R_Value',quantiles))
# 新的F_Value分数列表示1到4之间的数值分数。
Customer_seg['F_Score'] = Customer_seg['F_Value'].apply(FM_Score, args=('F_Value',quantiles))
# 新的M_Value分数列表示1到4之间的数值分数。
Customer_seg['M_Score'] = Customer_seg['M_Value'].apply(FM_Score, args=('M_Value',quantiles))
Customer_seg.head()


#将R、F、M分数添加到一个新列
Customer_seg['RFM_Score'] = Customer_seg.R_Score.astype(str)+ Customer_seg.F_Score.astype(str) + Customer_seg.M_Score.astype(str)
Customer_seg.head()

# 使用.unique（）和len方法可以找到总共有多少不同的客户细分。
count=Customer_seg['RFM_Score'].unique()
print(count)# Printing all Unique values
len(count)# Total count

# 可以看出，有33个不同的客户群。为了便于分割，将R、F、M分数相加
# Calculate RFM_Score
Customer_seg['RFM_Total_Score'] = Customer_seg[['R_Score','F_Score','M_Score']].sum(axis=1)
Customer_seg['RFM_Total_Score'].unique()
# 客户细分共有9个值，每个值分别指定了相应的名称。

# 定义rfm级别函数
def RFM_Total_Score(df):
    if (df['RFM_Total_Score'] >= 11):# RFM得分为11,12
        return 'Champions'
    elif (df['RFM_Total_Score'] == 10):# RFM得分为10
        return 'Loyal Customers'
    elif (df['RFM_Total_Score'] == 9): # RFM得分为9
        return 'Recent Customers'
    elif (df['RFM_Total_Score'] == 8): # RFM得分为8
        return 'Promising'
    elif (df['RFM_Total_Score'] == 7): # RFM得分为7
        return 'Customers Needing Attention'
    elif (df['RFM_Total_Score'] == 6): # RFM得分为6
        return 'Cant lose them'
    elif (df['RFM_Total_Score'] == 5): # RFM得分为5
        return 'At Risk'
    else:                               # 值小于5的RFM得分

        return 'Lost'
# 创建新变量RFM级别
Customer_seg['Customer_Segmentation'] =Customer_seg.apply(RFM_Total_Score, axis=1)
# 在控制台上打印前5行的标题
Customer_seg.head()


# 每个细分市场有多少客户？
# 计算每个RFM级别的平均值，并返回每个段的大小
Customer_seg['Customer_Segmentation'].value_counts().plot.pie(figsize=(10,10),
                                                              startangle=135, explode=(0,0,0,0.1,0,0,0,0),autopct='%.1f',shadow=True)
plt.title("Customer Segmentation",size=15)
plt.ylabel(" ")
plt.axis('equal')
plt.show()
'''
由于客户总数被划分为9个细分市场，可以看出，
11.4%的客户有失去客户的风险，11%的客户需要关注，
否则最终会失去客户，可以看出，4.4%的客户已经失去。
'''

# 我们的前十大客户中，有一段时间没有购买任何东西的客户数量最多
churned=Customer_seg[(Customer_seg['RFM_Score']=='411')].sort_values('M_Value', ascending=False).head(10)
churned

'''
这些客户过去经常下大量订单，但他们几乎一年都没有下订单，
这意味着他们从其他公司采购。这些人应该成为收购的目标，以获得回报。
'''

# 十大新的最佳客户谁往往下昂贵的订单。
#R_Score分数应尽可能低，F_Score分数、M_Score分数应尽可能高
Customer_seg[(Customer_seg['RFM_Score']=='144')|(Customer_seg['RFM_Score']=='143')].sort_values('M_Value', ascending=False).head(10)

'''
上述客户有潜力成为最佳客户这类人应该成为他们的目标，
将他们转化为忠诚的客户。
所有这些不同的客户群体都应该有不同的针对性广告和奖励，
以增加利润和提高客户的响应能力。
'''

'''
通过对公司数据集的分析发现，西欧和中美洲都是销售额最高的地区，
但公司仅从这些地区损失的收入也最多。而这两个地区被怀疑的欺诈交易和订单数量最多，
延迟交货的情况也更多。在2017年第3季度之前，公司的总销售额一直保持稳定，
每个季度总销售额增长10%，然后在2018年第1季度突然下降近65%。
10月和11月是全年销售额最多的月份。大多数人喜欢通过借记卡支付，
所有欺诈交易都是通过电汇进行的，因此，当客户使用电汇时，公司应小心，
因为公司被一个客户骗走了10万多英镑。所有有延迟交货风险的订单每次都会延迟交货。
大多数鞋钉、男鞋和女装类产品的订单都会导致延迟交货，而且这些产品的欺诈嫌疑也最大。
为欺诈检测而训练的神经网络分类器模型的性能优于所有机器学习分类器模型，f1得分为0.96。
与其他分类机器学习模型相比，决策树模型在识别延迟交货的订单和发现欺诈交易方面做得很好，
f1得分为0.80。对于回归类型的数据，线性回归模型对销售收入的预测效果更好，
而随机森林和极端梯度推进回归模型对需求的预测更为准确，MAE和RMSE得分低于神经网络模型。
但是神经网络回归模型的MAE、RMSE得分与这些ML模型之间的差异非常小。
随机森林和极端梯度增强模型的性能优于神经网络模型。为了进一步研究，
可以将所有的机器学习模型与不同的数据集进行比较，以确定同一个机器学习模型的性能是否更好，
并且可以通过超参数调整来提高这些机器学习模型的性能。
'''
