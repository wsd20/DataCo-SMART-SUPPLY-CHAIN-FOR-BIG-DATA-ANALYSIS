# 导入所有需要安装的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#导入数据集
dataset=pd.read_csv("./大作业/archive/DataCoSupplyChainDataset.csv",header= 0,encoding= 'unicode_escape')
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
dataset['Customer Full Name'] = dataset['Customer Fname'].\
                                    astype(str)+dataset['Customer Lname'].astype(str)

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
sns.heatmap(data.corr(),annot=True,linewidths=.5,fmt='.1g',cmap= 'coolwarm') # 相关矩阵热图展示
plt.show()

'''
可以看到，产品价格与销售、订单项目总数有着很高的相关性。
由于分析中使用的数据与供应链相关，因此找到哪个地区的销售最多是有意义的？
可以使用groupby方法找到，该方法将相似的市场区域分开，并使用“sum”函数添加该特定区域的所有销售。
'''

plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
market = data.groupby('Market') #通过Market聚合
region = data.groupby('Order Region')
plt.figure(1)
market['Sales per customer'].sum().sort_values(ascending=False).\
    plot.bar(figsize=(12,6), title="所有市场的总销售额")
plt.xticks(rotation=45)
plt.show()
plt.figure(2)
region['Sales per customer'].sum().sort_values(ascending=False).\
    plot.bar(figsize=(12,8), title="所有地区的总销售额")
plt.xticks(rotation=60)
plt.show()


'''
从图表中可以看出，欧洲市场的销售额最多，
而非洲市场的销售额最少，在这些市场中，
西欧地区和中美洲的销售额最高。
'''




'''
价格与销售额呈线性关系。哪个季度的销售额最高
将订货时间分为年、月、周、日、小时，可以更好地观察趋势。
'''
data['order_year']= pd.DatetimeIndex(data['order date (DateOrders)']).year
data['order_month'] = pd.DatetimeIndex(data['order date (DateOrders)']).month
data['order_week_day'] = pd.DatetimeIndex(data['order date (DateOrders)']).weekday
data['order_hour'] = pd.DatetimeIndex(data['order date (DateOrders)']).hour
data['order_month_year'] = pd.to_datetime(data['order date (DateOrders)']).dt.to_period('M')



'''
价格是如何影响销售的，什么时候，哪些产品有更多的销售，
订单数量最多的是10月份，其次是11月份，所有其他月份的订单都是一致的。
2017年客户的订单数量最多。周六的平均销量最高，周三的销量最少。
无论时间长短，平均销售额始终保持一致，标准差为3。
同样重要的是要知道什么样的支付方式是首选的人购买所有这些产品在所有地区？
可以使用.unique（）方法查看不同的付款方式。
'''


data['Type'].unique()
# 支付方式为'DEBIT', 'TRANSFER', 'CASH', 'PAYMENT'('借记'、'转账'、'现金'、'付款')
# 研究发现有四种支付方式

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
index=np.arange(n_groups)   # 下标范围
bar_width=0.2 # 柱宽
opacity=0.6   # 透明度
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
有些产品在每个订单上都有负收益，这表明订单给公司带来了收入损失。这些是什么产品？
'''
# 订单的收益存在正负
data['Benefit per order']

loss = data[(data['Benefit per order']<0)]
#显示损失最大的十大产品
# nlargest(10) 选择所有列中具有最大值的前10行
plt.figure(1)
loss['Category Name'].value_counts().nlargest(10).plot.bar(figsize=(20,8), title="损失最大的产品")
plt.xticks(rotation=0)
plt.show()
plt.figure(2)
loss['Order Region'].value_counts().nlargest(10).plot.bar(figsize=(20,8), title="损失最大的地区")
#损失的总销售额的总和
plt.xticks(rotation=0)
plt.show()
print('订单损失总收入',loss['Benefit per order'].sum())

'''
损失总额约390万，这是一个巨大的数额。可以看出，鞋类是损失销售最多的类别，其次是男式鞋。
大多数损失销售发生在中美洲和西欧地区。这种损失的销售可能是由于涉嫌欺诈或延迟交货而发生的。
找出使用哪种支付方式进行欺诈，可以有助于防止欺诈在未来发生
'''


data['Order Status'].unique()
# 订单存在以下几个状态：
# COMPLETE PENDING CLOSED PENDING_PAYMENT CANCELED PROCESSING SUSPECTED_FRAUD ON_HOLD PAYMENT_REVIEW
# 完成、待定、关闭、待付款、取消、处理、涉嫌欺诈、等待、付款审核

#查看用于进行除转账以外的欺诈的付款类型
xyz = data[(data['Type'] != 'TRANSFER')&(data['Order Status'] == 'SUSPECTED_FRAUD')]
xyz['Order Region'].value_counts()

'''
可以清楚地看到，没有欺诈进行借记，现金，付款方式，
所以所有可疑的欺诈订单都是通过国外电汇方式发出的。
哪个地区和什么产品最容易被怀疑欺诈？
'''


#分离涉嫌欺诈的订单
high_fraud = data[(data['Order Status'] == 'SUSPECTED_FRAUD') & (data['Type'] == 'TRANSFER')]
#绘制有关订单区域的饼图
fraud=high_fraud['Order Region'].value_counts().plot.pie(figsize=(24,12),
 startangle=180, explode=(0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),autopct='%.1f',shadow=True,)
plt.title("欺诈最高的地区",size=15,color='y')
plt.ylabel(" ")
fraud.axis('equal')
plt.show()

'''
可以观察到，涉嫌欺诈的订单数量最多的是西欧，约占订单总数的17.4%，
其次是中美洲，占15.5%。哪种产品被怀疑欺诈最多？
'''


high_fraud1 = data[(data['Order Status'] == 'SUSPECTED_FRAUD')] #
high_fraud2 = data[(data['Order Status'] == 'SUSPECTED_FRAUD') &(data['Order Region'] == 'Western Europe')]
#绘制各地区十大最可疑欺诈产品条形图
fraud1=high_fraud1['Category Name'].value_counts().\
    nlargest(10).plot.bar(figsize=(20,8), title="欺诈类别",color='orange')
#绘制西欧十大最可疑欺诈产品条形图
fraud2=high_fraud2['Category Name'].value_counts().\
    nlargest(10).plot.bar(figsize=(20,8), title="西欧欺诈产品",color='green')
plt.legend(["所有地区", "西欧"])
plt.title("欺诈检测率最高的十大产品", size=15)
plt.xlabel("产品", size=13)
plt.ylim(0,600)
plt.xticks(rotation=0)
plt.show()

'''
鞋钉部门被怀疑欺诈最多，其次是在所有地区和西欧的男鞋。
'''

#过滤出可疑的FRAUD订单
cus = data[(data['Order Status'] == 'SUSPECTED_FRAUD')]
#为十大最可疑欺诈客户绘制条形图
cus['Customer Full Name'].value_counts().nlargest(10).plot.\
    bar(figsize=(20,8), title="十大最可疑欺诈客户")
plt.show()


'''
一个叫玛丽·史密斯的顾客就要为自己的欺诈行为负责528次，
这是非常令人震惊的。查看她到底做了多少欺诈订单
'''

#过滤涉嫌欺诈的玛丽·史密斯的订单
amount = data[(data['Customer Full Name'] == 'MarySmith')&
              (data['Order Status'] == 'SUSPECTED_FRAUD')]
amount
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
因为如果产品不能按时交付，客户将不会满意。查看哪一类产品交货最晚
'''
#筛选具有延迟交付状态的列
late_delivery = data[(data['Delivery Status'] == 'Late delivery')]
#最迟交货的前10个产品
late_delivery['Category Name'].value_counts().nlargest(10).plot.\
    bar(figsize=(20,8), title="最迟交货的前10个产品")
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
type1=plt.bar(index,count1,bar_width,alpha=opacity,color='g',label='延迟交货的风险')
type2=plt.bar(index+bar_width,count2,bar_width,alpha=opacity,color='b',label='延迟交货')
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
以便客户提前知道产品何时到达。接着看看迟交订单的数量不同类型的装运方式在所有地区。
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
type1=plt.bar(index,count1,bar_width,alpha=opacity,color='b',label='标准舱')
type2=plt.bar(index+bar_width,count2,bar_width,alpha=opacity,color='r',label='头等舱')
type3=plt.bar(index+bar_width+bar_width,count3,bar_width,alpha=opacity,color='g',label='二等舱')
type4=plt.bar(index+bar_width+bar_width+bar_width,count4,bar_width,alpha=opacity,color='y',label='当天')
plt.xlabel('订单区域')
plt.ylabel('装运数量')
plt.title('各地区采用不同类型的运输方式')
plt.legend()
plt.xticks(index+bar_width,names,rotation=90)
plt.tight_layout()
plt.show()