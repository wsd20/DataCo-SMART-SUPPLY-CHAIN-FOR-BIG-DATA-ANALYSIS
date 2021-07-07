# 导入所有需要安装的库
import pandas as pd
import numpy as np
import xgboost as xgb
import datetime as dt
import warnings
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import svm,tree,preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score,recall_score,confusion_matrix,f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#隐藏警告
warnings.filterwarnings('ignore')

#导入数据集
dataset=pd.read_csv("./大作业/archive/DataCoSupplyChainDataset.csv",header= 0,encoding= 'unicode_escape')
dataset.head(5)# 查看数据前5条

# 整个数据集由180519条记录和53列组成
dataset.apply(lambda x: sum(x.isnull())) # 查看缺失值
'''
数据包括Customer Lname(客户名称)、产品描述、订单Zipcode和Customer Zipcode中的一些丢失值，
这些值应该被删除或替换。另外，由于有可能不同的客户可能具有相同的名字或相同的姓氏，因此创建一个新的列，
其中包含“customer full name”，以避免任何歧义。
'''

# 将名字和姓氏添加到一起以创建新列
dataset['Customer Full Name'] = dataset['Customer Fname'].astype(str)+dataset['Customer Lname'].astype(str)


# 为了便于分析，删除一些不重要的列
data=dataset.drop(['Customer Email','Product Status','Customer Password',
                   'Customer Street','Customer Fname','Customer Lname',
                   'Latitude','Longitude','Product Description','Product Image',
                   'Order Zipcode','shipping date (DateOrders)'],axis=1)

# Customer Zipcode列中缺少3个值。由于缺少的值只是邮政编码，
# 并不十分重要，因此在继续进行数据分析之前，这些值将被替换为零
data['Customer Zipcode']=data['Customer Zipcode'].fillna(0)#使用0来填充缺失值

'''
可以看出，价格与销售额呈线性关系。哪个季度的销售额最高
将订货时间分为年、月、周、日、小时，可以更好地观察趋势。
'''

# pd.DatetimeIndex()：生成时间index
data['order_year']= pd.DatetimeIndex(data['order date (DateOrders)']).year
data['order_month'] = pd.DatetimeIndex(data['order date (DateOrders)']).month
data['order_week_day'] = pd.DatetimeIndex(data['order date (DateOrders)']).weekday
data['order_hour'] = pd.DatetimeIndex(data['order date (DateOrders)']).hour
data['order_month_year'] = pd.to_datetime(data['order date (DateOrders)']).dt.to_period('M')

#计算每个订单的总价
data['TotalPrice'] = data['Order Item Quantity'] * data['Order Item Total']# 项目价格*订单数量

data['order date (DateOrders)'].max() # 计算最后一个订单何时来检查最近情况

'''
数据集中的最后一个订单是在2018年1月31日完成的。因此当前时间的设置略高于最后一个订单时间，以提高最近值的准确性。
'''

#现在的日期定在最后一个订单的第二天。i、 2018年2月1日
present = dt.datetime(2018,2,1)
data['order date (DateOrders)'] = pd.to_datetime(data['order date (DateOrders)'])

# 生成数据副本
train_data=data.copy()

'''
为涉嫌欺诈和延迟交货的订单创建了两个新的列，
使它们成为二分类，这反过来有助于更好地衡量不同模型的性能。
'''
# np.where(condition, x, y)
# 满足条件(condition)，输出x，不满足输出y。
train_data['fraud'] = np.where(train_data['Order Status'] == 'SUSPECTED_FRAUD', 1, 0)
train_data['late_delivery']=np.where(train_data['Delivery Status'] == 'Late delivery', 1, 0)


'''
现在为了精确测量分类模型，所有具有重复值的列都会像延迟交货风险列一样被删除，
因为所有具有延迟交货风险的产品都是延迟交货的。
因为创建了一个新的欺诈检测列，将这些列值交给机器学习模型预测输出
'''

#删除具有重复值的列
train_data.drop(['Delivery Status','Late_delivery_risk',
                 'Order Status','order_month_year',
                 'order date (DateOrders)'], axis=1, inplace=True)


# 检查数据中变量的类型，将非数值类型转变为数值类型
train_data.dtypes

'''
sklearn.preprocessing.LabelEncoder
标准化标签，将标签值统一转换成range(标签值个数-1)范围内
'''
# 创建Labelencoder对象
le = preprocessing.LabelEncoder()
# 将分类列转换为数字
train_data['Customer Country']  = le.fit_transform(train_data['Customer Country'])
train_data['Market']            = le.fit_transform(train_data['Market'])
train_data['Type']              = le.fit_transform(train_data['Type'])
train_data['Product Name']      = le.fit_transform(train_data['Product Name'])
train_data['Customer Segment']  = le.fit_transform(train_data['Customer Segment'])
train_data['Customer State']    = le.fit_transform(train_data['Customer State'])
train_data['Order Region']      = le.fit_transform(train_data['Order Region'])
train_data['Order City']        = le.fit_transform(train_data['Order City'])
train_data['Category Name']     = le.fit_transform(train_data['Category Name'])
train_data['Customer City']     = le.fit_transform(train_data['Customer City'])
train_data['Department Name']   = le.fit_transform(train_data['Department Name'])
train_data['Order State']       = le.fit_transform(train_data['Order State'])
train_data['Shipping Mode']     = le.fit_transform(train_data['Shipping Mode'])
train_data['order_week_day']    = le.fit_transform(train_data['order_week_day'])
train_data['Order Country']     = le.fit_transform(train_data['Order Country'])
train_data['Customer Full Name']= le.fit_transform(train_data['Customer Full Name'])

# 显示初始记录
train_data.dtypes
train_data.head()

'''
现在所有的数据都转换成int类型。将数据集分解为训练数据和测试数据，
利用训练数据对模型进行训练，并利用测试数据对模型的性能进行评价。
'''

# 构建训练集和测试集

#除了欺诈列的所有列
xf=train_data.loc[:, train_data.columns != 'fraud']
#只有欺诈列
yf=train_data['fraud']
#将数据分成两部分，其中80%的数据用于模型训练，20%的数据用于测试
xf_train, xf_test,yf_train,yf_test = train_test_split(xf,yf,test_size = 0.2,random_state = 42)

#除了延迟交货列的所有列
xl=train_data.loc[:, train_data.columns != 'late_delivery']
#只有延迟交货列
yl=train_data['late_delivery']
#将数据分成两部分，其中80%的数据用于模型训练，20%的数据用于测试
xl_train, xl_test,yl_train,yl_test = train_test_split(xl,yl,test_size = 0.2, random_state = 42)

# 由于有这么多不同的变量具有不同的范围，
# 因此在使用机器学习训练数据之前，
# 使用标准定标器对所有数据进行标准化，使其内部一致。
sc = StandardScaler()
xf_train=sc.fit_transform(xf_train)
xf_test=sc.transform(xf_test)
xl_train=sc.fit_transform(xl_train)
xl_test=sc.transform(xl_test)


'''
这些数据现在已经可以用于机器学习模型了.但是由于比较了许多不同的模型，
从一开始训练每个模型都很复杂，因此定义了一个函数，使过程变得简单。
输出是二进制分类格式，因此所有模型都用准确度得分、召回率得分和F1分数度量。
为了衡量不同模型的性能，采用F1分数作为主要指标，
因为F1评分是精确性评分和回调性评分的调和平均值，所有分数都乘以100，以便更好地理解
'''

# 计算召回率。
# 召回是比其中为真阳性的数量和假阴性的数量。召回率直观地是分类器找到所有正样本的能力。tp / (tp + fn)tpfn
# 最佳值为 1，最差值为 0。

#F1分数（F1-score）是分类问题的一个衡量指标。
# 一些多分类问题的机器学习竞赛，常常将F1-score作为最终测评的方法。
# 它是精确率和召回率的调和平均数，最大为1，最小为0。

accuracy_f_l = []
recall_f_l = []
f1_f_l = []

accuracy_l_l = []
recall_l_l = []
f1_l_l = []

def classifiermodel(model_f,model_l,xf_train, xf_test,yf_train,yf_test,xl_train, xl_test,yl_train,yl_test):
    model_f=model_f.fit(xf_train,yf_train) # 用于欺诈检测的训练数据训练
    model_l=model_l.fit(xl_train,yl_train) # 用于预测延迟交货训练数据的训练
    yf_pred=model_f.predict(xf_test)
    yl_pred=model_l.predict(xl_test)
    accuracy_f=accuracy_score(yf_pred, yf_test) #欺诈检测的准确性
    accuracy_l=accuracy_score(yl_pred, yl_test) #延迟交货预测的准确性
    recall_f=recall_score(yf_pred, yf_test) #欺诈检测召回率评分
    recall_l=recall_score(yl_pred, yl_test)# 迟交预判召回率评分

    # confusion_matrix: 矩阵为 把延迟交货预测为延迟交货的 把非延迟交货预测为延迟交货的
    #                         把延迟交货预测为非延迟交货的 把非延迟交货预测为延迟交货的
    conf_f=confusion_matrix(yf_test, yf_pred)# 欺诈检测
    conf_l=confusion_matrix(yl_test, yl_pred)#延迟交货的预测
    f1_f=f1_score(yf_test, yf_pred)#欺诈检测
    f1_l=f1_score(yl_test, yl_pred)#延迟交货的检测
    print('使用的模型参数为 :',model_f)
    print('预测欺诈的准确性        :', (accuracy_f)*100,'%')
    print('预测欺诈的召回率评分为        :', (recall_f)*100,'%')
    print('预测欺诈的的配置矩阵为        :\n',  (conf_f))
    print('F1预测欺诈评分为        :', (f1_f)*100,'%')
    print('预测延迟交货的准确度为:', (accuracy_l)*100,'%')
    print('预测延迟交货的召回率评分为:', (recall_l)*100,'%')
    print('预测延迟交货的配置矩阵为: \n',(conf_l))
    print('预测延迟交货F1评分为:', (f1_l)*100,'%')
    accuracy_f_l.append(accuracy_f)
    recall_f_l.append(recall_f)
    f1_f_l.append(f1_f)
    accuracy_l_l.append(accuracy_l)
    recall_l_l.append(recall_l)
    f1_l_l.append(f1_l)

'''
优化算法选择参数：solver
　　　　solver参数决定了我们对逻辑回归损失函数的优化方法，有4种算法可以选择，分别是：
　　　　a) liblinear：使用了开源的liblinear库实现，内部使用了坐标轴下降法来迭代优化损失函数。
　　　　b) lbfgs：拟牛顿法的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
　　　　c) newton-cg：也是牛顿法家族的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
　　　　d) sag：即随机平均梯度下降，是梯度下降法的变种，和普通梯度下降法的区别是每次迭代仅仅用一部分的样本来计算梯度，
      适合于样本数据多的时候，SAG是一种线性收敛算法，这个速度远比SGD快。
'''
#　Logistic逻辑回归分类模型
model_f = LogisticRegression(solver='lbfgs') #分类模型
model_l = LogisticRegression(solver='lbfgs') #分类模型
#为定义的函数提供输入
classifiermodel(model_f,model_l,xf_train, xf_test,yf_train,yf_test,xl_train, xl_test,yl_train,yl_test)
# 高斯朴素贝叶斯分类模型
model_f = GaussianNB()
model_l = GaussianNB()
classifiermodel(model_f,model_l,xf_train, xf_test,yf_train,yf_test,xl_train, xl_test,yl_train,yl_test)

# 支持向量机分类
model_f = svm.LinearSVC()
model_l = svm.LinearSVC()
classifiermodel(model_f,model_l,xf_train, xf_test,yf_train,yf_test,xl_train, xl_test,yl_train,yl_test)


# K近邻分类模型
# n_neighbors： 选择最邻近点的数目k
model_f = KNeighborsClassifier(n_neighbors=1)
model_l = KNeighborsClassifier(n_neighbors=1)
classifiermodel(model_f,model_l,xf_train, xf_test,yf_train,yf_test,xl_train, xl_test,yl_train,yl_test)

# 线性分类
model_f = LinearDiscriminantAnalysis()
model_l = LinearDiscriminantAnalysis()
classifiermodel(model_f,model_l,xf_train, xf_test,yf_train,yf_test,xl_train, xl_test,yl_train,yl_test)


# 随机森林分类
model_f = RandomForestClassifier()
model_l = RandomForestClassifier()
classifiermodel(model_f,model_l,xf_train, xf_test,yf_train,yf_test,xl_train, xl_test,yl_train,yl_test)


# Extra trees(极端随机树分类模型)
model_f = ExtraTreesClassifier()
model_l = ExtraTreesClassifier()
classifiermodel(model_f,model_l,xf_train, xf_test,yf_train,yf_test,xl_train, xl_test,yl_train,yl_test)

# 极端梯度推进分类
model_f = xgb.XGBClassifier()
model_l = xgb.XGBClassifier()
classifiermodel(model_f,model_l,xf_train, xf_test,yf_train,yf_test,xl_train, xl_test,yl_train,yl_test)

# 决策树分类
model_f = tree.DecisionTreeClassifier()
model_l = tree.DecisionTreeClassifier()
classifiermodel(model_f,model_l,xf_train, xf_test,yf_train,yf_test,xl_train, xl_test,yl_train,yl_test)

# 为了更好地理解和比较所有分数，创建了一个DataFrame
ls = ['Logistic逻辑回归分类模型','高斯朴素贝叶斯分类模型','支持向量机分类','K近邻分类模型','线性分类','随机森林分类','极端随机树分类模型','极端梯度推进分类','决策树分类']
classification_data = {
                       '分类模型': ls,
                       '欺诈预测准确度评分':     accuracy_f_l,
                       '预测欺诈的召回率评分':       recall_f_l,
                       'F1预测欺诈评分':           f1_f_l,
                       '预测延迟交货的准确度':       accuracy_l_l,
                       '预测延迟交货的召回率评分':         recall_l_l,
                       '预测延迟交货F1评分':             f1_l_l }
#使用列名创建DataFrame
classification_comparision = pd.DataFrame (classification_data,
                                           columns = ['分类模型','欺诈预测准确度评分','预测欺诈的召回率评分','F1预测欺诈评分',
                                        '预测延迟交货的准确度','预测延迟交货的召回率评分','预测延迟交货F1评分'])



'''
考虑到F1评分，很明显，决策树分类模型在分类类型上表现更好，
F1评分对于欺诈检测几乎为80%，对于延迟交货则为99.42%，
所有模型都期望gussian模型能以几乎98%的准确率预测订单的延迟交货，
为了确保模型预测的正确性，对模型进行了交叉验证，并将结果与模型的准确度进行了比较。
'''

'''什么是交叉验证'''
#比如说我们将数据集分为10折，做一次交叉验证，
# 实际上它是计算了十次，将每一折都当做一次测试集，
# 其余九折当做训练集，这样循环十次。通过传入的模型，
# 训练十次，最后将十次结果求平均值。将每个数据集都算一次

#定义交叉验证
def cross_validation_model(model_f,model_l,xf,yf,xl,yl):
    model_f= model_f.fit(xf,yf)
    model_l = model_l.fit(xl,yl)
    # 计算6次
    scores_f = cross_val_score(model_f, xf, yf, cv=6)
    scores_l = cross_val_score(model_l, xl, yl, cv=6)
    print('所用模型为',model_f)
    print('欺诈预测的交叉验证准确性: %0.2f (+/- %0.2f)' % (scores_f.mean(), scores_f.std() * 2))
    print('延迟交货的交叉验证准确性: %0.2f (+/- %0.2f)' % (scores_l.mean(), scores_l.std() * 2))

cross_validation_model(model_f,model_l,xf,yf,xl,yl)

'''
由于模型的交叉验证得分和准确度得分之间的差异非常小，
因此可以确认数据既不是过度拟合也不是欠拟合，
'''