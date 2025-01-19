
#导入模块和包
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import nan as NA #允许numpy用NA代替nan作为别名
from pandas import DataFrame,Series
import seaborn as sns
from pyecharts import options as opts
from pyecharts.charts import Bar,Grid,Pie
from pyecharts.globals import CurrentConfig, NotebookType
CurrentConfig.NOTEBOOK_TYPE = NotebookType.JUPYTER_LAB #声明notebook类型

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] #显示中文标签
plt.rcParams['axes.unicode_minus'] = False #显示负号
from IPython.core.interactiveshell import InteractiveShell
#设置下面一条参数等于'all'则会默认在jupyter notebook里面输出每个输出项
#InteractiveShell.ast_node_interactivity = "all"

#随机生成颜色函数
def get_colors(categories):
    cmap = plt.get_cmap(lut = len(categories))
    colors = cmap(np.arange(len(categories)))
    return colors


# 生成柱状图，利用随机生成函数确定颜色，并生成数值标签

fig = plt.figure(figsize = (10,8))
# 创建一个新的图形，设置图形的大小为宽 10 单位、高 8 单位
ax  = plt.gca()
# 获取当前轴对象
sku_brand.plot(kind = 'bar', width = 0.8, alpha = 0.6,color = get_colors(brands),ax = ax)
# 假设 sku_brand 是一个 pandas 的 Series 或 DataFrame，使用 plot 方法绘制柱状图
# kind='bar' 表示绘制柱状图
# width=0.8 表示柱状图的宽度为 0.8
# alpha=0.6 表示柱状图的透明度为 0.6
# color=colors 表示柱状图的颜色，这里 colors 应该是一个颜色列表
# ax=ax 表示将图形绘制在之前获取的轴对象 ax 上
plt.title('各品牌SKU数',fontsize = 12)
# 设置图形的标题，字体大小为 12
plt.ylabel('商品数量',fontsize = 10)
# 设置 y 轴的标签，字体大小为 10

#添加数据标签
#ax.patches 包含了柱状图的每个条
for p in ax.patches:
    # 获取当前条的高度
    height = p.get_height() 
    # 使用 ax.annotate 方法添加注释
    # 标注的内容是条的高度，使用 format 函数将高度转换为字符串
    # xy 是标注的位置，为条的中心位置，x 坐标为条的起始位置加上条宽的一半，y 坐标为条的高度
    # xytext 是文本的偏移量，这里是相对于 xy 位置向上偏移 3 个点
    # textcoords='offset points' 表示 xytext 的坐标单位是点
    # ha='center' 表示水平对齐方式为居中
    # va='bottom' 表示垂直对齐方式为底部
    ax.annotate(
        '{}'.format(height),#标注内容为条高度
        xy = (p.get_x()+ p.get_width()/2, height),
        xytext = (0,3),
        textcoords = 'offset points',
        ha = 'center',
        va = 'bottom'
    )
plt.show()
# 显示图形

# 利用子图生成多个柱状图
#可视化
# 使用 plt.subplots 函数创建一个包含 1 行 2 列的子图布局，设置图形的总大小为宽 18 单位、高 12 单位
# fig 是整个图形对象，axes 是一个包含两个轴对象的数组
fig,axes = plt.subplots(1,2,figsize = (18,15))

# 假设 brand_salecnt 是一个 pandas 的 Series 或 DataFrame，使用 plot 方法绘制水平柱状图
# ax=axes[0] 表示将图形绘制在第一个轴对象上
# kind='barh' 表示绘制水平柱状图
# width=0.6 表示柱状图的宽度为 0.6
# alpha=0.6 表示柱状图的透明度为 0.6
# color=colors 表示使用 colors 函数生成的颜色列表作为柱状图的颜色
brand_salecnt.plot(kind = 'barh', ax = axes[0], width = 0.6,alpha = 0.6,color = get_colors(brands))
# 为第一个轴对象设置标题，字体大小为 12
axes[0].set_title('各品牌总销量',fontsize = 12)
# 为第一个轴对象设置 x 轴标签，字体大小为 10
axes[0].set_xlabel('总销量',fontsize = 10)
for p in axes[0].patches:
    # 获取当前条的宽度（对于水平柱状图，宽度表示数据的值）
    width = p.get_width()
    # 将数据标签的格式以百万为单位显示，使用 '{:.2f}M' 格式化字符串
    label = '{:.2f}M'.format(width / 1e6)
    # 使用 annotate 方法添加数据标签
    # 标注的内容是格式化后的标签
    # xy 是标注的位置，为条的右侧中心位置，x 坐标为条的宽度，y 坐标为条的垂直中心位置
    # xytext 是文本的偏移量，这里是相对于 xy 位置向右偏移 5 个点
    # textcoords='offset points' 表示 xytext 的坐标单位是点
    # ha='left' 表示水平对齐方式为左对齐
    # va='center' 表示垂直对齐方式为居中
    axes[0].annotate(
        label,
        xy=(width, p.get_y() + p.get_height() / 2),
        xytext=(5, 0),
        textcoords='offset points',
        ha='left',
        va='center'
    )


brand_sale_money.plot(kind = 'barh', ax = axes[1], width = 0.6,alpha = 0.6, color = get_colors(brands))
axes[1].set_title('各品牌总销售额',fontsize = 12)
axes[1].set_xlabel('总销售额',fontsize = 10)
for p in axes[1].patches:
    width = p.get_width()
    # 将数据标签的格式以百万为单位显示，使用 '{:.2f}M' 格式化字符串
    label = '{:.2f}M'.format(width / 1e6)
    axes[1].annotate(
        label,
        xy=(width, p.get_y() + p.get_height() / 2),
        xytext=(5, 0),
        textcoords='offset points',
        ha='left',
        va='center'
    )

# 调整子图之间的水平间距为 0.4    
plt.subplots_adjust(wspace = 0.4)
# 显示图形
plt.show()

# 生成柱状图和折线图

fig,ax1 = plt.subplots(1,1,figsize = (12,6))
brand_salecnt1.plot(kind = 'bar',ax = ax1,width = 0.6,alpha = 0.6, color = get_colors(brands))
# 使用 brand_salecnt1 的 plot 方法绘制柱状图，ax=ax1 表示将图形绘制在 ax1 轴对象上
# kind='bar' 表示绘制垂直柱状图
# width=0.6 表示柱状图的宽度为 0.6
# alpha=0.6 表示柱状图的透明度为 0.6
# color=colors表示使用 get_colors 函数生成的颜色列表作为柱状图的颜色

# 设置图形的标题和 x 轴、y 轴标签
ax1.set_title('品牌总销量与销售额',fontsize = 12)
ax1.set_xlabel('品牌',fontsize = 10)
ax1.set_ylabel('总销量',fontsize = 10)
# 为柱状图添加数值标签（以百万为单位）
'''for p in ax1.patches:
    # 获取柱状图条的高度
    height = p.get_height()
    # 将数值转换为百万并保留两位小数
    label = '{:.2f}M'.format(height / 1e6)
    # 使用 annotate 方法添加注释，内容为以百万为单位的标签
    ax1.annotate(label,
                # 注释的位置为条的顶部中心
                xy=(p.get_x() + p.get_width() / 2, height),
                # 注释文本的偏移量
                xytext=(-3, -3),
                textcoords='offset points',
                # 水平对齐方式为居中
                ha='center',
                # 垂直对齐方式为底部
                va='bottom')'''


# 创建一个与 ax1 共享 x 轴的第二个轴对象 ax2
ax2 = ax1.twinx()
ax2.plot(brand_sale_money1.index,brand_sale_money1.values,marker = 'o',color = 'red', linestyle = '-')
'''for i, value in enumerate(brand_sale_money1.values):
    # 将数值转换为百万并保留两位小数
    label = '{:.2f}M'.format(value / 1e6)
    # 使用 annotate 方法添加注释，内容为以百万为单位的标签
    ax2.annotate(label,
                # 注释的位置为数据点的位置
                xy=(i, value),
                # 注释文本的偏移量
                xytext=(5, 6),
                textcoords='offset points',
                # 水平对齐方式为居中
                ha='center',
                # 垂直对齐方式为底部
                va='bottom')'''

# 在 ax2 上绘制品牌总销售额的折线图
ax2.set_ylabel('总销售额',fontsize = 10)
# 使用 ax2.plot 方法绘制折线图
# brand_sale_money1.index 是 x 轴数据
# brand_sale_money1.values 是 y 轴数据
# marker='o' 表示使用圆形标记点
# color='red' 表示折线的颜色为红色
# linestyle='-' 表示使用实线

# 设置图例
handles1,labels1 = ax1.get_legend_handles_labels()
handles2,labels2 = ax2.get_legend_handles_labels()
# 将 ax1 和 ax2 的图例句柄和标签合并，并将图例放置在右上角
ax1.legend(handles1+handles2,labels1 + labels2, loc = 'upper right')
# 将 x 轴刻度标签旋转 45 度
plt.xticks(rotation = 45)
plt.show()

#子图的方式绘制饼图
# 创建一个新的图形，设置图形的大小为宽 14 单位、高 6 单位
plt.figure(figsize = (14,6))
# 创建一个 1 行 2 列的子图布局，并选择第一个子图
plt.subplot(1,2,1)
# 使用 plt.pie 函数绘制第一个饼图
plt.pie(mt_salecnt,labels = mt_salecnt.index,autopct = '%1.1f%%', startangle = 140,colors = get_colors(mt_salecnt))
# 为第一个子图设置标题，字体大小为 12
plt.title('各主类别销量占比',fontsize = 12)
# 选择第二个子图
plt.subplot(1,2,2)
# 使用 plt.pie 函数绘制第二个饼图
plt.pie(mt_salemon,labels = mt_salemon.index,autopct = '%1.1f%%', startangle = 140, colors = get_colors(mt_salemon))
# 为第二个子图设置标题，字体大小为 12
plt.title('各主类别销售额占比',fontsize = 12)
#使用 tight_layout 调整子图布局
plt.tight_layout()
plt.show()

##利用for循环穿件多个柱状图与折线图

# 获取唯一主类别
main_types = brandmt_data['main_type'].unique()
brands = brandmt_data['brand'].unique()
# 使用 plt.subplots 创建一个 1 行 3 列的子图布局，设置图形的大小为宽 24 单位、高 6 单位，不共享 y 轴
fig,axes = plt.subplots(nrows = 1, ncols = 3,figsize = (24,6),sharey = False)

#遍历主类别，为每个主类别绘图
for ax, main_type in zip(axes, main_types):
    #过滤当前主类别数据
     # brandmt_data[brandmt_data['main_type'] == main_type] 筛选出 main_type 等于当前 main_type 的数据
    type_data = brandmt_data[brandmt_data['main_type']== main_type]
    # 按照 sale_count 对数据进行降序排序
    type_data = type_data.sort_values(by ='sale_count',ascending = False)
    # 使用 seaborn 的 barplot 绘制柱状图
    # x='brand' 表示 x 轴使用 brand 列的数据
    # y='sale_count' 表示 y 轴使用 sale_count 列的数据
    # ax=ax 表示将图形绘制在当前的轴对象 ax 上
    # ci=None 表示不显示置信区间
    # alpha=0.6 表示柱状图的透明度为 0.6
    # palette='viridis' 表示使用 viridis 颜色映射
    sns.barplot(x = 'brand',y = 'sale_count', data = type_data, ax = ax, ci = None, alpha = 0.6, palette = 'viridis')
    # 创建一个与 ax 共享 x 轴的第二个轴对象 ax2
    ax2 = ax.twinx()
    # 使用 seaborn 的 lineplot 绘制折线图
    # x='brand' 表示 x 轴使用 brand 列的数据
    # y='sale_money' 表示 y 轴使用 sale_money 列的数据
    # ax=ax2 表示将图形绘制在 ax2 上
    # ci=None 表示不显示置信区间
    # color='red' 表示折线图的颜色为红色
    # marker='o' 表示使用圆形标记点
    # linestyle='-' 表示使用实线
    sns.lineplot(x = 'brand', y = 'sale_money', data = type_data, ax = ax2, ci = None, color = 'red', marker = 'o', linestyle = '-')

    #折线图标签
    '''for i,(x,y) in enumerate(zip(type_data['brand'],type_data['sale_money'])):
        # 在每个数据点的位置添加文本标签
        # x=i 表示 x 轴位置为数据点的索引
        # y=y 表示 y 轴位置为数据点的 y 值
        # s=f'{y:.2f}' 表示标签的内容为 y 值保留两位小数
        # ha='right' 表示水平对齐方式为右对齐
        # va='bottom' 表示垂直对齐方式为底部对齐
        # fontsize=8 表示字体大小为 8
        # color='red' 表示字体颜色为红色
      
         ax2.text(x=i, y=y, s=f'{y:.2f}', ha='right', va='bottom', fontsize=8, color='red')'''
    # 设置 x 轴刻度标签的旋转角度为 45 度
    ax.tick_params(axis= 'x',rotation = 45)
    #设置图例
    #设置子图的标题
    ax.set_title(f'各品牌{main_type}类别销量、销售额',fontsize = 12)
    # 设置 x 轴的标签
    ax.set_xlabel('品牌',fontsize = 8)
    # 设置第一个 y 轴（柱状图）的标签
    ax.set_ylabel('销量',fontsize = 10)
    # 设置第二个 y 轴（折线图）的标签
    ax2.set_ylabel('销售额',fontsize = 10)
# 调整子图之间的水平间距为 0.3
plt.subplots_adjust(wspace = 0.3)
plt.show()

#利用for循环创建多行多列的子图
# 创建一个 7 行 2 列的子图布局，设置图形大小为宽 16 单位、高 54 单位
fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(16, 54))

#遍历子类别，并绘图
#axes.flat，将二维数组展平为一维数组
# 遍历子类别，并绘图
for ax, sub_type in zip(axes.flat, sub_types):
    # 筛选出当前子类别数据并排序
    type_data = brandst_data[brandst_data['sub_type'] == sub_type]
    type_data = type_data.sort_values(by='sale_count', ascending=False)
     # 使用 seaborn 的 barplot 绘制柱状图
    sns.barplot(x='brand',
                y='sale_count',
                data=type_data,
                ax=ax,
                ci=None,
                alpha=0.6,
                palette='viridis')
     # 创建与 ax 共享 x 轴的 ax2 轴对象
    ax2 = ax.twinx()
    sns.lineplot(x='brand',
                 y='sale_money',
                 data=type_data,
                 ax=ax2,
                 ci=None,
                 color='red',
                 marker='o',
                 linestyle='-')
    # 以下是添加折线图数据标签的代码，目前处于注释状态
    #for i, (brand, sale_money) in enumerate(
            #zip(type_data['brand'], type_data['sale_money'])):
        #ax2.text(i,
                # sale_money,
                # f'{sale_money:.2f}',
                # ha='right',
                # va='bottom',
                # fontsize=8,
                # color='red')
     # 设置 x 轴刻度标签旋转角度
    ax.tick_params(axis='x', rotation=45)
    # 设置子图标题
    ax.set_title(f'各品牌{sub_type}子类别销量、销售额', fontsize=12)
    # 设置左侧 y 轴（柱状图）的标签
    ax.set_ylabel('销量', fontsize=10)
    # 设置右侧 y 轴（折线图）的 label
    ax2.set_ylabel('销售额', fontsize=10)
# 调整子图之间的水平间距
plt.subplots_adjust(wspace=0.4)
plt.show()


##堆叠条形图

'''transform('sum') 对分组后的每个组内的 sale_money 列进行求和操作，
并将结果广播到原 DataFrame 中的每一行。也就是说，
对于每个组，计算 sale_money 的总和，
然后将这个总和赋值给该组内每一行的新列 money_total。'''
brandmt_data['count_total'] = brandmt_data.groupby('brand')['sale_count'].transform('sum')
brandmt_data['money_total'] = brandmt_data.groupby('brand')['sale_money'].transform('sum')

#计算各品牌内部主类别相对占比,axis = 1 按行操作，axis = 0 按列操作 
brandmt_data['count_percent'] = brandmt_data.apply(lambda row: row['sale_count']/row['count_total']*100 if row['count_total'] != 0 else 0, axis = 1)
brandmt_data['money_percent'] = brandmt_data.apply(lambda row: row['sale_money']/row['money_total']*100 if row['money_total'] != 0 else 0, axis = 1)
#brandmt_data
#颜色映射定义
maincolor_map = {
    '其他':'#70DB93',
    '化妆品':'#2980B9',
    '护肤品':'#7F8C8D'
}

#品牌相对主类别数据占比可视化
plt.figure(figsize = (18,10))
# 创建一个与品牌数量相同长度的数组，用于表示 x 轴的位置
x_positions = np.arange(len(brands))
# 柱状图的宽度
bar_width = 0.5

# 遍历品牌
for idx,brand in enumerate(brands):
    # 筛选出品牌等于当前品牌的数据，并重置索引
    brand_data = brandmt_data[brandmt_data['brand']== brand].reset_index()
    # 按照 main_type 对数据进行排序
    brand_data = brand_data.sort_values(by = 'main_type')
    # 用于存储每个堆叠柱状图的底部位置
    bottoms = np.zeros(len(brand_data))
    
    # 遍历品牌数据的每一行
    for i,row in brand_data.iterrows():
        # 绘制堆叠柱状图
        # x_positions[idx] 表示当前品牌的 x 轴位置
        # row['count_percent'] 表示当前行的 count_percent 列的值，即柱状图的高度
        # width=bar_width 表示柱状图的宽度
        # bottom=bottoms[i] 表示当前柱状图的底部位置
        # color=maincolor_map.get(row['main_type'], 'black') 根据 main_type 从 maincolor_map 中获取颜色，如果不存在则使用黑色
        plt.bar(x_positions[idx],row['count_percent'],width = bar_width , bottom = bottoms[i],color = maincolor_map.get(row['main_type'],'black'))
        # 更新 bottoms 数组，用于下一个柱状图的底部位置
        bottoms[i+1:] += row['count_percent']

# 设置 x 轴刻度标签，使用品牌名称，并将标签旋转 45 度
plt.xticks(x_positions,brands, rotation = 45)
# 创建图例的手柄和标签
handles = [plt.Rectangle((0,0),1,1,color = maincolor_map[main_type]) for main_type in maincolor_map.keys()]
labels = maincolor_map.keys()

# 添加图例
plt.legend(handles, labels, title = '主类别')

# 设置标题和轴标签
plt.title('各品牌主类别销量相对占比图',fontsize = 12)
plt.xlabel('品牌',fontsize = 10)
plt.ylabel('销量占比(%)')
plt.show()


## 散点图

x = round(data.groupby('brand')['sale_count'].agg('mean'),2) #各品牌销量均值

y = round(data.groupby('brand')['comment_count'].agg('mean'),2) # 各品牌评论数均值

s = round(data.groupby('brand')['price'].agg('mean') ,2) #各品牌价格均值
'''
使用 .size() 计算每个组的大小，得到一个 Series。
对于这个例子，可能得到类似 {'brand1': 2, 'brand2': 2, 'brand3': 1} 的 Series，其中键是品牌名称，值是该品牌出现的次数。
使用 .index 从这个 Series 中提取索引，也就是品牌名称，结果会是一个包含 ['brand1', 'brand2', 'brand3'] 的 Index 对象。
'''
txt = data.groupby('brand').size().index # 获取品牌文本（即品牌名称）

# 基于x,y,s 这三个维度绘制气泡图
plt.figure(figsize = (14,12))

# hue参数应该设置为一个分类变量，以便为不同的类别生成不同的颜色
# 使用 pd.cut 函数将 s 列的数据划分为三个类别（低、中、高）
# bins=[0, s.quantile(0.33), s.quantile(0.67), s.max()] 定义了三个分位数作为分割点
# labels=['低', '中', '高'] 为每个类别分配标签
price_categories = pd.cut(s,bins = [0,s.quantile(0.33),s.quantile(0.67),s.max()],labels = ['低','中','高'])


# 使用 seaborn 的 scatterplot 绘制气泡图
# x=x 表示 x 轴的数据来自 x 变量
# y=y 表示 y 轴的数据来自 y 变量
# size=s 表示气泡的大小由 s 变量决定
# hue=s 表示使用 s 变量为不同的类别生成不同的颜色
# sizes=(100, 1500) 设定气泡大小的范围，最小为 100，最大为 1500
scatter = sns.scatterplot(
                  x = x,
                  y =y, 
                  size = s, #点大小由价格均值决定
                  hue = price_categories, # 气泡大小的设定，现设定为销量的均值，可调整
                  sizes = (100,1500), #点大小阈值范围
                  )
# 标注每个点对应的品牌文本
for i in range(len(txt)):
    plt.annotate(txt[i],#品牌文本
                xy = (x.iloc[i], y.iloc[i]),#点的坐标
                xytext = (5,5),# 文本相对于点的偏移量
                textcoords = "offset points",
                ha = 'center'
                )
    
# 图标基本配置
plt.ylabel('热度')
plt.xlabel('销量')
plt.title('各品牌的销量、热度')
# 设置图表的对齐方式（左上方开始）
plt.legend(loc = 'upper left')
plt.show()


#箱型图

# 通过箱型图了解各品牌价格的分布情况

#查看价格的箱型图
plt.figure(figsize = (14,6))
sns.boxplot(x = 'brand',
           y = 'price',
           data = data,
           color = 'pink')
plt.ylim(0,3500) # 如果不限制，就不容易看清箱型，所以把轴缩小
plt.title("各品牌价格箱型图")
plt.xticks(rotation = 45)
plt.show()

## 带参考线的柱状图
fig = plt.figure(figsize = (12,6))
ax  = plt.gca()
avg_price.sort_values(ascending = False).plot(kind = 'bar', width = 0.8, alpha = 0.6,color = get_colors(brands),ax = ax,label = '各品牌平均价格')
y = data['price'].agg('mean')
plt.axhline(y,0 ,5, color = 'red', label = '全品牌平均价格')
#ax.patches 包含了柱状图的每个条
for p in ax.patches:
    # 获取当前条的高度
    height = p.get_height() 
    # 使用 ax.annotate 方法添加注释
    # 标注的内容是条的高度，使用 format 函数将高度转换为字符串
    # xy 是标注的位置，为条的中心位置，x 坐标为条的起始位置加上条宽的一半，y 坐标为条的高度
    # xytext 是文本的偏移量，这里是相对于 xy 位置向上偏移 3 个点
    # textcoords='offset points' 表示 xytext 的坐标单位是点
    # ha='center' 表示水平对齐方式为居中
    # va='bottom' 表示垂直对齐方式为底部
    ax.annotate(
        '{}'.format(height),#标注内容为条高度
        xy = (p.get_x()+ p.get_width()/2, height),
        xytext = (0,3),
        textcoords = 'offset points',
        ha = 'center',
        va = 'bottom'
    )
plt.ylabel('各品牌平均价格')
plt.title('各品牌产品的平均价格', fontsize = 24)
plt.legend(loc = 'upper right')
plt.xticks(rotation = 45)
plt.show()


##双轴折线图
fig, ax = plt.subplots(1,1,figsize = (12,6))

day_salecnt.plot(kind = 'line', ax = ax ,marker = 'o',color = 'blue',linestyle = '-', label = '每日销量')
ax.set_title('每日销量趋势',fontsize = 12)
ax.set_xlabel('品牌',fontsize = 10)
ax.set_ylabel('销量', fontsize = 10)

ax2 = ax.twinx()
day_salemon.plot(kind = 'line', ax = ax2, marker = 'o', color = 'red', linestyle = '-', label = '每日销售额')
ax2.set_ylabel('销售额',fontsize = 10)
# 获取 ax 的图例手柄和标签
# handles, labels = ax.get_legend_handles_labels() 获取 ax 的图例手柄和标签
handles, labels = ax.get_legend_handles_labels()
# 获取 ax2 的图例手柄和标签
# handles2, labels2 = ax2.get_legend_handles_labels() 获取 ax2 的图例手柄和标签
handles2, lables2 = ax2.get_legend_handles_labels()
ax.legend(handles + handles2, labels + labels2, loc = 'upper right')
plt.show()

##单轴折线图，for循环生成多个子图
#随机生成颜色
def randomColor(brands):
    brand_colors = {}#创建新列表
    for brand in brands:
        color = np.random.choice(list('0123456789ABCDEF'),size = (6,))
        brand_colors[brand] ='#'+''.join(color)
    return brand_colors

fig , axes = plt.subplots(nrows = 2, ncols = 1,figsize = (18,20))
for brand in brands:
    subset = day_brand[day_brand['brand'] == brand]
    axes[0].plot(subset['update_time'], subset['sale_count'],label = f'{brand}', marker = 'o',color = brand_colors[brand],linestyle = '-')

for brand in brands:
    subset = day_brand[day_brand['brand'] == brand]
    axes[1].plot(subset['update_time'],subset['sale_money'],label =  '{}'.format(brand),marker = 'o', color = brand_colors[brand],linestyle = '-')

axes[0].set_title('各品牌每日销量趋势', fontsize = 12)
axes[1].set_title('各品牌每日销售额趋势图',fontsize = 12)
axes[0].set_xlabel('Day')
axes[0].set_ylabel('每日总销量',color = 'k')
axes[1].set_ylabel('每日总销售额',color = 'k')

axes[0].legend(loc='upper right',fontsize = 8)
axes[1].legend(loc='upper right',fontsize = 8)
plt.show()

### 用第二列没行的名称，转至成表格的列名
# 统计品牌每天变化趋势
# 按 'day' 和 'brand' 进行分组，并对 'sale_count' 列进行求和聚合
# data.groupby(['day', 'brand'])['sale_count'].agg(sum) 按 'day' 和 'brand' 对数据进行分组，并对 'sale_count' 列进行求和操作
#.unstack(level=1) 将结果中的二级索引（这里是 'brand'）转换为列索引
#.reset_index() 将结果的索引重置为默认的整数索引
sale_transformation = data.groupby(['day','brand'])['sale_count'].agg(sum).unstack(level = 1).reset_index()
# sale_transformation 的两层索引全部转化到列索引上
sale_transformation