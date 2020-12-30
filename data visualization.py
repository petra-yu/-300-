import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re
import pandas as pd # 数据框操作
import numpy as np
import matplotlib.pyplot as plt # 绘图
import jieba # 分词
from wordcloud import WordCloud # 词云
import matplotlib as mpl # 配置字体
from pyecharts.charts import Geo # 地理图
data = pd.read_csv('/Users/apple/Desktop/fine！/未命名文件夹/n+3/N+3.csv')
data.head()
mpl.rcParams['font.sans-serif'] = ['SimHei']
data.info()
num_com = data.company_scale.value_counts()
num_salary = data.salary.value_counts()
num_city = data.city.value_counts()
num_fre  = data.frequency.value_counts()
num_dur  = data.duration.value_counts()
# 各实习岗位对于实习生的基本要求：week month degree

fig = plt.figure(figsize=(15,5))
plt.ylabel('jobs_count',fontsize=18)

plt.subplot(121)
data.groupby('frequency')['frequency'].count().plot.bar()
plt.ylabel('jobs_count',fontsize=18)
plt.xlabel('frequency',fontsize=18)
plt.tick_params(labelsize=12)

plt.subplot(122)
data.groupby('duration')['duration'].count().plot.bar()
plt.xlabel('duration',fontsize=18)
plt.tick_params(labelsize=12)

plt.tight_layout()

# job_money列

# 该列数据需转换为数值，转换规则为取中间数
def trans_money1(x):
    if x=='薪资面议':
        x3 = x
    else:
        try:
            x1 = re.split('-|/',x)[0]
            x2 = re.split('-|/',x)[1]
            x3 = (int(x1)+int(x2))/2
        except: 
            x1 = re.split('/',x)[0]
            x3 = int(x1)
    return x3
data['day_salary'] = data['salary'].map(trans_money1)
data.day_salary.value_counts()


# day_salary中有 面议 ，将面议填充为均值
money_avg = data['day_salary'][data['day_salary']!='薪资面议'].mean().round(1)
data['day_salary'][data['day_salary']=='薪资面议'] = int(money_avg)

# 各地数据分析实习生的日薪
data['day_salary'] = data['day_salary'].apply(int)
temp0 = data.groupby(['city'])['city'].count()
position = list(temp0[temp0>10].index)
temp = data[data['city'].isin(position)].groupby(['city'])['day_salary'].mean().apply(int)

temp.plot.bar()
for i in range(len(list(temp))):
    v = int(list(temp)[i])
    plt.text(i,v+1,v,ha='center',fontsize=15)
plt.xlabel('city',fontsize=18)
plt.ylabel('day_salary',fontsize=18)
plt.tick_params(labelsize=12,rotation=0)
# 绘制一条工资平均线
avg = int(data.day_salary.mean())
print(avg)
plt.axhline(y=avg,ls=":",c="k")
avg_line = plt.text(6,avg+80,'avg=167',fontsize=15)

#词云
import matplotlib.pyplot as plt
import jieba
from wordcloud import WordCloud

text = open('text1.txt','r',encoding='utf-8').read()
cut_text = jieba.cut(text)
result = " ".join(cut_text)
wc = WordCloud(
    font_path='Microsoft Yahei.ttf',     #字体路劲
    background_color='white',   #背景颜色
    width=1000,
    height=600,
    max_font_size=50,            #字体大小
    min_font_size=10,
    mask=plt.imread('/Users/apple/Desktop/fine！/timg.jpg'),  #背景图片
    max_words=1000
)
wc.generate(result)
#wc.to_file('jielun.png')    #图片保存

plt.figure('jielun')   #图片显示的名字
plt.imshow(wc)
plt.axis('off')        #关闭坐标
plt.show()

#热力图
from pyecharts.globals import GeoType
g0 = Geo()
g0.add_schema(maptype='china')
g0.add('',mylist1,symbol_size=15)
g0.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
g0.set_global_opts(visualmap_opts=opts.VisualMapOpts(is_piecewise=False,max_=900),title_opts=opts.TitleOpts(title='全国实习工作分布'))
g0.render_notebook()


g0 = Geo()
g0.add_schema(maptype='china')
g0.add('',mylist3,symbol_size=15)
g0.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
g0.set_global_opts(visualmap_opts=opts.VisualMapOpts(is_piecewise=False,max_=300),title_opts=opts.TitleOpts(title='全国实习工资分布'))
g0.render_notebook()