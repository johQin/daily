from pandas import DataFrame
from pandas import Series
df1 = DataFrame({
    'age': Series([26, 29, 24]),
    'name': Series(['Ken', 'Jerry', 'Ben'])
})
print(df1)
#    age   name
# 0   26    Ken
# 1   29  Jerry
# 2   24    Ben

# 访问列
print(df1['name'])

# 0      Ken
# 1    Jerry
# 2      Ben
#访问多列：df1[['name','age']]

# .loc为按索引名提取， .iloc为按索引号提取
# 访问行
print(df1.loc[1:])
#    age   name
# 1   29  Jerry
# 2   24    Ben

# 访问块
print(df1.iloc[0:2, 0:2])# 获取第0行到第2行（不包含），获取第0列到第2列（不包含）
#    age   name
# 0   26    Ken
# 1   29  Jerry

# 访问指定位置
print(df1.at[1, 'name'])
# Jerry

# 修改列名
df1.columns = ['Age', 'Name']
print(df1)
#    Age   Name
# 0   26    Ken
# 1   29  Jerry
# 2   24    Ben

#增加行，改变原df
df1.loc[len(df1)] = [24, 'qin']
print(df1)
#增加列，改变原df
df1['Sex'] = [1, 1, 2, 1]
print(df1)
#删除行，不改变原df
df2 =df1.drop(1, axis=0)
print(df1)
print(df2)

#删除列，不改变原df
df3 = df1.drop('Name', axis=1)
print(df1)
print(df3)
