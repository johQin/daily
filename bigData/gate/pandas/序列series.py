from pandas import Series
s1 = Series(['a', True, 1], index=['first', 'second', 'third'])
print(s1)
# first        a
# second    True
# third        1
# dtype: object

print(s1[1])# 按索引号访问，True

print(s1['second'])# 按索引名访问，True
s2 = Series([False, 2])
s3 = s1.append(s2)# 可以追加序列，不可以追加单个元素
print(s3)
# first         a
# second     True
# third         1
# 0         False
# 1             2
# dtype: object

print(s1)#原序列不发生变化
print(s1.index)
# Index(['first', 'second', 'third'], dtype='object')
print(s1.values)
# ['a' True 1]

