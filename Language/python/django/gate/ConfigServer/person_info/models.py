from django.db import models

class Person(models.Model):
    GENDER = ((1, '男'), (0, '女'))

    name = models.CharField(max_length=32)
    age = models.IntegerField()
    gender = models.BooleanField(choices=GENDER)
    id_card = models.CharField(max_length=18)
    temperature = models.FloatField()

    # 是这个model的管理器或配置中心
    class Meta:
        db_table='personList'   # 设置映射的数据表名，默认为“应用名_模型名”
        permissions = ()


