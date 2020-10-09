class StrReplace():
    def replace(self,str,rep='%20'):
        b_list=[]
        for i in rep:
            b_list.append(i)
        a_list=[]
        for i in str:
            if i==' ':
                a_list.extend(b_list)
            else:
                a_list.append(i) 
        res=''.join(a_list)
        return res
if __name__ == "__main__":
    #法一：采用python内置的replace
    str='if you are not happy, please try to reading or sporting'
    res0=str.replace(' ','%20')
    print('res0',res0)
    #法二： 新建数组，依次添加
    sr=StrReplace()
    res=sr.replace('if you are not happy, please try to reading or sporting')
    print('res',res)