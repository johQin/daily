# 0 环境搭建

## 0.1.anaconda

### 0.1.1 anaconda与python的区别

1、anaconda 是一个python的发行版，包括了python和很多常见的软件库, 和一个包管理器conda。常见的科学计算类的库都包含在里面了，使得安装比常规python安装要容易。

2、Anaconda是专注于数据分析的Python发行版本，包含了conda、Python等190多个科学包及其依赖项。

### 0.1.2 anaconda安装包命名含义

Anaconda3-2019.10-Windows-x86_64

3表示支持的python的版本是3.x

2019.10是anaconda的版本号。

Windows-x86支持windows32位系统，Windows-x86_64支持windows64位系统。

### 0.1.3 anaconda修改国内镜像源

打开Anaconda Prompt窗口，执行如下命令：

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pks/free/ 

conda config --set show_channel_urls yes

查看当前配置信息：conda info

到C:\Users\Administrator（用户名） 下找到 .condarc，这是一个配置文件，打开删除channels的defaults项

![修改配置文件](/home/buntu/gitRepository/daily/bigData/legend/configmirrors.png)



查看anaconda是否安装成功

anaconda prompt 下

查看anaconda 版本 ：conda --version 

查看anaconda第三方依赖包列表：conda list

在安装其他任何包之前先安装pip：conda install pip

![安装成功](/home/buntu/gitRepository/daily/bigData/legend/installed.png)



## 0.2 sandbox

Sandboxie(又叫沙箱、沙盘)即是一个虚拟系统程序，它创造了一个类似沙盒的独立作业环境，在其内部运行的程序并不能对硬盘产生永久性的影响。

创建沙箱：conda create -n 沙箱名 python=3.6	//沙箱名，可以任意取，3.6代表支持的python 版本号。创建成功后，我们可以在anaconda Navigator/Environment，除了root环境，还有一个名叫tensorflow的沙箱环境

激活沙箱：activate 沙箱名 		//在使用时，需要激活

关闭沙箱：deactivate 沙箱名 或activate root	//不需要使用时，关闭

删除沙箱环境：conda remove -n 沙箱名 --all

查看沙箱列表：conda info -e 或conda env list

### 0.2.1 新环境中使用jupyter

**jupyter notebook的默认环境是base(root)**，当新建沙箱环境后，在新环境中安装tensorflow，然后在jupyter中使用（假设你在base(root)中没有安装tensorflow），在引入tensorflow时，是找不到tensorflow的，你必须将新环境注入到内核kernel中。

解决办法[windows jupyter notebook 切换默认环境](<https://blog.csdn.net/u014264373/article/details/86541767>)

主要步骤

1. 激活新环境：activate new_env
2. 安装内核：conda install ipykernel        //ipykernel：为不同的虚拟机或CONDA环境设置多个IPython内核
3. 将选择的conda环境注入Jupyter Notebook：python -m ipykernel install --user --name 《new_env_name》 --display-name "Python [conda env:《new_env_name》]"
4. 删除jupyter中的内核环境：jupyter kernelspec remove env_name

完成后的效果

![jupyter_new_env](/home/buntu/gitRepository/daily/bigData/legend/jupyter_new_env.png)

![jupyter_kernel_notebook](/home/buntu/gitRepository/daily/bigData/legend/jupyter_kernel_notebook.png)

要在新环境下运行程序，必须要激活此环境，否则将会出现环境死掉的情况，推荐通过prompt 去activate 新环境，然后通过jupyter notebook打开

## 0.3 jupyter notebook

### 0.3.1 界面简介

在prompt 窗口中

打开jupyter notebook：jupyter notebook	//也可以在开始/Anaconda3(64-bit)/jupyter notebook，点击打开

关闭jupyter notebook：Ctrl + C

![jupyter](/home/buntu/gitRepository/daily/bigData/legend/jupyter.png)

点击python3就可以新建python代码。

![jupyter_new_python3](/home/buntu/gitRepository/daily/bigData/legend/jupyter_new_python3.png)

![快捷点击项](/home/buntu/gitRepository/daily/bigData/legend/jupyter_notebook.png)

### 0.3.2 tab项

带扩写

### 0.3.3 jupyter配置文件

在

**C:/用户/用户名/.jupyter/jupyter_notebook_config.py**

如果没有jupyter_notebook_config.py文件，在prompt中**jupyter notebook --generate-config** 即可

**修改**jupyter_notebook_config.py中配置文件**默认存储路径**：**c.NotebookApp.notebook_dir = '路径'**。并删除行首的 ' # '以取消python的注释。

**eg：D:\\\JDI\\\tensorflow\\\practice**，由于python语法，字符串中有' \ '需要转义，用双反斜杠才实际代表单反斜杠。也可以使用**r'D:\JDI\tensorflow\practice'**

如果做了如上修改依旧没有改变文件的默认存储位置，那么还需要右击jupyter notebook---》下拉菜单点击”属性“---》删除"目标"中的%USERPROFILE%，如下图

![jupyter_default_path](/home/buntu/gitRepository/daily/bigData/legend/jupyter_default_path.png)

### 0.3.4 快捷键

选中单元为蓝色边框，编辑单元为绿色边框

1. **Ctrl-Enter :** 运行本单元
2. **Alt-Enter :** 运行本单元，在其下插入新单元
3. **A：**在选中单元上插入代码块
4. **B：**在选中单元下方插入代码块
5. **D-D:**连续按两个D，删除选中单元
6. **Ctrl-?：**注释和解除注释
7. **Ctrl-Shift-- :** 分割单元
8. **Ctrl-Shift-Subtract :** 分割单元
9. **Ctrl-S :** 文件存盘

### 0.3.5 kernel

![kernel下拉框选项](/home/buntu/gitRepository/daily/bigData/legend/jupyter_kernel_tab.png)

1. **Interrupt**：是终止一个 cell，不影响跑过的 cell
2. **Restart**：restart the current kernel。 All variables will be lost。可以清空之前模型训练的结果
3. **Restart & Clear Output**： restart the current kernel and clear all output。All variables and outputs will be lost.
4. **Restart & Run All：**restart the current kernel and re-execute the whole notebook。All variables and outputs will be lost。可以重新跑程序，并按单元输出结果
5. **Reconnect：**
6. **Shutdown：**是终止一个 ipython kernel，kernel 的堆栈直接清空

## 0.4 prompt 和 依赖包

### conda

安装多个包：conda install package_name1  package_name2

**安装特定版本的安装包：conda install package_name = 版本号**

卸载包：conda remove package_name

更新包：conda update package_name

更新环境中的所有包：conda update --all

查看anaconda 版本 ：conda --version 

查看anaconda第三方依赖包列表：conda list

查看指定依赖包的信息(此办法也可用于查看环境中是否有此依赖包)：conda list package_name

**在安装指定版本包的时候，如果找不到，我们可以看线上存在的版本**

查看线上有哪些包：anaconda search -t conda package_name

查看包的详细信息：anaconda show user/package_name

安装此包：conda install --channel channel_url package_name

### pip

安装包：`pip install <包名>==版本号`

更新包：`pip install --upgrade package_name`

卸载包：`pip uninstall <包名>`

 查看当前已安装的包及其版本号：pip freeze

**conda和pip的区别：**

 [Pip](https://pip.pypa.io/en/stable/)是Python Packaging Authority推荐的用于从[Python Package Index](https://pypi.org/)安装包的工具。Pip安装打包为wheels或源代码分发的Python软件。

[Conda](https://conda.io/docs/)是跨平台的包和环境管理器，可以安装和管理来自[Anaconda repository](https://repo.anaconda.com/)以 [Anaconda Cloud](https://anaconda.org/)的conda包。conda包不仅限于Python软件。它们还可能包含C或C ++库，R包或任何其他软件。

conda安装会根据包的依赖关系安装多个包以期环境相适应，而pip则不会。

### 安装时经常会遇到的问题

网络错误（网断了）：Could not fetch URL https://pypi.org/simple/matplotlib/: There was a problem confirming the ssl certificate: HTTPSConnectionPool

## 0.5 安装tensorflow

打开Anaconda Prompt窗口

普通版TensorFlow：conda install tensorflow

GPU版TensorFlow：conda install tensorflow-gpu

测试TensorFlow是否安装成功

打开jupyter

![installedtensorflow](/home/buntu/gitRepository/daily/bigData/legend/provedInstalledTensorflow.png)

![installedtensorflow](/home/buntu/gitRepository/daily/bigData/legend/provedInstalledTensorflow2.png)

由于tensorflow2.x与tensorflow1.x区别较大，很多函数的操作写法都不尽相同，视频上的版本为1.2.1，代码都是在1.2.1的基础上写成的

## 0.6  模块安装log

1. [No Module Named 'Symbol'](https://stackoom.com/en/question/4nDnI)

# 1 log

1. [pycharm 搭建anaconda集成开发环境](https://blog.csdn.net/weixin_51009494/article/details/124542500)

2. [查看当前环境](https://www.coder.work/article/7496109)

   ```bash
   # 当前激活的环境名称
   $CONDA_DEFAULT_ENV
   # 当前激活环境的路径
   $CONDA_PREFIX
   ```

3. 切换沙箱

   ```bash
   # 激活沙箱
   conda activate pa_env
   # 回到base沙箱
   conda deactivate
   ```

   

4. 

## 1.1 [conda环境和迁移](https://zhuanlan.zhihu.com/p/87344422)

```bash
 # 在源环境的prompt（命令行）生成environment.yml
 conda env export > environment.yml
 
 # 在目标环境（命令行）按照environment.yml
 conda env create -f environment.yml
```

