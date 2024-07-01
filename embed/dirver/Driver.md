# Driver

操作系统五大功能：进程管理，内存管理，网络管理，文件系统，设备管理

计算机系统中存在着大量的设备，操作系统要求能够控制和管理这些硬件，而驱动就是帮助操作系统完成这个任务。

驱动相当于硬件的接口，它直接操作、控制着我们的硬件，操作系统通过驱动这个接口才能管理硬件。

<img src="legend/image-20240628155105025.png" alt="image-20240628155105025" style="zoom: 80%;" />

# 0 概述

## 0.1 驱动与应用程序

- 驱动程序本身也是代码，但与应用程序不同，它不会主动去运行，而是被调用。这调用者就是应用程序。
- 驱动一般不会主动运行，由应用程序调用而被动执行。
- 驱动与应用是服务与被服务的关系。驱动是为应用服务的。应用程序通过**系统调用**陷入内核调用驱动，从而操作硬件。
  - 系统调用：内核提供给用户程序的一组“特殊”函数接口，用户程序可以通过这组接口获得内核提供的服务
- 应用程序运行在用户空间(用户态)，驱动代码运行于内核空间(内核态)。
- 驱动开发的所有接口均来自内核，只要内核能提供的接口，就不要自己开发。

![image-20240628155503775](legend/image-20240628155503775.png)

应用程序访问驱动的流程：

![image-20240628160548642](legend/image-20240628160548642.png)

- 应用程序调用函数库完成一系列功能，库函数通过系统调用由内核完成相应功能，内核处理系统调用，内核通过设备类型和设备号找到唯一的驱动，再由驱动操作硬件。



**驱动开发过程中使用的库函数，内核通常都有重新实现，接口类似标准C库**

## 0.2 驱动分类

linux中驱动分为：

- 字符设备驱动：读写线性实时
  - I/O传输过程中以字符（字节）为单位进行传输
  - 用户对字符设备发出读/写请求时，实际的硬件读/写操作一般紧接着发生
- 块设备驱动：读写非线性非实时
  - 数据传输以块（内存缓冲）为单位传输
  - 磁盘类、闪存类等设备都封装成块设备。
- 网络设备驱动，通过套接字socket接口函数访问



## 0.3 设备文件/设备号

### 设备文件

linux中设备文件在/dev中，

- linux把设备抽象成文件,“一切设备皆文件”。所以对硬件的操作全部抽象成对文件的操作。
- 设备文件大小为0
- 设备文件主要记录了设备类型（b 块设备，c字符设备）和设备号（主设备号，从设备号）
- 设备文件是应用程序访问驱动程序的桥梁

```bash
# 从ls可以看出来，本来如果是普通文件，它是有内存大小的区别的，而我们的设备文件是没有内存大小的，而且在原本该显示内存大小的地方却显示了设备号。
# 第五列可以看出来，普通文件显示的是占用的内存大小，而设备文件是设备号
# 并且在total行，一个显示的是188，一个显示的是0
ls -l /dev
total 0
brw-rw----    1 root     root      179,   0 Jan  1  1970 mmcblk0					# b代表的是块设备
brw-rw----    1 root     root      179,   8 Jan  1  1970 mmcblk0boot0				# 第5列显示的是 主设备号，从设备号
crw-rw----    1 root     root        5,   0 Jan  1  1970 tty						# c代表的是字符设备
crw-rw----    1 root     root        4,   0 Jan  1  1970 tty0
crw-rw----    1 root     root        4,   1 Jan  1  1970 tty1

ls -l /etc
total 188
-rwxr--r--    1 1003     1003            90 Mar 20  2017 README.txt
-rwxr-xr-x    1 1003     1003           377 Nov 27  2013 fstab
-rwxr-xr-x    1 1003     1003            15 Nov 27  2013 group
drwxr-xr-x    2 1003     1003          4096 Nov 27  2013 hotplug
-rwxr-xr-x    1 1003     1003          5615 Aug 15  2013 httpd.conf
```

### 设备号

- **主设备号**
  - 用于标识驱动程序，一个驱动程序只能有一个主设备号，主设备号一样的设备文件将使用同一类驱动程序。
  - 范围：1-254，0有特殊的用途
- **从设备号**
  - 用于标识使用同一驱动程序的不同具体硬件和功能。
  - 从设备号一般都是用户手动指定分配的
  - 范围：0-255

通过`ls -l /dev`查看第五列，可以看到主设备号和从设备号

```bash
# 通过/proc/devices文件可以查看主设备号，但无法查看从设备号
cat /proc/devices 

Character devices:
  1 mem
  4 /dev/vc/0
  4 tty
  5 /dev/tty
  5 /dev/console
  5 /dev/ptmx
  7 vcs
 10 misc
 13 input

Block devices:
259 blkext
  7 loop
  8 sd
 11 sr
 65 sd
 66 sd
 67 sd
 68 sd
 69 sd

```

## 0.4 linux模块化编程

- 控制内核大小（不需要的组件可以不编入内核）
- 调试开发灵活（模块可以同普通软件一样，从内核中添加或删除）
- 独立于内核，可以单独编译和加载运行

Linux内核模块的编译方法有**两种**：

- 放入Linux内核源码中编译

  * 将写好的模块放入Linux内核任一目录下
  * 修改相应目录下的Kconfig和Makefile文件
  * 执行make modules
  * 会在相同目录下生成与源文件同名的.ko文件

- 采用独立的方法编译模块

  - 需要独立的Makefile

    ```makefile
    obj‐m := demo_module.o #模块名字，与C文件同名
    KERNELDIR = /…/kernel‐3.4.39 #内核路径得根据自己的实际解压路径进行修改
    PWD = $(shell pwd) #当前路径
    default: #编译过程
    	$(MAKE) ‐C $(KERNELDIR) M=$(PWD) modules
    	rm ‐rf *.order *.mod.* *.o *.symvers
    clean:
    	rm ‐rf *.ko
    ```

  - 三步实现一个内核模块

    ```c
    // 首先确定你的模块存放位置，建议存放在内核源码目录下的debug目录
    
    // 相关宏，相关函数放在此头文件
    #include <linux/module.h>
    
    static int __init demo_module_init(void)
    {
    	printk(KERN_WARNING "L%d‐>%s()\n",__LINE__,__FUNCTION__);
    	return 0;
    }
    static void __exit demo_module_exit(void)
    {
    	printk(KERN_WARNING "L%d‐>%s()\n",__LINE__,__FUNCTION__);
    }
    // 声明模块初始化回调
    module_init(demo_module_init);
    // 声明模块退出时回调
    module_exit(demo_module_exit);
    
    // 第一步：声明GPL
    MODULE_LICENSE("GPL");
    MODULE_DESCRIPTION("xxxxxxxxxxxxxxxx");
    ```
  
- 模块操作指令

  - lsmod 列举当前系统中的所有模块
  - insmod xxx.ko 加载指定模块到内核
  - rmmod xxx 卸载指定模块(不需要.ko后缀)
  - modinfo xxx.ko 查看模块信息



# 1 字符设备驱动

内核中超过一半的代码都是驱动代码，而驱动代码里面有一半代码都是字符设备的驱动代码。

字符设备是最基本、最常用的设备。它将千差万别的各种硬件设备采用一个统一的接口封装起来，屏蔽硬件差异，简化了应用层的操作。

- **字符驱动开发的过程，就是一个实现 与系统调用一一对应函数接口 的过程**
- 接口实现之后，应用程序通过file_operations与驱动建立连接。
- **驱动的open/read/write函数**实际上是由一个叫**file_operations**的结构体统一管理的。**file_operations里面包含了一组函数指针,这组函数指针指向驱动open/read/write等几个函数。**一个打开的设备文件就和该结构体关联起来，结构体中的函数实现了对文件的系统调用。

```c
struct file_operations {
	struct module *owner;
	loff_t (*llseek) (struct file *, loff_t, int);
	ssize_t (*read) (struct file *, char __user *, size_t, loff_t *);
	ssize_t (*write) (struct file *, const char __user *, size_t, loff_t *);
	ssize_t (*aio_read) (struct kiocb *, const struct iovec *, unsigned long, loff_t);
	ssize_t (*aio_write) (struct kiocb *, const struct iovec *, unsigned long, loff_t);
	int (*readdir) (struct file *, void *, filldir_t);
	unsigned int (*poll) (struct file *, struct poll_table_struct *);
	long (*unlocked_ioctl) (struct file *, unsigned int, unsigned long);
	long (*compat_ioctl) (struct file *, unsigned int, unsigned long);
	int (*mmap) (struct file *, struct vm_area_struct *);
	int (*open) (struct inode *, struct file *);
	int (*flush) (struct file *, fl_owner_t id);
	int (*release) (struct inode *, struct file *);
	int (*fsync) (struct file *, loff_t, loff_t, int datasync);
	int (*aio_fsync) (struct kiocb *, int datasync);
	int (*fasync) (int, struct file *, int);
	int (*lock) (struct file *, int, struct file_lock *);
	ssize_t (*sendpage) (struct file *, struct page *, int, size_t, loff_t *, int);
	unsigned long (*get_unmapped_area)(struct file *, unsigned long, unsigned long, unsigned long, unsigned long);
	int (*check_flags)(int);
	int (*flock) (struct file *, int, struct file_lock *);
	ssize_t (*splice_write)(struct pipe_inode_info *, struct file *, loff_t *, size_t, unsigned int);
	ssize_t (*splice_read)(struct file *, loff_t *, struct pipe_inode_info *, size_t, unsigned int);
	int (*setlease)(struct file *, long, struct file_lock **);
	long (*fallocate)(struct file *file, int mode, loff_t offset,
			  loff_t len);
};
```



# 其他

1. [sourceInsight](https://blog.csdn.net/wkd_007/article/details/131316924)
   - 【Ctrl + F】文件中查找操作
   - 【ctrl + /】 全局搜索关键字
2. [sourceinsight 自动补全](https://blog.csdn.net/byhyf83862547/article/details/137090831)
   - 选项卡options -> preference -> symbol lookups -> import symbols for all Projects -> add， 加入相关头文件文件夹到list表
3. 

