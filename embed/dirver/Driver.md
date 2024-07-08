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
    obj-m += demo_module.o
    KERNELDIR := /home/buntu/sambaShare/kernel-3.4.39
    PWD := $(shell pwd)
    
    modules:
    	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules
    	rm -rf *.order *.mod.* *.o *.symvers
    
    clean:
    	make -C $(KERNELDIR) M=$(PWD) clean
    	rm -rf *.ko
    
    
    # 如果make时有问题（但不是代码问题），那么手动敲一遍
    # obj-m 是用于 Linux 内核模块编译的 Makefile 中的一种特殊约定，它用来指定要编译的模块对象文件。
    # KERNELDIR：指定内核源码的路径
    
    # $(MAKE) ‐C $(KERNELDIR) M=$(PWD) modules
    # $(MAKE)：MAKE变量用来确保在多层次的构建过程中，使用相同的 make 程序。并且$(MAKE) 还会自动传递命令行参数给子模块，例如父模块的makefile用的是make -j4 ，那么子模块在$(MAKE)就代表的是make -j4，这里会带参传递
    # -C 选项的作用是指将当前工作目录转移到你所指定的位置。
    # “M=”选项的作用是，当用户需要以某个内核为基础编译一个外部模块的话，需要在make modules 命令中加入“M=dir”，程序会自动到你所指定的dir目录中查找模块源码，将其编译，生成KO文件。
    # modules 是一个通用的目标，适用于编译所有在当前目录中的模块源文件。你不需要将 modules 具体化为某个特定的模块名，它会自动识别并编译 Makefile 中定义的所有模块。
    
    # "$(MAKE) -C $(KDIR) M=$(PWD)"与"$(MAKE) -C $(KDIR) SUBDIRS =$(PWD)"的作用是等效的，后者是较老的使用方法。推荐使用M而不是SUBDIRS
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
  
- **上述模块是在虚拟机中编译，而下面的指令是在开发板中执行。**

- 模块操作指令

  - lsmod 列举当前系统中的所有模块

  - insmod xxx.ko 加载指定模块到内核

    ```bash
    # 如果你在虚拟机中，执行insmod，就会报如下的错
    insmod demo_module.ko 
    insmod: ERROR: could not insert module demo_module.ko: Invalid module format
    
    # 因为虚拟机linux内核的版本，和开发板内核的版本不一致
    # https://blog.csdn.net/zhangna20151015/article/details/119596386
    ```

    

  - rmmod xxx 卸载指定模块(不需要.ko后缀)

  - modinfo xxx.ko 查看模块信息

- 手动创建设备文件（节点）：`mknod /dev/mychardev c 240 0`



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



## 1.1 开发框架

### 1.1.1 手动创建设备驱动

- 实现file_operations接口
- 在xxx_module_init里：注册字符设备驱动`register_chrdev(major, name, fops);`，手动指定主设备号
- 在xxx_module_exit里：注销字符设备驱动`unregister_chrdev(major, name);`
- 模块编译make（参考0.4 linux模块化编程）
- 安装驱动模块`insmod xxx.ko`：使驱动可以在`cat /proc/devices` 中看到
- 创建设备节点`mknod /dev/mychardev c 242 0`
  - c-字符设备，242-主设备号，0-从设备号
  - 使其可以在`/dev`文件夹下看到设备文件

```c
#include <linux/module.h>
#include<linux/fs.h>

static int demo_open (struct inode *pinode, struct file *pfile){
	printk(KERN_WARNING "L%d‐>%s()\n",__LINE__,__FUNCTION__);
	return 0;
}

static ssize_t demo_read (struct file *pifle, char __user *pbuf, size_t count, loff_t *off){
	printk(KERN_WARNING "L%d‐>%s()\n",__LINE__,__FUNCTION__);
	return 0;
}

static ssize_t demo_write (struct file *pifle, const char __user *pbuf, size_t count, loff_t *off){
	printk(KERN_WARNING "L%d‐>%s()\n",__LINE__,__FUNCTION__);
	return count;	//返回0是写入失败，返回>0写入成功

}
static int demo_release (struct inode *pinode, struct file *pfile){
	printk(KERN_WARNING "L%d‐>%s()\n",__LINE__,__FUNCTION__);
	return 0;
}



static struct file_operations fops = {
	.owner = THIS_MODULE,
	.open = demo_open,
	.read = demo_read,
	.write = demo_write,
	.release = demo_release,
		
};

static int __init demo_module_init(void)
{
	printk(KERN_WARNING "L%d‐>%s()\n",__LINE__,__FUNCTION__);
	register_chrdev(242, "demo_chr", &fops);		// 返回值为主设备号
    // 如果242修改为0，即可动态获取设备号
	return 0;
}

static void __exit demo_module_exit(void)
{
	printk(KERN_WARNING "L%d‐>%s()\n",__LINE__,__FUNCTION__);
	unregister_chrdev(242, "demo_chr");
}

// 声明模块初始化回调
module_init(demo_module_init);

// 声明模块退出时回调
module_exit(demo_module_exit);

// 第一步：声明GPL
MODULE_LICENSE("GPL");


MODULE_DESCRIPTION("xxxxxxxxxxxxxxxx");
```



### 1.1.2 灵活创建设备驱动

- 实现file_operations接口
- 在xxx_module_init里
  - 动态获取设备号：为防止设备号冲突
    - 使用 `alloc_chrdev_region` 分配设备号。
  - 初始化并添加设备驱动：使驱动可以在`cat /proc/devices` 中看到
    - 使用 `cdev_init` 初始化字符设备结构体，并使用 `cdev_add` 将其添加到内核中。
  - 创建设备类和设备节点
    - 使用 `class_create(owner,classname)` 创建设备类，这个函数会在`/sys/class`中创建一个同name名的文件夹
    - 使用 `device_create` 创建设备节点
      - 在`/sys/class/classname`创建了一个`devicename`文件夹，cat 这个`devicename`文件夹下的uevent文件，可以看到主设备号，从设备号，设备名。
      - 在这个使其可以在`/dev`文件夹下看到设备文件`/dev/devicename`
- 在xxx_module_exit里，和注册的顺序相反
  - 注销设备device_destroy
  - 注销设备类class_destroy
  - 删除设备结构体cdev_del
  - 回收设备号资源unregister_chrdev_region

```c
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/uaccess.h>
#include <linux/device.h>

#define DEVICE_NAME "mychardev"
#define DEVICE_CLASS_NAME "mycharcls"
#define DEVICE_COUNT 1

static dev_t dev;
static struct cdev my_cdev;
static struct class *my_class;

static int my_open(struct inode *inode, struct file *file)
{
    printk(KERN_INFO "Device opened\n");
    return 0;
}

static int my_release(struct inode *inode, struct file *file)
{
    printk(KERN_INFO "Device closed\n");
    return 0;
}

static ssize_t my_read(struct file *file, char __user *buf, size_t count, loff_t *ppos)
{
    printk(KERN_INFO "Read from device\n");
    return 0;
}

static ssize_t my_write(struct file *file, const char __user *buf, size_t count, loff_t *ppos)
{
    printk(KERN_INFO "Write to device\n");
    return count;		//返回0是写入失败，返回>0写入成功
}

static struct file_operations fops = {
    .owner = THIS_MODULE,
    .open = my_open,
    .release = my_release,
    .read = my_read,
    .write = my_write,
};

static int __init my_init(void)
{
    int ret;

    // 分配设备号
    ret = alloc_chrdev_region(&dev, 0, DEVICE_COUNT, DEVICE_NAME);
    if (ret < 0) {
        printk(KERN_ERR "Failed to allocate chrdev region\n");
        return ret;
    }

    // 初始化 cdev 结构体
    cdev_init(&my_cdev, &fops);
    my_cdev.owner = THIS_MODULE;

    // 将 cdev 添加到系统中
    ret = cdev_add(&my_cdev, dev, DEVICE_COUNT);
    if (ret < 0) {
        unregister_chrdev_region(dev, DEVICE_COUNT);
        printk(KERN_ERR "Failed to add cdev\n");
        return ret;
    }

    // 创建设备类
    my_class = class_create(THIS_MODULE, DEVICE_CLASS_NAME);
    if (IS_ERR(my_class)) {
        cdev_del(&my_cdev);
        unregister_chrdev_region(dev, DEVICE_COUNT);
        printk(KERN_ERR "Failed to create class\n");
        return PTR_ERR(my_class);
    }

    // 创建设备节点
    device_create(my_class, NULL, dev, NULL, DEVICE_NAME);

    printk(KERN_INFO "Device initialized successfully\n");
    return 0;
}

static void __exit my_exit(void)
{
    device_destroy(my_class, dev);
    class_destroy(my_class);
    cdev_del(&my_cdev);
    unregister_chrdev_region(dev, DEVICE_COUNT);
    printk(KERN_INFO "Device exited successfully\n");
}

module_init(my_init);
module_exit(my_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name");
MODULE_DESCRIPTION("A simple character device driver");
```

```bash
cat /dev/mychardev
[157267.417000] Device opened
[157267.417000] Read from device
[157267.417000] Device closed

echo xxxx > /dev/mychardev
[157323.403000] Device opened
[157323.404000] Write to device
[157323.404000] Device closed
```

### register_chrdev和alloc_chrdev_region的区别

设备号注册两种办法：

- 指定主从设备号并告知内核
- 从内核中动态申请主从设备号

- `alloc_chrdev_region` 
  - 用于动态分配一个主设备号和多个次设备号。
  - 它可以确保主设备号不会与现有的设备号发生冲突，因为内核会选择一个未被使用的主设备号进行分配。
- `register_chrdev` 
  - 既可以分配一个静态指定的主设备号，也可以动态分配一个主设备号（通过传入 0 作为主设备号）
  - 函数返回值为主设备号（>0分配成功并为设备号，小于0分配失败）。
  - 并注册字符设备的文件操作结构体。相比 `alloc_chrdev_region`，它还会同时注册一个字符设备驱动程序，并关联文件操作函数。

### 1.1.3 简单调用

```c
#include<linux/module.h>
#include<linux/fs.h>
#include<device.h>
#include<uaccess.h>

....

static ssize_t my_read(struct file *file, char __user *buf, size_t count, loff_t *ppos)
{
    printk(KERN_INFO "Read from device\n");
    int ret;
    int len = min(count,sizeof(data));
    ret = copy_to_user(buf, data, len);
    printk(KERN_WARNING "L%d->%s()\n", __LINE__, __FUNCTION__);
    return len;			// 在使用cat命令时，如果返回0则读取结束，如果大于0则继续读
}

static ssize_t my_write(struct file *file, const char __user *buf, size_t count, loff_t *ppos)
{
    int ret;
    char data[100];
    int len = min(count,sizeof(data));
    ret = copy_from_user(data, buf, len);
    printk(KERN_WARNING "L%d->%s():%s\n", __LINE__, __FUNCTION__, data);
    return count;		//返回0是写入失败，返回>0写入成功
}
...
```

```c
#include<stdio.h>
#include<sys/types.h>
#include<sys/stat.h>
#include<fcntl.h>
#include<string.h>

void read_test(){
    int fd;
    char data[100];
    fd = open("/dev/mychardev", O_RDWR);
    read(fd, data, 100);
    printf("%s\n", data);
}
void write_test(){
    int fd;
    char data[100] = "hello driver!\n";
    fd = open("/dev/mychardev", O_RDWR);
    write(fd, data, strlen(data) + 1);
    printf("%s\n", data);
}
int main(int argc, char** argv){
    read_test();
    write_test();
    return 0;
}
```

### 1.1.4 设备号的应用



```c
static int my_open(struct inode *inode, struct file *file)
{
    // 可以通过inode获取主从设备号
    printk(KERN_WARNING "major = %d, minor = %d\n", imajor(pinode), iminor(pinode));
	// 如果在open，read，write，close操作到不同的硬件设备（接口）的时候，就可以使用一个全局变量将从设备号存起来，做一个区分。
    return 0;
}
int my_major;
static int __init my_init(void)
{
    int i;
    printk(KERN_WARNING "L%d->%s()\n", __LINE__, __FUNCTION__);
    // register，设备号输入为0，就表示动态分配主设备号
    my_major = register_chrdev(0,DEVICE_NAME,&fops);
    my_class = class_create(THIS_MODULE, DEVICE_CLASS_NAME);
    for(i = 0; i< 10;i++)
        device_create(my_class, NULL, MKDEV(my_major, i+88), NULL, "%s%d",DEVICE_NAME,i+88);
    return 0;
}
static void __exit my_exit(void)
{
   
    int i;
   	printk(KERN_WARNING "L%d->%s()\n", __LINE__, __FUNCTION__);
    for(i = 0; i< 10;i++)
        device_destroy(my_class, MKDEV(my_major, i+88));
    class_destroy(my_class);
    unregister_chrdev(my_major, DEVICE_NAME);
    printk(KERN_INFO "Device exited successfully\n");
}
```

## 1.2 GPIO

找到S5P6818用户手册（SEC_S5P6818X_Users_Manual_preliminary_Ver_0.00.pdf），从中我们可以知道：

- S5P6818的GPIO被分成了GPIOA-GPIOE共5组
- 每组GPIO有32个引脚GPIOX0-GPIOX31
- 确定 GPIO的功能，分成Fun0-3(必须参考数据手册确定其功能，参考2.3 I/O Function Description Ball List Table)，其中GPIOxn表示普通GPIO功能

![image-20240702181235580](legend/image-20240702181235580.png)

板卡的原理图（底板：x6818bv2.pdf，核心板：x4418cv3_release20150713.pdf），核心板焊在底板上

LED的部分

![image-20240702175233797](legend/image-20240702175233797.png)

按键的部分：

![](./legend/KEY_原理图.png)

```c
#include <mach/devices.h> //PAD_GPIO_A+n
#include <mach/soc.h> //nxp_soc_gpio_set_io_func();
#include <mach/platform.h> //PB_PIO_IRQ(PAD_GPIO_A+n);
#include <linux/gpio.h>

/* 内核api接口

// 设置引脚功能
nxp_soc_gpio_set_io_func(unsigned int io,int func); 		// io:寄存器地址，func：功能0,1,2,3
// 确定GPIO输入输出方向
nxp_soc_gpio_set_io_dir(unsigned int io, int out);			//out：0输入，1输出
// 设置 GPIOl引脚输出电平
nxp_soc_gpio_set_out_value(unsigned int io, int out);		//0：输出低电平，1：输出高电平
// 读取 GPIOl引脚输入电平
nxp_soc_gpio_get_in_value(unsigned int io);					//通过返回值得到高低电平

*/


static void my_gpio_init(void)
{
   	// LED
	nxp_soc_gpio_set_io_func(PAD_GPIO_C+11,1); 		//设置引脚功能0‐3
	nxp_soc_gpio_set_io_dir(PAD_GPIO_C+11,1);		//0：输入，1：输出
	nxp_soc_gpio_set_out_value(PAD_GPIO_C+11,1);	//0：输出低电平，1：输出高电平
    
    // KEYBOARD
    nxp_soc_gpio_set_io_func(PAD_GPIO_A+28,1); 		//设置引脚功能0‐3
	nxp_soc_gpio_set_io_dir(PAD_GPIO_A+28,1);		//0：输入，1：输出
 }

static int my_open(struct inode *inode, struct file *file)
{
    printk(KERN_INFO "Device opened\n");
    my_gpio_init();
    return 0;
}
static int my_release(struct inode *inode, struct file *file)
{
    printk(KERN_INFO "Device closed\n");
    nxp_soc_gpio_set_out_value(PAD_GPIO_C+11,1);
    return 0;
}
static ssize_t my_write(struct file *file, const char __user *buf, size_t count, loff_t *ppos)
{
    int ret;
    char data[100] = "";
    int len = min(count,sizeof(data));
    ret = copy_from_user(data, buf, len);
    
    // 设置led灯的亮灭
    if(data[0] == '0'){
        nxp_soc_gpio_set_out_value(PAD_GPIO_C+11,0);
    }else{
        nxp_soc_gpio_set_out_value(PAD_GPIO_C+11,1);
    }
    printk(KERN_WARNING "L%d->%s():%s\n", __LINE__, __FUNCTION__, data);
    return count;		//返回0是写入失败，返回>0写入成功
}
static ssize_t my_read(struct file *file, char __user *buf, size_t count, loff_t *ppos)
{
    int ret;
    printk(KERN_INFO "Read from device\n");
    
    // 读取keyboard的值
    ret = nxp_soc_gpio_get_in_value(PAD_GPIO_A+28);	// 通过返回值得到高低电平
    nxp_soc_gpio_set_out_value(PAD_GPIO_C+11,ret);
    if(ret == 0)
        return 0;
    else
        return 1;
    
}
```



```bash
# 0开启led
echo 0 > /dev/mychardev
# 1关闭led
echo 1 > /dev/mychardev
```

```c
void write_test(){
    int fd;
    char data[100] = "0";
    fd = open("/dev/mychardev", O_RDWR);
    while(1){
        data[0] = '0';
        write(fd, data, strlen(data) + 1);
        usleep(300 * 1000);
        data[0] = '1';
        write(fd, data, strlen(data) + 1);
    }
}
void read_test(){
    int fd,ret;
    char data[100] = "0";
    fd = open("/dev/mychardev", O_RDWR);
    // 轮询
    while(1){
        ret = read(fd, data, 100);
        if(ret == 0){
            printf("key down\n");
        }
    }
}
```



## 1.3 杂项设备注册

- 主设备号默认规定为10，从设备号动态分配
- 可以作为拓展设备驱动数量的一种手段
- **依然是一个字符设备驱动**，是字符设备驱动的另一种更加**简单**的注册方式

```c
#include<linux/miscdevice.h>
// file_operations依然要保留
struct file_operations fops = {
    .owner = THIS_MODULE,
    .open = my_open,
    .release = my_release,
    .read = my_read,
    .write = my_write,
};

static struct miscdevice my_misc = {
    .minor = MISC_DYNAMIC_MINOR,				// MISC_DYNAMIC_MINOR这个宏无需自行定义，不然会报重复定义
    .name = "mychrdev",
    .fops = &fops,
}

static int __init my_module_init(void)
{
	printk(KERN_WARNING "L%d‐>%s()\n",__LINE__,__FUNCTION__);
	misc_register(&my_misc);
	return 0;
}

static void __exit my_module_exit(void)
{
	printk(KERN_WARNING "L%d‐>%s()\n",__LINE__,__FUNCTION__);
	misc_deregister(&my_misc);
}


MODULE_LICENSE("GPL");
```



## 1.4 cdev注册

- register_chrdev()其实是cdev注册过程的封装
- cdev自己封装注册，对系统资源利用率更高

流程：

1. 自动分配主设备号申请接口：alloc_chrdev_region()
2. 自己定义数据结构：struct cdev
3. 初始化cdev：cdev_init()
4. 注册cdev：cdev_add()
5. 创建设备类：class_create()
6. 创建设备文件：device_create()
7. 删除设备文件device_destroy(my_class, dev)
8. 删除设备类 class_destroy(my_class)
9. 注销cdev：cdev_del()
10. 释放动态申请的主设备号：unregister_chrdev_region()

```c
int register_chrdev_region( dev_t from, unsigned count, const char *name);
// from : 主设备号和从设备号，通过宏生成 MKDEV(my_major, my_minor)
// count：占用从设备号数目
// name: 驱动名称
// 返回：失败小于0

int alloc_chrdev_region(dev_t *dev,unsigned baseminor,unsigned count,const char *name);
// baseminor：指定申请时的起始从设备号
// 返回：失败非0

void unregister_chrdev_region(dev_t from, unsigned count);

struct cdev {
	struct kobject kobj; 				// 内核生成，用于管理
	struct module *owner; 				// 内核生成，用于管理
	const struct file_operations *ops;
	struct list_head list; 				// 内核生成，用于管理
	dev_t dev;							//设备号
	unsigned int count; 				// 引用次数
};

// 初始化cdev变量，并设置fops
void cdev_init(struct cdev *cdev, const struct file_operations *fops);
// 添加cdev到linux内核，完成驱动注册
int cdev_add(struct cdev *p, dev_t dev, unsigned count);

// 上面两个函数成功后，可以使驱动在`cat /proc/devices` 中看到

// 从内核中删除cdev数据
void cdev_del(struct cdev *p);
```

```c
static struct cdev my_cdev;

static int __init my_init(void)
{
    int ret;

    // 分配设备号
    ret = alloc_chrdev_region(&dev, 0, DEVICE_COUNT, DEVICE_NAME);
    if (ret < 0) {
        printk(KERN_ERR "Failed to allocate chrdev region\n");
        return ret;
    }

    // 初始化 cdev 结构体
    cdev_init(&my_cdev, &fops);
    my_cdev.owner = THIS_MODULE;

    // 将 cdev 添加到系统中
    ret = cdev_add(&my_cdev, dev, DEVICE_COUNT);
    if (ret < 0) {
        unregister_chrdev_region(dev, DEVICE_COUNT);
        printk(KERN_ERR "Failed to add cdev\n");
        return ret;
    }

    // 创建设备类
    my_class = class_create(THIS_MODULE, DEVICE_CLASS_NAME);
    if (IS_ERR(my_class)) {
        cdev_del(&my_cdev);
        unregister_chrdev_region(dev, DEVICE_COUNT);
        printk(KERN_ERR "Failed to create class\n");
        return PTR_ERR(my_class);
    }

    // 创建设备节点
    device_create(my_class, NULL, dev, NULL, DEVICE_NAME);

    printk(KERN_INFO "Device initialized successfully\n");
    return 0;
}

static void __exit my_exit(void)
{
    device_destroy(my_class, dev);
    class_destroy(my_class);
    cdev_del(&my_cdev);
    unregister_chrdev_region(dev, DEVICE_COUNT);
    printk(KERN_INFO "Device exited successfully\n");
}
```



# 2 linux内核api

## 2.1 中断

Linux操作系统下同裸机程序一样，需要利用中断机制来处理硬件的异步事件。而用户态是不允许中断事件的，因此中断必须由驱动程序来接收与处理。

中断上下文包含进程上下文，在中断上下文中需要注意：

- 不能使用导致睡眠的处理机制（信号量、等待队列等）
- 不能与用户空间交互数据(copy_to/from_user)
- 中断处理函数执行时间应尽可能短

```bash
# 显示系统中断的状态和统计信息
cat /proc/interrupts
			CPU0       CPU1       CPU2       CPU3       
  0:         53          0          0          0   IO-APIC-edge      timer
  1:          2          0          0          0   IO-APIC-edge      i8042
  8:          1          0          0          0   IO-APIC-edge      rtc0
  9:          0          0          0          0   IO-APIC-fasteoi   acpi
 12:          4          0          0          0   IO-APIC-edge      i8042
 16:        203          0          0          0   IO-APIC-fasteoi   ehci_hcd:usb1
 23:       1014          0          0          0   IO-APIC-fasteoi   ehci_hcd:usb2
 40:       1920          0          0          0   PCI-MSI-edge      eth0
 41:       3000        100          0          0   PCI-MSI-edge      eth1
 NMI:         10         12         14         15   Non-maskable interrupts
 LOC:     123456     234567     345678     456789   Local timer interrupts
 
 # 每行开头的数字表示中断号。这个号唯一标识一个特定的中断源。
 # 中间的数字是每个CPU核心处理该中断的次数。
 # 再后面就是中断类型，中断源（设备）
```

### 2.1.1 中断api

```c
#include <linux/interrupt.h>	// 中断的api
#include <linux/irqreturn.h>	// 中断处理函数的返回值
#include <mach/irqs.h>			// 中断号
#include <linux/irq.h> 			// 外部中断的触发方式
// 中断注册（申请中断）
int request_irq(
    unsigned int irq,
	irqreturn_t(*handler)(int,void*),
	unsigned long irqflag,
	const char *devname,
	void *dev_id
);
// irq：中断号，所申请的中断向量，比如EXIT0中断等定义在mach/irqs.h, 外部中断获取中断编号接口：gpio_to_irq(unsigned int io);
// eg: gpio_to_irq(PAD_GPIO_A+28)
// handler：中断处理函数
// irqflag：中断属性设置
// devname：中断名称（中断源）
// dev_id：私有数据，给中断服务函数传递数据
// 注册成功后，自动开启中断，不需要单独调用enable使能


// 单独设置中断触发方式
int set_irq_type(int irq, int edge);
// edge: 外部中断触发方式定义在 #include <linux/irq.h>
// IRQ_TYPE_LEVEL_LOW //低电平触发
// IRQ_TYPE_LEVEL_HIGH //高电平触发
// IRQ_TYPE_EDGE_FALLING //下降沿触发
// IRQ_TYPE_EDGE_RISIN //上升沿触发
// IRQ_TYPE_EDGE_BOTH //双边沿触发

// 释放中断
void free_irq(unsigned int irq, void *dev_id);

// 使能中断
void enable_irq(unsigned int irq);

// 关闭中断，并等待中断处理完成后返回
void disable_irq(unsigned int irq);

// 关闭中断，立即返回
void disable_irq_nosync(unsigned int irq);

// 服务中断函数
irqreturn_t handler(int irq,void *dev_id)
{
	...... // 中断处理
	return IRQ_HANDLED；				// IRQ_HANDLED在#include <linux/irqreturn.h>中定义
}
```

### 2.1.2 外部中断

```c
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/uaccess.h>
#include <mach/devices.h> 		//PAD_GPIO_A+n
#include <mach/soc.h> 			//nxp_soc_gpio_set_io_func();
#include <mach/platform.h> 		//PB_PIO_IRQ(PAD_GPIO_A+n);
#include <linux/gpio.h>
#include <linux/interrupt.h>	// 中断的api
#include <linux/irqreturn.h>	// 中断处理函数的返回值
#include <mach/irqs.h>			// 中断号
#include <linux/irq.h> 			// 外部中断的触发方式



#define DEVICE_NAME "mychardev"
#define DEVICE_CLASS_NAME "mycharcls"
#define DEVICE_COUNT 1

static dev_t dev;
static struct cdev my_cdev;
static struct class *my_class;


static void my_gpio_init(void);
// 中断处理函数
irqreturn_t keyboard_handler(int irq,void *dev_id);


static int my_open(struct inode *inode, struct file *file)
{
    printk(KERN_INFO "Device opened\n");
	printk(KERN_WARNING "major = %d, minor = %d\n", imajor(inode), iminor(inode));
    my_gpio_init();
    return 0;
}


static int my_release(struct inode *inode, struct file *file)
{
    printk(KERN_INFO "Device closed\n");
	free_irq(gpio_to_irq(PAD_GPIO_A+28), NULL);
    return 0;
}


static struct file_operations fops = {
    .owner = THIS_MODULE,
    .open = my_open,
    .release = my_release,
};

static void my_gpio_init(void)
{
	int ret;
    
    // 注册外部中断
	ret = request_irq(gpio_to_irq(PAD_GPIO_A+28), keyboard_handler, IRQ_TYPE_EDGE_FALLING, "KEYBOARD_IRQ", NULL);
    // gpio_to_irq(PAD_GPIO_A+28) 也可以使用 IRQ_GPIO_A_START+28代替
	
 }

// 中断处理函数
irqreturn_t keyboard_handler(int irq, void * dev_id){
	static int key_num = 0;
	printk(KERN_INFO "This is keyboard_handler: KEY DOWN %d \n", key_num++);
	return IRQ_HANDLED;		// IRQ_HANDLED在#include <linux/irqreturn.h>中定义
}



static int __init my_init(void)
{
    int ret;

    // 分配设备号
    ret = alloc_chrdev_region(&dev, 0, DEVICE_COUNT, DEVICE_NAME);
    if (ret < 0) {
        printk(KERN_ERR "Failed to allocate chrdev region\n");
        return ret;
    }

    // 初始化 cdev 结构体
    cdev_init(&my_cdev, &fops);
    my_cdev.owner = THIS_MODULE;

    // 将 cdev 添加到系统中
    ret = cdev_add(&my_cdev, dev, DEVICE_COUNT);
    if (ret < 0) {
        unregister_chrdev_region(dev, DEVICE_COUNT);
        printk(KERN_ERR "Failed to add cdev\n");
        return ret;
    }

    // 创建设备类
    my_class = class_create(THIS_MODULE, DEVICE_CLASS_NAME);
    if (IS_ERR(my_class)) {
        cdev_del(&my_cdev);
        unregister_chrdev_region(dev, DEVICE_COUNT);
        printk(KERN_ERR "Failed to create class\n");
        return PTR_ERR(my_class);
    }

    // 创建设备节点
    device_create(my_class, NULL, dev, NULL, DEVICE_NAME);

    printk(KERN_INFO "Device initialized successfully\n");
    return 0;
}

static void __exit my_exit(void)
{
    device_destroy(my_class, dev);
    class_destroy(my_class);
    cdev_del(&my_cdev);
    unregister_chrdev_region(dev, DEVICE_COUNT);
    printk(KERN_INFO "Device exited successfully\n");
}

module_init(my_init);
module_exit(my_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name");
MODULE_DESCRIPTION("A simple character device driver");

```

```c
// 测试程序test.c
#include<stdio.h>
#include<sys/types.h>
#include<sys/stat.h>
#include<fcntl.h>
#include<string.h>

void open_test(){
	int fd,ret;
	fd = open("/dev/mychardev", O_RDWR);
	while(1){
		usleep(300 * 1000);
	}
}

int main(int argc, char** argv){
	open_test();
    return 0;
}
```



```bash
# 虚拟机编译测试程序
arm-linux-gcc -o test test.c
# 虚拟机编译驱动
make clean
make

# 开发板卸载模块驱动
rmmod demo_module
lsmod
# 开发板安装模块驱动
insmod demo_module
# 开发板启动测试程序，后台执行，后面使用完毕后，记得杀掉
./test &
# 多次按键
[164911.658000] This is keyboard_handler: KEY DOWN 1 
[164911.659000] This is keyboard_handler: KEY DOWN 2 
[164911.874000] This is keyboard_handler: KEY DOWN 3 
[164914.771000] This is keyboard_handler: KEY DOWN 4 
[164915.481000] This is keyboard_handler: KEY DOWN 5 
[164916.104000] This is keyboard_handler: KEY DOWN 6 
[164916.620000] This is keyboard_handler: KEY DOWN 7 
[164917.078000] This is keyboard_handler: KEY DOWN 8 
[164917.562000] This is keyboard_handler: KEY DOWN 9 
[164917.760000] This is keyboard_handler: KEY DOWN 10 
[164918.048000] This is keyboard_handler: KEY DOWN 11 
[164918.271000] This is keyboard_handler: KEY DOWN 12 
[164919.022000] This is keyboard_handler: KEY DOWN 13 
[164919.559000] This is keyboard_handler: KEY DOWN 14 
[164919.752000] This is keyboard_handler: KEY DOWN 15 

# 查看中断统计
cat /proc/interrupts
          CPU0       CPU1       CPU2       CPU3       CPU4       CPU5       CPU6       CPU7       
 33:          0          0          0          0          0          0          0          0       GIC  pl08xdmac
 34:          0          0          0          0          0          0          0          0       GIC  pl08xdmac
 37:          0          0          0          0          0          0          0          0       GIC  rtc 1hz
 39:        110        333        256        341        619        654        572        574       GIC  nxp-uart
 48:       2857       3103       3187       3386       4446       3604       3096       3877       GIC  s3c2440-i2c.1
134:          0          5          2          2          1          4          2          1      GPIO  KEYBOARD_IRQ


```

嵌入式系统中**裸机**的中断服务特点：

- 没有返回值
- 没有参数
- 尽量不要进行浮点运算（处理尽量快）

### 2.1.3 中断底半部

在大多数真实的系统中，当中断来临时，要完成的工作往往不能立即完成，而是需要大量的耗时处理。

如果我们的系统，拥有一个单核的cpu，如果在中断处理函数中，有比较耗时的任务，那么将阻塞主进程上下文（或者说是进程间的调度）的执行。

```bash
# 查看某一核cpu的在线情况
cat /sys/devices/system/cpu/cpu1/online
1
# 关闭某一核cpu的运行
echo 0 > /sys/devices/system/cpu/cpu1/online
```

中断处理可以分两个部分：

- 顶半部：处理紧急的硬件操作（大家熟知的中断服务函数）。
-  底半部：处理不紧急的耗时操作，执行过程中，中断是使能的，可被打断。
  - 实现机制：
  - 软中断（softirq）：供内核使用的机制
  -  微线程（tasklet）：微线程通过软中断机制来调度
  - 工作队列等（workqueue）：工作队列将工作交由一个内核线程处理

一般遇到耗时任务，顶半部用于创建工作队列，初始化工作，将耗时任务交给内核线程去执行。

#### 工作队列

```c
// api接口

#include <linux/workqueue.h>

// 定义一个工作队列结构体指针
static struct workqueue_struct *key_workqueue;
// 创建工作队列
struct workqueue_struct *create_workqueue(char * queue_name);
// 销毁工作队列
void destroy_workqueue(struct workqueue_struct *);

// 创建工作
struct work_struct work;
// 工作初始化宏
INIT_WORK(work_struct* work, void (*func)());
// 添加工作到任务队列
int queue_work(struct workqueue_struct*wq, struct work_struct *work);

// 终止队列中的工作（即使处理程序已经在处理该任务）
int cancel_work_sync(struct work_struct *work);
int work_pending(struct work_struct work );
```

```c
// 功能：每按一次按键，num就加1，应用程序每次读取num的值。

static struct workqueue_struct *key_workqueue;
static struct work_struct key_work;
static int num = 0;

static int __init my_init(void)
{
	...
    //创建工作队列
    key_workqueue = create_workqueue("key_queue");
    // 初始化工作
    INIT_WORK(&key_work, key_work_func);
    
}
static ssize_t my_read(struct file *file, char __user *buf, size_t count, loff_t *ppos)
{
	static int old_num = -1;
    int len = min(count, sizeof(num));
    int ret;
  	if(old_num == num){
		return 0;
    }else{
        old_num = num;
        copy_to_user(buf, &num, len);
        return sizeof(num);
    }
    
}
// 中断处理函数，顶半部
irqreturn_t keyboard_handler(int irq, void * dev_id){
    disable_irq_nosync(IRQ_GPIO_A_START+28);
    // 添加工作
    queue_work(key_workqueue, &key_work);
	return IRQ_HANDLED;		// IRQ_HANDLED在#include <linux/irqreturn.h>中定义
}
static void key_work_func(struct work_struct *work){
    // 耗时任务
    
    //
    num++;
    enable_irq(IRQ_GPIO_A_START+28);
}
static void __exit my_exit(void)
{
   	...
    // 销毁工作队列
    destroy_workqueue(key_workqueue);
}
```

```c
// 读取按键的次数
// 测试程序

#include<stdio.h>
#include<sys/types.h>
#include<sys/stat.h>
#include<fcntl.h>
#include<string.h>
void read_test(){
    int fd=0;
    int ret=0;
    int key=0;

    fd = open("/dev/mychardev", O_RDWR);
    if(fd < 0){
        perror("/dev/mychardev");
        return -1;
    }
    
    // 轮询
    while(1){
        ret = read(fd, &key, sizeof(key));
        if(ret != 0){
            printf("key down = %d\n", key);
        }
    }
    close(fd);
    return;
}
int main(int argc, char** argv){
	read_test();
    return 0;
}
```

## 2.3 异步数据处理kfifo

数据的采集（驱动侧）与处理使用（应用程序侧）往往不同步，于是驱动编程中数据采集方需要将采集的数据暂时放到一个缓冲区中，使用方在需要处理数据时从缓冲区中将数据读出。

我们可以选择自己编写一个队列，也可以利用内核中现有队列kfifo来实现。

场景：内核产生消息，应用程序无法及时处理

```c
#include <linux/kfifo.h>

//kfifo结构体类型
struct kfifo {
	unsigned char *buffer; 	//存放数据的缓存区
	unsigned int size; 		//buffer空间大小
	unsigned int in; 		//指向buffer队尾
	unsigned int out; 		//指向buffer队头
};
// kfifo是一个循环队列（环状），in指针和out指针都沿着环逆时针移动，in往里加数据，加一个后移一个。out消费数据，消费一个后移一个
// 假设缓冲区大小size为8, 缓冲区读写下标分别为：in%size，out%size


// 申请kfifo空间
int kfifo_alloc(struct kfifo *fifo, unsigned int size, gfp_t gfp_mask);
// size：申请的空间大小，单位字节
// gfp_mask：内存标志位

// 释放kfifo
void kfifo_free(struct kfifo *fifo);

// 存数据
unsigned int kfifo_in(struct kfifo *fifo, const void *from, unsigned int len);
// 消费数据
unsigned int kfifo_out(struct kfifo *fifo, void *to, unsigned int len);
// from和to：写/读数据的首地址
// len：读写数据的大小

// 获取fifo内的已用数据个数
unsigned int kfifo_len(struct kfifo *fifo);
// 获取fifo总大小
unsigned int kfifo_size(struct kfifo *fifo);
// 检查kfifo是否为空
int kfifo_is_empty(struct kfifo *fifo);
// 检查kfifo是否为满
int kfifo_is_full(struct kfifo *fifo);
```

```c
// 功能：每按一次按键，num就加1，应用程序每次读取num的值。

static struct workqueue_struct *key_workqueue;
static struct work_struct key_work;
static int num = 0;
static struct kfifo key_fifo;

// 模块加载时去申请一块队列空间，模块卸载接口中释放申请的空间
static int __init my_init(void)
{
    int ret;
	...
    
    // 创建工作队列
    key_workqueue = create_workqueue("key_queue");
    // 初始化工作
    INIT_WORK(&key_work, key_work_func);
    
    //申请128字节fifo内存
    ret= kfifo_alloc(&key_fifo, 128, GFP_KERNEL);
    
}
static ssize_t my_read(struct file *file, char __user *buf, size_t count, loff_t *ppos)
{
    int len = min(count, sizeof(num));
    int ret;
    int data;
    // kfifo判空
    if(kfifo_is_empty(&key_fifo)) return 0;
    // 消费数据
    ret = kfifo_out(&key_fifo,&data,sizeof(data));
	ret = copy_to_user(buf, &data, len);
    return sizeof(len);
    
}
// 中断处理函数，顶半部
irqreturn_t keyboard_handler(int irq, void * dev_id){
    disable_irq_nosync(IRQ_GPIO_A_START+28);
    // 添加工作
    queue_work(key_workqueue, &key_work);
	return IRQ_HANDLED;		// IRQ_HANDLED在#include <linux/irqreturn.h>中定义
}
static void key_work_func(struct work_struct *work){
    int ret;
    // 耗时任务
    
    //
    num++;
    if(kfifo_is_full(&key_fifo)) return;
	ret = kfifo_in(&key_fifo, &num, sizeof(num));
    enable_irq(IRQ_GPIO_A_START+28);
}
static void __exit my_exit(void)
{
   	...
    // 销毁工作队列
    destroy_workqueue(key_workqueue);
	// 释放fifo
    kfifo_free(&key_fifo);
}
```

## 2.4 并发与同步

资源（硬件资源，全局变量，静态变量）有限，资源竞争

竞争产生的原因：

1. 抢占式内核：用户程序在执行系统调用期间可以被高优先级进程抢占
2. 多处理器SMP
3. 中断程序

内核中解决办法：

1. 信号量（semaphore）
2. 自旋锁（spinlock）
3. 原子变量（atomic）
4. 读写锁
5. 互斥体（mutex）

### 2.4.1 信号量

信号量**采用睡眠等待机制**。

中断服务函数不能进行睡眠，因此信号量不能用于中断当中，但可以使用后面介绍的自旋锁。

信号量资源开销比较大（因为程序睡眠时，保存上下文，唤醒时，恢复上下文，来回折腾比较麻烦）。

应用场景：

- 一般用于对公共资源访问频率比较低，资源占用时间比较长的场合
- 不可用在中断程序顶半部中

```c
#include <linux/semaphore.h>

// 定义一个信号量
struct semaphore my_sem;

// 初始化信号量
void sema_init(struct semaphore *sem, int val);
// val: 信号量的计数值

// 获取信号量(减操作)，在拿不到信号量的时候，会导致调用者睡眠（挂起），睡眠不可被系统消息中断。
// 也就是说，如果进入睡眠，并且没有收到其他地方释放信号量（up）的消息，那么这个进程将永远睡眠，无法被中断（CTRL + C 也不可以，只有关机才行）
void down(struct semaphore *sem);
// 获取信号量(减操作)，会导致调用者睡眠，但可以被系统消息中断
int down_interruptible(struct semaphore *sem);
// 尝试获取信号量,成功返回0,失败返回非0，不会导致调用者睡眠
int down_trylock(struct semaphore *sem);

// 释放信号量，即使信号量加1（如果线程睡眠，将其唤醒）
void up(struct semaphore *sem);
```

```c

// 功能：每按一次按键，num就加1，应用程序每次读取num的值。

#include<linux/semaphore.h> 	// 信号量

static struct workqueue_struct *key_workqueue;
static struct work_struct key_work;
static int num = 0;
static struct kfifo key_fifo;
static struct semaphore my_sem;

// 模块加载时去申请一块队列空间，模块卸载接口中释放申请的空间
static int __init my_init(void)
{
    int ret;
	...
    
    // 创建工作队列
    key_workqueue = create_workqueue("key_queue");
    // 初始化工作
    INIT_WORK(&key_work, key_work_func);
    
    //申请128字节fifo内存
    ret= kfifo_alloc(&key_fifo, 128, GFP_KERNEL);
    
    sema_init(&my_sem, 1);
    
}
static ssize_t my_read(struct file *file, char __user *buf, size_t count, loff_t *ppos)
{
    int len = min(count, sizeof(num));
    int ret;
    int data;
    // kfifo判空
    if(kfifo_is_empty(&key_fifo)) return 0;
    
    // 消费数据
    down(&my_sem);
    ret = kfifo_out(&key_fifo,&data,sizeof(data));
    up(&my_sem);
    
	ret = copy_to_user(buf, &data, len);
    return sizeof(len);
    
}
// 中断处理函数，顶半部
irqreturn_t keyboard_handler(int irq, void * dev_id){
    disable_irq_nosync(IRQ_GPIO_A_START+28);
    // 添加工作
    queue_work(key_workqueue, &key_work);
	return IRQ_HANDLED;		// IRQ_HANDLED在#include <linux/irqreturn.h>中定义
}
static void key_work_func(struct work_struct *work){
    int ret;
    // 耗时任务
    
    //
    num++;
    if(kfifo_is_full(&key_fifo)) return;
    
    down(&my_sem);
	ret = kfifo_in(&key_fifo, &num, sizeof(num));
    up(&my_sem);
    
    enable_irq(IRQ_GPIO_A_START+28);
}
static void __exit my_exit(void)
{
   	...
    // 销毁工作队列
    destroy_workqueue(key_workqueue);
	// 释放fifo
    kfifo_free(&key_fifo);
}
```

### 2.4.2 自旋锁

尝试获取一个自旋锁，如果锁空闲就获取该自旋锁并继续向下执行；如果锁已被占用就循环检测该锁是否被释放（原地打转直到锁被释放）

使用忙等待机制，自旋锁资源开销小。

场景：

- 一般用于对公共资源访问频率比较低，资源占用时间比较长的场合
- 可以用在中断服务程序中（中断顶半部）

```c
#include <linux/spinlock.h>

// 定义自旋锁变量
struct spinlock my_spinlock;
// 或
spinlock_t my_spinlock;

// 自旋锁初始化
spin_lock_init(&my_spinlock);

// 获得自旋锁（可自旋等待，可被软、硬件中断）
void spin_lock(spinlock_t *my_spinlock);
// 释放自旋锁，退出临界区
void spin_unlock(spinlock_t *lock);

// 获得自旋锁(可自旋等待，保存中断状态并关闭软、硬件中断)
void spin_lock_irqsave(spinlock_t *my_spinlock,unsigned long flags);

// 释放自旋锁，退出临界区后，恢复中断
void spin_unlock_irqrestore(spinlock_t *lock,unsigned long flags);

// 尝试获得自旋锁（不自旋等待，成功返回1、失败则返回0）
int spin_trylock(spinlock_t *lock)
```

## 2.5 定时和延时

内核全局变量：

- HZ：为每秒的定时器的节拍数，HZ是一个与体系结构相关的常数，Linux为大多数平台提供HZ值范围为50-1200，x86 PC平台默认为1000，我们的内核为1000
- jiffies：用来记录自内核启动以来的时钟滴答总数（即每隔1/HZ秒加1）

```c
// 延时
#include <linux/delay.h>
// 忙等待延时函数，一般不太长的时间可以用它
void ndelay(unsigned long nsecs); //纳秒级延时
void udelay(unsigned long usecs); //微秒级延时
void mdelay(unsigned long msecs); //毫秒级延时

// 睡眠等待延时函数
void msleep(unsigned int millisecs); 	// 毫秒级延时
unsigned long msleep_interruptible(unsigned int millisecs);		// 毫秒级延时，可提前（被系统消息）唤醒
void ssleep(unsigned int seconds);		//秒级延时
    
// 定时器
// 内核定时器可在未来的某个特定时间点调度执行某个函数，完成指定任务
// 假设HZ的值为1000，Linux定时器最短定时时间为1ms，小于该时间的定时需要选择精度更高的定时器或直接硬件定时
#include <linux/timer.h>
struct timer_list
{
	struct list_head entry;
	//链表节点，由内核管理
	unsigned long expires;
	//定时器到期时间（指定一个时刻）
	void (*function)(unsigned long)；
	// 定时器处理函数
	unsigned long data;
	// 作为参数被传入定时器处理函数
	......
};

// 初始化定时器
void init_timer(struct timer_list *timer);
// 添加定时器。定时器开始计时
void add_timer(struct timer_list * timer);
// 删除定时器,在定时器到期前禁止一个已注册定时器
int del_timer(struct timer_list * timer);
// 如果定时器函数正在执行则在函数执行完后返回(SMP)
int del_timer_sync(struct timer_list *timer);

// 更新定时器到期时间，并开启定时器
int mod_timer(struct timer_list *timer, unsigned long expires);
// 查看定时器是否正在等待被调度运行
int timer_pending(const struct timer_list *timer);			// 返回值为真表示正在等待被调度运行
```

```c

static struct timer_list timer;
static void my_delay(int ms){
    unsigned int old_time = jiffies;
    while((jiffies - old_time) <= ms);		// old_time有溢出的风险
}
static void time_fun(unsigned long data){
    static int i = 0;
    i++;
    printk(KERN_INFO "%02d:%02d\n",i/60, i%60);
    
    // 循环执行
    mod_timer(&timer,jiffies + 1000);
}
static int my_open(struct inode *inode, struct file *file)
{
    
    printk(KERN_INFO "HZ = %d, jiffies = %ld", HZ, jiffies);
    my_delay(3000);			// 延时3秒
    printk(KERN_INFO "HZ = %d, jiffies = %ld", HZ, jiffies);
    
    // 在1000ms后执行
    timer.expires = jiffies + 1000;
    
    // 启动定时器
    add_timer(&timer);
    return 0;
}

static int my_release(struct inode *inode, struct file *file)
{
    printk(KERN_INFO "Device closed\n");
    
    // 删除定时器
	del_timer(&timer);
    return 0;
}
static ssize_t my_read(struct file *file, char __user *buf, size_t count, loff_t *ppos)
{
    return 1;		// cat /dev/mychrdev 就不会退出
}
static int __init my_init(void)
{
    // 绑定定时器任务
    timer.function = time_fun;
    // 初始化定时器
    init_timer(&timer);
}
```

### 实例

- 按键去抖：延时
- up/down：中断采用双沿触发
- 长短时按键：

```c
static struct key_info{
    int status;		//up=0, down=1
    int type;		//short=0, long=1
    int code;		//键值
}

struct key_info key_val = {
    .status = 0,
    .type = 0,
    .code = 0,
}

static void key_work_func(struct work_struct *work){
    int ret;
    // 区分up/down，前提是要打开双沿触发
    if(key_val.status == 0){		//down
        // 去抖30ms
        mdelay(30);
        if(nxp_soc_gpio_get_in_value(PAD_GPIO_A+28) != 0 ){
			enable_irq(IRQ_GPIO_A_START+28);
            return;
        }
        num++;
        key_val.status = 1;
        key_val.type = 0;
        
        // 区分长短按计时器
        timer.expires = jiffies + HZ*1;
        add_timer(&timer);
    }else{							//up
        key_val.status = 0;
        // 如果up，就要删除计时器
        del_timer_sync(&timer);
    }
    if(!kfifo_is_full(&key_fifo)){
       key_val.code = num;
       down(&my_sem);
       ret = kfifo_in(&key_fifo,&key_val, sizeof(key_val));
       up(&my_sem);
    }
}

static ssize_t my_read(struct file *file, char __user *buf, size_t count, loff_t *ppos)
{
    int len = min(count,sizeof(key_val));
    int ret;
    struct key_info data;
    
    if(kfifo_is_empty(&key_fifo))	return 0;
    down(&my_sem);
    ret = kfifo_out(&key_fifo,&data,sizeof(data));
    up(&my_sem);
    ret = copy_to_user(buf,&data,len);
    return sizeof(data);
}

static void time_fun(unsigned long data){
    int ret;
    key_val.type = 1;		// 达到长按键周期
    if(!kfifo_is_full(&key_fifo)){
       down(&my_sem);
       ret = kfifo_in(&key_fifo,&key_val, sizeof(key_val));
       up(&my_sem);
    }
    mod_timer(&timer,jiffies + 100);		// 达到长按键后，连续输出的时间间隔100ms
}
```

```c
#include<stdio.h>
#include<sys/types.h>
#include<sys/stat.h>
#include<fcntl.h>
#include<string.h>

static struct key_info{
    int status;		//up=0, down=1
    int type;		//short=0, long=1
    int code;		//键值
};
void read_test(){
    int fd=0;
    int ret=0;
	struct key_info key={0,0,0};
    
    fd = open("/dev/mychardev", O_RDWR);
    if(fd < 0){
        perror("/dev/mychardev");
        return -1;
    }
    
    // 轮询
    while(1){
        ret = read(fd, &key, sizeof(key));
        if(ret != 0){
            printf("%s: %s : = %d\n", key.status ? "down":"up",key.type ? "long":"short",, key.code);
        }
    }
    close(fd);
    return;
}
int main(int argc, char** argv){
	read_test();
    return 0;
}
```

## 2.6 I/O阻塞与非阻塞

阻塞：让出cpu，挂起（睡眠）

非阻塞：不出让cpu，不睡眠，轮询

**也就是要达到，应用程序发起系统调用read，如果驱动里面没有数据供读，那么应用程序就一直阻塞在read那里，不会执行下一行代码，这就是阻塞。**

方案：

- 等待队列
- 轮询加阻塞操作

### 2.6.1 等待队列

```c
#include <linux/wait.h>

// 定义一个等待队列
wait_queue_head_t my_queue;
// 初始化一个等待队列头
init_waitqueue_head(&my_queue);
// 合并前面两个步骤，定义并初始化一个等待队列头
DECLARE_WAIT_QUEUE_HEAD(my_queue);

// 无条件阻塞
sleep_on(wait_queue_head_t *q);												//直接阻塞，不可中断
interruptible_sleep_on(wait_queue_head_t *q);								//直接阻塞，可中断
long sleep_on_timeout(wait_queue_head_t *q, long timeout);					//不可中断，可超时
long interruptible_sleep_on_timeout(wait_queue_head_t *q, long timeout);	//可中断，可超时
// interruptible函数要成对使用

// 有条件阻塞
wait_event(wait_queue_head_t wq, int condition);							//condition=0进入阻塞
wait_event_interruptible(wait_queue_head_t wq,int condition);
wait_event_timeout(wait_queue_head_t wq, int condition, long timeout);
wait_event_interruptiblble_timeout(wait_queue_head_t wq,int condition, long timeout);
// wake_up唤醒进程之前要将wait_event中的condition变量的值赋为真，否则该进程被唤醒后会立即再次进入睡眠

// 唤醒阻塞进程
wake_up(wait_queue_head_t *wq);
wake_up_interruptible(wait_queue_head_t *wq);
```

```c
#include <linux/wait.h>

DECLARE_WAIT_QUEUE_HEAD(key_queue);
static ssize_t my_read(struct file *file, char __user *buf, size_t count, loff_t *ppos)
{
    int len = min(count,sizeof(key_val));
    int ret;
    struct key_info data;
    
    // 判断应用程序那里采用的是以什么方式的open
    // 如果应用程序那里的open("/dev/mychardev", O_RDWR | O_NONBLOCK)，那就采用非阻塞模式
    // open("/dev/mychardev", O_RDWR)，那就采用阻塞模式
    if((file->f_flags & O_NONBLOCK) == 0){
        if(kfifo_is_empty(&key_fifo)){
            // 无条件阻塞，将条件写到if里面了
            sleep_on(&key_queue);
        }
        // 有条件阻塞
        // wait_event(key_queue, !kfifo_is_empty(&key_fifo));
        
    }
    
    down(&my_sem);
    ret = kfifo_out(&key_fifo,&data,sizeof(data));
    up(&my_sem);
    ret = copy_to_user(buf,&data,len);
    return sizeof(data);
}

static void key_work_func(struct work_struct *work){
    int ret;
    // 区分up/down，前提是要打开双沿触发
    if(key_val.status == 0){		//down
        // 去抖30ms
        mdelay(30);
        if(nxp_soc_gpio_get_in_value(PAD_GPIO_A+28) != 0 ){
			enable_irq(IRQ_GPIO_A_START+28);
            return;
        }
        num++;
        key_val.status = 1;
        key_val.type = 0;
        
        // 区分长短按计时器
        timer.expires = jiffies + HZ*1;
        add_timer(&timer);
    }else{							//up
        key_val.status = 0;
        // 如果up，就要删除计时器
        del_timer_sync(&timer);
    }
    
    
    if(!kfifo_is_full(&key_fifo)){
       key_val.code = num;
       down(&my_sem);
       ret = kfifo_in(&key_fifo,&key_val, sizeof(key_val));
       up(&my_sem);
	   //唤醒阻塞	        
       wake_up(&key_queue);
    }
}

static void time_fun(unsigned long data){
    int ret;
    key_val.type = 1;		// 达到长按键周期
    if(!kfifo_is_full(&key_fifo)){
       down(&my_sem);
       ret = kfifo_in(&key_fifo,&key_val, sizeof(key_val));
       up(&my_sem);
       wake_up(&key_queue);
    }
    mod_timer(&timer,jiffies + 100);		// 达到长按键后，连续输出的时间间隔100ms
}
```



### 2.6.2 轮询加阻塞操作

- 一个用户进程可以**实现多个设备驱动**的同时监听
- 应用程序需要定义一个集合来保存所有打开的设备
- 通过select()对多个设备进行阻塞监听，而内核对设备进行轮询
- poll()接口主要是协助内核完成驱动可操作性的监听工作（轮询）
- poll()接口被select()调用时将等待队列加入内核轮询列表，当唤醒时，内核会**再次**调用poll()接口

```c
// 应用层接口
#include <sys/select.h>

// 文件描述符集合的变量的定义
fd_set fds;
// 清空描述符集合
FD_ZERO(fd_set *set);
// 加入一个文件描述符到集合中
FD_SET(int fd, fd_set *set);
// 从集合中清除一个文件描述符
FD_CLR(int fd, fd_set *set);

// 判断文件描述符是否被置位
FD_ISSET(int fd, fd_set *set);
// 这里返回非零，表示置位（该文件描述集合中有文件可进行读写操作，或产生错误）

int select(
    int numfds,			
	fd_set *readfds,
	fd_set *writefds,
	fd_set *exceptfds,
	struct timeval *timeout
);
// numfds：需要监听的文件描述符的个数 + 1，最大支持FD_SETSIZE=1024
// readfds：需要监听读属性变化的文件描述符s集合
// writefds：需要监听写属性变化的文件描述符集合
// exceptfds：需要监听异常属性变化的文件描述符集合
// timeout：超时时间，表示等待多长时间之后就放弃等待
		//传 NULL 表示等待无限长的时间，持续阻塞直到有事件就绪才返回。
    	// 大于0，超时时间
    	// =0，不等待立即返回
		/* 
		struct timeval{
        	long tv_sec;//秒
        	long tv_usec;//微秒
    	}
    	*/
//返回值：变化的文件描述符个数



// poll()函数：为file_operation成员之一
static unsigned int poll(struct file *file, struct poll_table_struct *wait);
// file:是文件结构指针
// wait:轮询表指针，管理着一系列等待列表
// 以上两个参数是由内核传递给poll函数

// poll函数返回的状态掩码
/*
可读状态掩码:
	POLLIN:有数据可读
	POLLRDNORM:有普通数据可读
	POLLRDBAND:有优先数据可读
	POLLPRI:有紧迫数据可读
可写状态掩码:
	POLLOUT:写数据不会导致阻塞
	POLLWRNORM:写普通数据不会导致阻塞
	POLLWRBAND:写优先数据不会导致阻塞
	POLLMSG/SIGPOLL:消息可用
错误状态掩码:
	POLLER:指定的文件描述符发生错误
	POLLHUP:指定的文件描述符挂起事件
	POLLNVAL:指定的文件描述符非法
*/

// 添加等待队列到wait参数指定的轮询列表中
void poll_wait(struct file *filp, wait_queue_heat_t *wq, poll_table *wait);
// poll_wait()将可能引起文件状态变化的进程添加到轮询列表，由内核去监听进程状态的变化，不会阻塞进程
// 一旦进程有变化(wake_up)，内核就会自动去调用poll()，而poll()是返回给select()的
// 所以当进程被唤醒以后，poll()应该将状态掩码返回给select()，从而select()退出阻塞。
// 完成一次监测，poll函数被调用一次或两次
//      第一次为用户执行select函数时被执行
// 		第二次调用poll为内核监测到进程的wake_up操作时或进程休眠时间到唤醒再或被信号唤醒时
```



```c
#include<sys/select.h>

void read_test(){
    int fd=0;
    fd_set fds;
    int ret=0;
	struct key_info key={0,0,0};
    
    FD_ZERO(&fds);
    
    fd = open("/dev/mychardev", O_RDWR);
    if(fd < 0){
        perror("/dev/mychardev");
        return -1;
    }
    
    FD_SET(fd, &fds);
    
    // 轮询
    while(1){
        // select是一个系统调用，在驱动中会有一个poll跟他相对应
        ret = select(fd +1,&fds,NULL,NULL,NULL);
        if(ret > 0 && FD_ISSET(fd)){
            ret = read(fd, &key, sizeof(key));
            printf("%s: %s : = %d\n", key.status ? "down":"up",key.type ? "long":"short",, key.code);
        }
    }
    close(fd);
    return;
}
```

```c
DECLARE_WAIT_QUEUE_HEAD(select_queue);

unsigned int my_poll (struct file *pfile, struct poll_table_struct *ptable){
    poll_wait(pfile, &select_queue, ptable);
    return kfifo_is_empty(&key_fifo)? 0:POLLIN;		// 0表示没有消息，POLLIN表示有消息
}

static void key_work_func(struct work_struct *work){
    int ret;
    // 区分up/down，前提是要打开双沿触发
    if(key_val.status == 0){		//down
        // 去抖30ms
        mdelay(30);
        if(nxp_soc_gpio_get_in_value(PAD_GPIO_A+28) != 0 ){
			enable_irq(IRQ_GPIO_A_START+28);
            return;
        }
        num++;
        key_val.status = 1;
        key_val.type = 0;
        
        // 区分长短按计时器
        timer.expires = jiffies + HZ*1;
        add_timer(&timer);
    }else{							//up
        key_val.status = 0;
        // 如果up，就要删除计时器
        del_timer_sync(&timer);
    }
    
    
    if(!kfifo_is_full(&key_fifo)){
       key_val.code = num;
       down(&my_sem);
       ret = kfifo_in(&key_fifo,&key_val, sizeof(key_val));
       up(&my_sem);
	   //唤醒阻塞	        
       wake_up(&key_queue);
        
       wake_up(&select_queue);
    }
}

static void time_fun(unsigned long data){
    int ret;
    key_val.type = 1;		// 达到长按键周期
    if(!kfifo_is_full(&key_fifo)){
       down(&my_sem);
       ret = kfifo_in(&key_fifo,&key_val, sizeof(key_val));
       up(&my_sem);
       wake_up(&key_queue);
        
       wake_up(&select_queue);
    }
    mod_timer(&timer,jiffies + 100);		// 达到长按键后，连续输出的时间间隔100ms
}

static struct file_operations fops = {
    .owner = THIS_MODULE,
    .open = my_open,
    .release = my_release,
    .read = my_read,
    .write = my_write,
    .poll = my_poll,
};
```

## 2.7 内核线程

```c
#include <linux/kthread.h>

//定义线程指针
struct task_struct *kernel_thread;
// 创建内核线程
struct task_struct *kthread_create(int (*threadfn)(void *data), void *data, const char namefmt[], ...);
// threadfn：现成函数指针, 该函数必须能让出CPU，以便其他线程能够得到执行，
// data: 函数参数
// namefmt：线程名称，这个函数可以像printk一样传入某种格式的线程名
// 启动线程
int wake_up_process(struct task_struct *p);

// 创建并启动线程
kthread_run(threadfn, data, namefmt, ...);

// 停止线程检测函数 （线程函数内使用），接收现成函数外kthread_stop发送的停止信号
int kthread_should_stop(void);
// 停止内核线程函数 (线程函数外使用)，给线程函数发送停止信号，线程函数内部通过kthread_should_stop接收停止信号后，返回真
int kthread_stop(struct task_struct *k);
// 如果线程函数内部没有kthread_should_stop接收停止信号 或者 线程函数不结束，那么此函数将一直等待下去
```



# 3 工程化

## 3.1 platform总线

总线：各部件之间传递信息的公共通道

一个现实的linux设备和驱动通常需要挂载在一种总线上，这条总线可以是：

- 物理总线：USB/PCI/I2C/SPI总线
- 虚拟总线：Platform总线（触摸屏、LCD…）

我们之前的代码将资源设备（gpio口）和驱动（逻辑）放在一起，这样会导致资源设备和驱动缺乏相互独立性，给管理维护和移植带来诸多不便。

总线目的：**让设备驱动和设备资源更加独立且统一（总线是二者的中间层）**，**使得设备驱动程序更加通用**

platform机制开发并不复杂，由**三部分**组成：

- platform_device：
  - 是用来描述当前驱动使用的平台硬件信息，一般情况定义在厂家提供的板级支持包中。
  - 我们当前平台的硬件资源位于：arch/arm/plat-s5p6818/x6818/device.c和devices.c
- platfrom_driver：
  - 驱动具体的操作接口，是一些针对当前成功匹配设备的操作函数接口实现。

- platform总线：是系统内核创建的：**platform_bus**

### api

```c
#include<linux/platform_device.h>

// 设备资源相关结构体
struct platform_device {
	const char *name; 								// 资源名字
	int id; 										// 一般写0或-1
	struct device dev;
	u32 num_resources;								// 资源大小
	struct resource *resource; 						// 资源
	const struct platform_device_id *id_entry;
	struct pdev_archdata archdata;
};
struct device {
	void (*release)(struct device *dev);
	void *platform_data;
	... ...
}
struct resource {
	resource_size_t start; 							// 资源起始的物理地址
	resource_size_t end; 							// 资源结束的物理地址
	const char *name;
	unsigned long flags; 							// 资源类型IO资源：IORESOURCE_MEM, 中断号：IORESOURCE_IRQ等等
	struct resource *parent, *sibling, *child;
};

// 驱动相关结构体
struct platform_driver {
    int (*probe)(struct platform_device *); 						//匹配原来的init回调函数
	int (*remove)(struct platform_device *);						//匹配原来的exit回调函数
	void (*shutdown)(struct platform_device *);
	int (*suspend)(struct platform_device *, pm_message_t state);
	int (*resume)(struct platform_device *);
	struct device_driver driver;
	const struct platform_device_id *id_table;
};
struct device_driver {
	struct module *owner; 					//填写THIS_MODULE
	const char *name; 						//驱动名字，要和platform_device里的name相一致
};


// 驱动侧获取设备资源
struct resource *platform_get_resource(
	struct platform_device *dev,
	unsigned int type,
	unsigned int num
);
// dev: 内核传过来platform_device的指针
// type: 资源类型，与device的flag对应
// num: 同类资源序号

// 这个函数会在probe回调函数中调用，内核会向probe回调函数传platform_device，而这个变量正是platform_get_resource所需要的
// 通过这个函数获取到资源后，赋值给全局变量，然后在所有需要使用设备的地方使用全局变量代替。
```

### makefile

```makefile
# 加上你的device资源设备代码
obj-m += driver.o device.o
KERNELDIR := /home/buntu/sambaShare/kernel-3.4.39
PWD := $(shell pwd)

modules:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules
	rm -rf *.order *.mod.* *.o *.symvers

clean:
	make -C $(KERNELDIR) M=$(PWD) clean
	rm -rf *.ko
```

### device.c

```c

#include <linux/module.h>	/* module_init */
#include <linux/fs.h>	/* file_operations */
#include <linux/device.h>	/* class device */
#include <linux/sched.h>		/* current */
#include <linux/mount.h>		/* struct vfsmount */
#include <asm/io.h>	/* writel() */
#include <linux/uaccess.h> /* copy_to_user() */
#include <mach/devices.h> 	//PAD_GPIO_A+n
#include <mach/soc.h> 		//nxp_soc_gpio_set_io_func();
#include <mach/platform.h>	//PB_PIO_IRQ(PAD_GPIO_A+n);
#include <linux/interrupt.h>	/*request_irq*/
#include <linux/irq.h>	/*set_irq_type*/
#include <linux/delay.h> /* mdelay() */
#include <linux/kfifo.h> /* kfifo */
#include <linux/poll.h> /* poll */
#include <linux/kthread.h> /* kthread */
#include <linux/cdev.h>
#include <linux/platform_device.h>
#include <linux/kernel.h>
#include <linux/irq.h>
#include <asm/irq.h>


#define DRIVER_NAME 		"demo_driver"
#define DEVICE_NAME 		"demo_dev"
#define DEMO_PLATFORM_NAME 		"demo_platform"
#define DEVICE_COUNT 	5

static struct resource demo_resource[] = {

	[0] = {/* key row_1 */
		.start = PAD_GPIO_A+28,
		.end = PAD_GPIO_A+28,
		.flags = IORESOURCE_IO,
	},
	[1] = {/* key irq_row_1 */
		.start = PAD_GPIO_C+14,
		.end = PAD_GPIO_C+14,
		.flags = IORESOURCE_IO,
	}
};

static void demo_release(struct device *dev) 
{
	printk(KERN_WARNING "%s\n",__FUNCTION__);
	return ;
}

static struct platform_device demo_pdev = {
	.name = DEMO_PLATFORM_NAME,
	.id = 0,
	.num_resources = ARRAY_SIZE(demo_resource), 
	.resource = demo_resource,
	.dev = {
		.platform_data = NULL,
        .release = demo_release,
        
	}
};


static int  __init demo_dev_init(void)
{
    printk(KERN_WARNING "%s\n",__FUNCTION__);
	return platform_device_register(&demo_pdev);
}

static void  __exit demo_dev_exit(void)
{
    printk(KERN_WARNING "%s\n",__FUNCTION__);
	platform_device_unregister(&demo_pdev);
}

module_init(demo_dev_init);
module_exit(demo_dev_exit);

MODULE_LICENSE("GPL");	
MODULE_AUTHOR("qin");
MODULE_DESCRIPTION("used for studing linux drivers");
```





### driver.c

```c
// driver.c

#define DEMO_PLATFORM_NAME 		"demo_platform"

struct key_dev_struct_t{
    resource_size_t key_io;			// 按键	
    resource_size_t key_beep;		//蜂鸣器
}
static key_dev_struct_t key_dev = {
    .key_io = 0,
    .key_beep = 0,
}

// 曾经的demo_module_init修改为demo_probe
static int _devinit demo_probe(struct platform_device *pdev)
{
    int ret;
	struct resource *res;
    // 获取platform设备资源
    
    // 按键资源
    res = platform_get_resource(pdev,IORESOURCE_IO, 0);
    key_dev.key_io = res->start;
    // 蜂鸣器
    res = platform_get_resource(pdev,IORESOURCE_IO, 1);
    key_dev.key_beep = res->start;
    
    // 在所有需要使用设备的地方，用这两个全局变量代替
    
    // 分配设备号
    ret = alloc_chrdev_region(&dev, 0, DEVICE_COUNT, DEVICE_NAME);
    if (ret < 0) {
        printk(KERN_ERR "Failed to allocate chrdev region\n");
        return ret;
    }

    // 初始化 cdev 结构体
    cdev_init(&my_cdev, &fops);
    my_cdev.owner = THIS_MODULE;

    // 将 cdev 添加到系统中
    ret = cdev_add(&my_cdev, dev, DEVICE_COUNT);
    if (ret < 0) {
        unregister_chrdev_region(dev, DEVICE_COUNT);
        printk(KERN_ERR "Failed to add cdev\n");
        return ret;
    }

    // 创建设备类
    my_class = class_create(THIS_MODULE, DEVICE_CLASS_NAME);
    if (IS_ERR(my_class)) {
        cdev_del(&my_cdev);
        unregister_chrdev_region(dev, DEVICE_COUNT);
        printk(KERN_ERR "Failed to create class\n");
        return PTR_ERR(my_class);
    }

    // 创建设备节点
    device_create(my_class, NULL, dev, NULL, DEVICE_NAME);

    printk(KERN_INFO "Driver initialized successfully\n");
    return 0;
}

// 曾经的demo_module_exit修改为demo_remove
static int demo_remove(struct platform_device *dev){
    device_destroy(my_class, dev);
    class_destroy(my_class);
    cdev_del(&my_cdev);
    unregister_chrdev_region(dev, DEVICE_COUNT);
    printk(KERN_INFO "Driver exited successfully\n");
}

static int  __init demo_module_init(void)
{
    printk(KERN_WARNING "%s\n",__FUNCTION__);
	return platform_driver_register(&demo_driver);
}

static void  __exit demo_module_exit(void)
{
    printk(KERN_WARNING "%s\n",__FUNCTION__);
	platform_driver_unregister(&demo_driver);
}

static struct platform_driver demo_driver = {
    .driver = {
		.owner = THIS_MODULE,
		.name = DEMO_PLATFORM_NAME,
	},
	.probe = demo_probe,
	.remove = demo_remove,
	
};

module_init(demo_module_init);
module_exit(demo_module_exit);
```



```bash
# 在安装了device.ko和driver.ko后，可以在以下两个文件夹中看到对应的文件
ls /sys/bus/platform/devices
ls /sys/bus/platform/drivers
```

## 3.2 [input子系统](https://blog.csdn.net/weixin_42031299/article/details/125111946)

上层应用不知道定义什么数据类型去接驱动返回的值。

input以“兼容并包”的大一统思想，把这些输入设备的键码、键值、上传方式都分类统一起来，提高驱动通用性(linux和andriod都适用)，减少应用和驱动开发者的沟通成本。

主要是输入上报数据的标准化，标准化成了input_event。

这里面不需要再使用file_operations，应用程序中也不会使用read，write接口

![image-20240705164001041](./legend/image-20240705164001041.png)

```c
#include<linux/input.h>

// 申请、初始化input_dev
struct input_dev *input_allocate_device(void);
// 注册input_dev
int input_register_device(struct input_dev *dev);
// 注销input_dev
void input_unregister_device(struct input_dev *dev);

struct input_dev {
	const char *name;											//设备名称
	const char *phys;											//设备在系统中的物理路径
	const char *uniq;											//设备唯一识别符
    unsigned long evbit[BITS_TO_LONGS(EV_CNT)];					//设备支持的事件类型
    unsigned long keybit[BITS_TO_LONGS(KEY_CNT)];				//设备支持的具体的按键、按钮事件
	unsigned long relbit[BITS_TO_LONGS(REL_CNT)]; 				//户设备支持的具体的相对坐标事件
	unsigned long absbit[BITS_TO_LONGS(ABS_CNT)];				//设备支持的具体的绝对坐标事件
	unsigned long mscbit[BITS_TO_LONGS(MSC_CNT)];				//设备支持的具体的混杂事件
	unsigned long ledbit[BITS_TO_LONGS(LED_CNT)];				//设备支持的具体的LED指示灯事件
	unsigned long sndbit[BITS_TO_LONGS(SND_CNT)];				//户设备支持的具体的音效事件
	unsigned long ffbit[BITS_TO_LONGS(FF_CNT)];					//设备支持的具体的力反馈事件
	unsigned long swbit[BITS_TO_LONGS(SW_CNT)];					//设备支持的具体的开关事件
	....
}
```



**模仿usbkbd.c开发**

```c
//

#include<linux/input.h>

static struct input_dev *demo_input;
static int _devinit demo_probe(struct platform_device *pdev)
{
    int ret;
	struct resource *res;
    // 获取platform设备资源
    
    // 按键资源
    res = platform_get_resource(pdev,IORESOURCE_IO, 0);
    key_dev.key_io = res->start;
    // 蜂鸣器
    res = platform_get_resource(pdev,IORESOURCE_IO, 1);
    key_dev.key_beep = res->start;
    
    // 在所有需要使用设备的地方，用这两个全局变量代替
    
    ...//
    
    demo_input = input_allocate_device();
    demo_input->name = "demo_key_input";
    demo_input->phys = "demo_key_phy";
    demo_input->evbit[0] = BIT_MASK(EV_KEY) | BIT_MASK(EV_SYN);
    // 只有注册了可支持的按键才能在后面上报
    set_bit(KEY_A, demo_input->keybit);			// 在本文件中所有用到num的地方，使用KEY_A代替。
    // demo_input->open = demo_input_open;		// demo_input在注册input_register_device的时候调用这个open
    // demo_input->close = demo_input_close;	// demo_input在注册input_unregister_device的时候调用这个close
    
    input_register_device(&demo_input);
}
static int demo_remove(struct platform_device *dev){
    device_destroy(my_class, dev);
    class_destroy(my_class);
    cdev_del(&my_cdev);
    unregister_chrdev_region(dev, DEVICE_COUNT);
    input_unregister_device(&demo_input);
    printk(KERN_INFO "Driver exited successfully\n");
}
static int thread_ops_key(void *pdata){
    while(1) {
        if(kfifo_is_empty(&key_fifo)) continue;
        spin_lock(&my_spinlock);
        ret = kfifo_out(&key_fifo,&key_val,sizeof(key_val));
        // 第二个参数可以是KEY_A,KEY_B,但前提是你要注册这些按键
        input_report_key(demo_input, key_val.code, key_val.status);
        input_sync(demo_input);
    }
}
```

```bash
# 在安装模块之前先看看input的设备有哪些
ls /dev/input
# 安装driver.ko后，就可以看到ls /dev/input多个一个eventx
# 然后，就可以看到对应名字的input
cat /sys/class/input/eventx/device/name
demo_key_input
cat /sys/class/input/eventx/device/phys
demo_key_phy

cat /dev/input/eventx
hd /dev/input/eventx

```

```bash
# 如果我们想看到屏幕打出A来，我们需要
ps
PID   USER     TIME   COMMAND
    1 root       0:02 {linuxrc} init
    2 root       0:00 [kthreadd]
    3 root       0:00 [ksoftirqd/0]
    4 root       0:00 [kworker/0:0]
    6 root       0:00 [migration/0]
    7 root       0:00 [watchdog/0]
    8 root       0:00 [migration/1]
    9 root       0:00 [kworker/1:0]
   10 root       0:00 [ksoftirqd/1]
   11 root       0:06 [watchdog/1]
  109 root       0:00 [ext4-dio-unwrit]
  123 root       0:00 /usr/sbin/telnetd
  124 root       0:00 /usr/bin/tcpsvd 0 21 ftpd -w /home
  128 root       0:00 /usr/boa/boa
  145 root       0:00 -/bin/ash			# 如果COMMAND列中命令首部有一个短横线，通常表示该进程是前台进程组的成员，这就是当前的bash控制台的
 2375 root       0:00 [kworker/6:0]
 2429 root       0:00 [kworker/6:1]

ls /proc/145/fd -l
total 0
lrwx------    1 root     root            64 Jan  5 00:20 0 -> /dev/console		# 0表示标准输入
lrwx------    1 root     root            64 Jan  5 00:20 1 -> /dev/console		# 1表示标准输出
lrwx------    1 root     root            64 Jan  5 00:20 10 -> /dev/tty
lrwx------    1 root     root            64 Jan  5 00:20 2 -> /dev/console		# 2表示标准错误输出
# 标准输入/输出/错误输出都指向了控制台/dev/console，我正处在这个控制台上

# 我们的A，输出不到当前控制台上
# 现在，我们将标准输入重定向到/dev/tty1上去，
# 在重定向之前，我们需要记录一下当前设备的ip地址，在windows系统上cmd，然后telnet ip，即可进入bash控制台，
# 这样做是为了在重定向输入后，通过在telnet中，敲kill -9 145（这个145就是你之前看到的控制台的进程id），然后就可以在原来的console中敲命令了

exec 0</dev/tty1
# 现在你再按按钮，你就可以看见a
# 但是现在你无法在当前敲入任何命令，除非你在windows cmd的telnet控制台中敲入kill -9 145，就可以恢复

```

应用程序获取这些input key，可以查一下。

# 4 I2C总线

![img](legend/I2C通信示意.gif)

## 4.1 总线通信基本概念

1. 通信方向划分：单工，半双工，全双工
2. 同步通信
   - eg： I2C，SPI，USB3.0
   - 一般不支持远距离传输，通常是板级之间的距离小于50cm的
   - 也可通过走差分信号实现更远距离的通信
3. 异步通信
   - eg：UART、USB2.0、RJ45
   - 通信距离会稍远一些，通常是主机或设备之间的通信
   - 为了实现更远距离通信，一般走差分信号，eg：RS232、RS485、RS422、CAN等

## 4.2  I2C通信时序

通信时序：

- 主机发起启始信号（时钟高电平期间，数据产生一个下降沿）
- 发送地址，通信之前先通过从机地址选中要通信的从机设备
  - 地址一般从模块芯片数据手册中获得
  - 有的也提供外部IO来手动指定（当地址冲突的时候）
  - 通常是7位数据表示，也有10位的地址
- 主机接收对应从机的应答
  - ACK表示正确应答
  - NACK表示异常应答
- 开始传输数据，且每传输8bit数据应答一位
- 主机发起停止信号（时钟高电平期间，数据产生一个上升沿）
- 本次通信结束

![image-20240708110422545](legend/image-20240708110422545.png)

特点：

1. 字节序：大端字节序
2. SDA数据在SCL高电平周期保持稳定，在SCL低电平周期才可以切换下一个数据。（时钟高电平采集数据，时钟低电平准备数据）
3. i2c是电平触发数据传输，不同于spi的边沿触发
4. 位速率可达400kbit/s（快速模式），100kbit/s（标准），3.4Mbit/s（高速），用来传输普通的传感器数据是有余的，但用来传输视频数据是不够的。

## MMA8653

先从底板x6818bv2.pdf 找到mma8653，再从其中的引脚名MCU_SCL_2，定位到核心板x4418cv3_release20150713.pdf的GPIO口，再由GPIO口在芯片手册SEC_S5P6818X_Users_Manual_preliminary_Ver_0.00.pdf定位GPIO口对应的功能模式

![](./legend/mma8653_io口查找.png)

mma8653操作流程

- 芯片在上电后需要配置CTRL_REG1(0x2A)为模式ACTIVE，此时芯片就可以正常工作
- 检测chid_id是否正确(0Dh)
- 读取坐标信息(x、y、z/01h-06h)

# 其他

1. [sourceInsight](https://blog.csdn.net/wkd_007/article/details/131316924)

   - 【Ctrl + F】文件中查找操作
   - 【ctrl + /】 全局搜索关键字

2. [sourceinsight 自动补全](https://blog.csdn.net/byhyf83862547/article/details/137090831)

   - 选项卡options -> preference -> symbol lookups -> import symbols for all Projects -> add， 加入相关头文件文件夹到list表

3. [sourceinsight显示行号](https://blog.csdn.net/weixin_42727214/article/details/132128146)

4. 如果把声明写在代码的后面，会报警告： warning: ISO C90 forbids mixed declarations and code

   - 在C90标准下编译时出现了混淆声明和代码的情况。C90（即C99之前的标准）规定，函数内的变量声明必须放在代码的开始处，不允许在for循环、if语句或任何其他代码块中声明变量。

   ```c
   static ssize_t my_read(struct file *file, char __user *buf, size_t count, loff_t *ppos)
   {
       printk(KERN_INFO "Read from device\n");
       int ret;		// printk在int ret前面，所以会报警告
       // 读取keyboard的值
       ret = nxp_soc_gpio_get_in_value(PAD_GPIO_A+28);	// 通过返回值得到高低电平
       nxp_soc_gpio_set_out_value(PAD_GPIO_C+11,ret);
       if(ret == 0)
           return 0;
       else
           return 1;
   }
   ```

   

5. error: stray '\357' in program：在复制别处代码，粘到本地项目后，可能会报如此错，只要自己重新写一遍就好了。



