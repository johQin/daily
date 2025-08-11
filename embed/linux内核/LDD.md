# 4 调试技术

## 4.1 内核的调试支持

内核开发的配置选项，这些选项均出现在内核配置工具的“Kernel hacking”菜单中，但并非所有的体系结构都支持其中的某些选项。

```bash
# 编译时生效
CONFIG_DEBUG_KERNEL		# # 该配置项仅仅使其他调试选项可用，但它本身不会打开全部的调试功能
CONFIG_DEBUG_INFO	#生成可供 gdb 或 crash 工具分析的调试信息。
# 这些选项会增大内核镜像体积（如添加调试符号），但不会直接影响运行时行为。

# 运行时生效
内核故障检测（如 CONFIG_DEBUG_SLAB 用于内存分配调试）。
动态打印调试信息（如 CONFIG_DYNAMIC_DEBUG 控制 pr_debug() 输出）。
锁调试（如 CONFIG_DEBUG_SPINLOCK 检测自旋锁错误）。
# 这些选项会增加运行时开销（如性能下降或日志激增），但提供实时调试能力。


```

## 4.2 打印调试

主动打印日志

### 4.2.1 printk

```c
printk(KERN_DEBUG "Here I am %s:%i\n", __FILE__, __LINE__);
```

在头文件`<linux/kernel>`中定义了八种（0~7）可用的日志级别字符串，数值越小，严重程度越高。

当优先级小于`console_loglevel`才会被打印到控制台（`echo 8 > /proc/sys/kernel/printk`可以修改console_loglevel），这个控制台可以是字符模式的终端，也可以是串口打印机或者并口打印机。每次输出一行（必须以newline符结尾）。

如果系统同时运行了klogd和syslogd，则无论console_loglevel为何值，内核消息都将追加到`/var/log/messages`（这个文件地址由syslogd配置的优先）。

如果klogd没有运行，这些消息就不会传递到用户空间，这种情况只能查看`/proc/kmsg`文件。

### 4.2.2 重定向控制台消息

### 4.2.3 消息如何被记录以及记录到何处

printk将日志写到`__LOG_BUF_LEN__`字节的环形缓冲区。

klogd运行时，会读取内核消息并将他们分发到syslogd，它们的配置文件，会定义日志将被放在何处。

### 4.2.4 打印速度限制

```c
if(printk_ratelimit()) printk(KERN_NOTICE "The printer is still on fire \n");
// printk_ratelimit通过跟踪发送到控制台的消息数量而工作。
// 如果输出的速度超过某个阈值，printk_ratelimit将返回0，从而避免发送重复消息
```

### 4.2.5 打印设备编号

```c
int print_dev_t(char *buffer, dev_t dev); 
char *format_dev_t(char *buffer, dev_t dev);
```



## 4.3 查询调试

用户在需要的时候去查询系统信息，而不是像printk一样持续不断的产生数据。

### 4.3.1 使用/proc

`/proc`下面的每一个文件都绑定于一个内核函数，用户读取其中的文件时，该函数动态地生成文件内容。

#### 创建自己的/proc文件

```c
// 自定义内核函数
int (*read_proc)(char *page, char **start, off_t offset, int count, int *eof, void *data);
// 注册自定义的内核函数
struct proc_dir_entry *create_proc_read_entry(const char *name,mode_t mode, struct proc_dir_entry *base, read_proc_t *read_proc, void *data);
// 注销自定义内核函数
remove_proc_entry(const char *name, NULL /* parent dir */);

// eg:
// 用户在执行`cat /proc/my_fiel`，内核将调用my_read_proc函数
int my_read_proc(char *page, char **start, off_t off, int count, int *eof, void *data) {
    int len = sprintf(page, "Hello from kernel!\n");
    return len;
}
```

缺点：

1. 消息的内容大小有限：内核分配了一页内存(就是说, PAGE_SIZE 字节)，驱动可以写入数据来返回给用户空间
2. 用同样的名子注册两个入口. 内核信任驱动, 不会检查名子是否已经注册了

### 4.3.2 seq_file（推荐）

可支持大数据量

`<linux/seq_file.h>`

```c
// 首先需要一个
static struct seq_operations { 
 void *start(struct seq_file *sfile, loff_t *pos);
 void *next(struct seq_file *sfile, void *v, loff_t *pos); 
 void stop(struct seq_file *sfile, void *v);
 int show(struct seq_file *sfile, void *v);
};
```

seq_operations <----file_operations.open()----> file_operations 这样建立联系，用户通过系统调用与文件系统建立联系

```c
// 创建/proc文件
struct proc_dir_entry *create_proc_entry(const char *name,mode_t mode,struct proc_dir_entry *parent);

entry = create_proc_entry("scullseq", 0, NULL);
if (entry) 
 entry->proc_fops = &scull_proc_ops;		// scull_proc_ops就是file_operations
```

### 4.3.3 ioctl

ioctl接收一个命令号和另一个（可选参数），命令号用以标识将要执行的命令，而可选参数通常是一个指针（指向用户空间，通过copy_from_user和copy_to_user，和内核交换数据）

## 4.4 监视调试

strace 命令时一个有力工具, 显示所有的用户空间程序发出的系统调用. 它不仅显示调用, 还以符号形式显示调用的参数和返回值

## 4.5 系统故障

### 4.5.1 oops

通常，我们可以在看到oops之后卸载自己有缺陷的驱动程序，然后重试。但是如果我们看到任何说明系统整体出现问题的信息后，最好的办法就是立即重新引导系统。