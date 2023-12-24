# log

1. [检查torch是否是gpu版本](https://www.cnblogs.com/tommickey/p/17691926.html)
   - 查看PyTorch版本：打开Python交互式环境，导入torch包，使用命令`torch.__version__`查看PyTorch版本，如果版本名称中包含“cuda”，则表示是GPU版本。例如，如果版本名称为“1.7.0+cu101”，则是支持CUDA 10.1的GPU版本。
   - 查看torch.cuda：在Python交互式环境中，导入torch包后，使用命令`torch.cuda.is_available()`检查CUDA是否可用。如果返回值为True，则表示是GPU版本。
   - 查看GPU设备列表：在Python交互式环境中，导入torch包后，使用命令`torch.cuda.device_count()`检查当前系统中可用的GPU设备数量，如果返回值大于0，则表示是GPU版本。可以使用`torch.cuda.get_device_name()`命令查看每个设备的名称。例如，如果返回值为1，并且使用`torch.cuda.get_device_name(0)`命令返回GPU设备的名称，则说明是GPU版本。
2. 