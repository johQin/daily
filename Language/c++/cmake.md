# [CMAKE](https://cmake.org)

1. [add_library](https://www.jianshu.com/p/31db466bc4e5)：用指定文件给工程添加一个库（这个库一般在工程的子目录下），子目录有自己的CMakeLists，但不编译库

   - [包括](https://cmake.org/cmake/help/latest/command/add_library.html#command:add_library)：

   - ```cmake
     # 1. 普通库
     add_library(
     	<name>
     	[STATIC | SHARED | MODULE]
     	[EXCLUDE_FROM_ALL] 
     	[source1] [source2 ...]
     	)
     	
     SET(LIBHELLO_SRC hello.c java.c)
     ADD_LIBRARY(hello SHARED ${LIBHELLO_SRC})
     
     	
     # 2. 导入库 IMPORTED
     #  直接导入已经生成的库，cmake不会给这类library添加编译规则。
     add_library(
     	<name>
     	<SHARED|STATIC|MODULE|OBJECT|UNKNOWN>
     	IMPORTED
     	[GLOBAL]
     	)
     
     
     # 3. 对象库  OBJECT
     # 编译了源文件，但不链接
     add_library(<name> OBJECT <src>...)
     
     # 4. 别名库  ALIAS
     # 给定library添加一个别名，后续可使用<name>来替代<target>
     add_library(<name> ALIAS <target>)
     
     # 5. 接口库 INTERFACE
     add_library(<name> INTERFACE [IMPORTED [GLOBAL]])
     
     ```

   

2. [add_subdirectory](https://www.jianshu.com/p/07acea4e86a3)：为工程（父目录）添加一个子目录（source_dir），并按照子目录下的CMakeLists.txt构建（编译）该子目录，然后将构建产物输入到binary_dir。主要为工程（父目录）编译库（子目录，至于是动态库还是静态库，由子目录下的CMakeLists.txt 里的add_library 参数决定）

   ```bash
   add_subdirectory (source_dir [binary_dir] [EXCLUDE_FROM_ALL])
   # source_dir
   # 必选参数。该参数指定一个子目录，子目录下应该包含CMakeLists.txt文件和代码文件。
   # 子目录可以是相对路径也可以是绝对路径，如果是相对路径，则是相对当前目录的一个相对路径。
   
   # binary_dir
   # 可选参数。该参数指定一个目录，用于存放输出文件。
   # 可以是相对路径也可以是绝对路径，如果是相对路径，则是相对当前目录的一个相对路径。
   # 如果该参数没有指定，则默认的输出目录使用source_dir。
   
   # EXCLUDE_FROM_ALL
   # 可选参数。当指定了该参数，则子目录下的目标不会被父目录下的目标文件包含进去
   # 父目录的CMakeLists.txt不会构建子目录的目标文件，必须在子目录下显式去构建。
   # 例外情况：当父目录的目标依赖于子目录的目标，则子目录的目标仍然会被构建出来以满足依赖关系（例如使用了target_link_libraries）
   ```

3. 链接库：link_directories,  LINK_LIBRARIES,  target_link_libraries

   - ```cmake
     # 1.添加需要链接的库文件目录（https://blog.csdn.net/fengbingchun/article/details/128292359）
     # 相当于g++命令的-L选项
     link_directories(directory1 directory2 ...)
     # 添加路径使链接器可以在其中搜索库。提供给此命令的相对路径被解释为相对于当前源目录。
     # 该命令只适用于在它被调用后创建的target。
     # 还有
     target_link_directories()
     
     # 2.添加需要链接的库文件路径（绝对路径），将库链接到稍后添加的所有目标。
     link_libraries(absPath1 absPath2...)
     link_libraries("/opt/MATLAB/R2012a/bin/glnxa64/libeng.so")
     
     # 3.添加要连接的库文件名称
     # 指定链接 给定目标和其依赖项时 要使用的库或标志。
     target_link_libraries(<target>
                           <PRIVATE|PUBLIC|INTERFACE> <item>...
                          [<PRIVATE|PUBLIC|INTERFACE> <item>...]...)
                          
     # PUBLIC 在public后面的库会被Link到你的target中，并且里面的符号也会被导出，提供给第三方使用。
     # PRIVATE 在private后面的库仅被link到你的target中，并且终结掉，第三方不能感知你调了啥库
     # INTERFACE 在interface后面引入的库不会被链接到你的target中，只会导出符号。
     target_link_libraries(myProject eng mx)     
     #equals to below 
     #target_link_libraries(myProject -leng -lmx) `
     #target_link_libraries(myProject libeng.so libmx.so)`
     
     ```

   - **target_link_libraries 要在 add_executable '之后'，link_libraries 要在 add_executable '之前'**

4. 包含头：[include_directories](https://blog.csdn.net/sinat_31608641/article/details/121666564)，[target_include_directories](https://blog.csdn.net/sinat_31608641/article/details/121713191)

   ```cmake
   # 将指定目录添加到编译器的头文件搜索路径之下，指定的目录被解释成当前源码路径的相对路径。
   include_directories ([AFTER|BEFORE] [SYSTEM] dir1 [dir2 ...])
   # include_directories 的影响范围最大，可以为CMakelists.txt后的所有target添加头文件目录
   # 一般写在最外层CMakelists.txt中影响全局
   
   # 默认情况下，include_directories命令会将目录添加到查找列表最后，可以通过命令设置CMAKE_INCLUDE_DIRECTORIES_BEFORE变量为ON来改变它默认行为，将目录添加到列表前面。
   # 也可以在每次调用include_directories命令时使用AFTER或BEFORE选项来指定是添加到列表的前面或者后面。
   # 如果使用SYSTEM选项，会把指定目录当成系统的搜索目录。该命令作用范围只在当前的CMakeLists.txt。
   
   
   # 指定目标需要包含的头文件路径。
   target_include_directories(
   	<target> 
   	[SYSTEM] [AFTER|BEFORE]
     	<INTERFACE|PUBLIC|PRIVATE> [items1...]
     	[<INTERFACE|PUBLIC|PRIVATE> [items2...] ...]
   )
   # target_include_directories的影响范围可以自定义。如加关键子PRIVATE或这PUBLIC。
   ```

   

5. 



# 工具函数

1. aux_source_directory

   ```cmake
   # 搜集所有在指定路径下的源文件的文件名，将输出结果列表储存在指定的变量中。
   aux_source_directory(< dir > < variable >)
   ```

   

**目标** 由 add_library() 或 add_executable() 生成。

target相关：add_executable、add_library

target_link_libraries 要在 add_executable '之后'，link_libraries 要在 add_executable '之前'

编译原理相关：

# 其它

1. [**target_xxx 中的 PUBLIC，PRIVATE，INTERFACE**](https://zhuanlan.zhihu.com/p/82244559)
   - PRIVATE：当前的修饰对象item对target的上层（target的调用层）不可见，target的调用层不使用当前修饰对象的功能。target无需暴露并且隐藏该item
   - INTERFACE：target不使用当前修饰对象item的功能，可target的调用层会使用到当前修饰对象的功能，target需要暴露当前item的接口
   - PUBLIC = INTERFACE + PRIVATE
2. 