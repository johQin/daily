# [CMAKE](https://cmake.org)

# 1 基本打包

1. [add_executable](https://blog.csdn.net/HandsomeHong/article/details/122402395)

   ```cmake
   # 使用指定的源文件创建出一个可执行文件（target）
   
   # 1. 普通可执行文件
   add_executable(<name> [WIN32] [MACOSX_BUNDLE]
                  [EXCLUDE_FROM_ALL]
                  [source1] [source2 ...])
   # name：可执行文件名（target name）
   # [source]: 构建可执行目标文件所需要的源文件。也可以通过target_sources(在add_executable或add_library之后使用)继续为可执行目标文件添加源文件
   
   # 2. 导入可执行文件
   add_executable(<name> IMPORTED [GLOBAL])
   
   set(GIT_EXECUTABLE "/usr/local/bin/git")
   add_executable(Git::Git IMPORTED)
   set_property(TARGET Git::Git PROPERTY IMPORTED_LOCATION "${GIT_EXECUTABLE}") 
   
   # 3. 别名可执行文件
   add_executable(<name> ALIAS <target>)
   # 为目标文件取一个别名，以便后续继续使用。为目标创建别名之后，可以使用别名读取目标的属性，但不能修改目标属性。
   # The <target> may not be an ALIAS.
   ```

   

2. 声明或手动编译一个库（target）[add_library](https://www.jianshu.com/p/31db466bc4e5)：用指定文件给工程添加一个库（这个库一般在工程的子目录下），子目录有自己的CMakeLists，但不编译库（由上层主动编译），除非手动

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

   

3. 编译一个库：[add_subdirectory](https://www.jianshu.com/p/07acea4e86a3)：为工程（父目录）添加一个子目录（source_dir），并按照子目录下的CMakeLists.txt构建（编译）该子目录，然后将构建产物输入到binary_dir。主要为工程（父目录）编译库（子目录，至于是动态库还是静态库，由子目录下的CMakeLists.txt 里的add_library 参数决定）。

   - 在父目录的CMakeLists.txt执行到此函数时，子目录的CMakeLists.txt开始执行，然后再回到父目录的CMakeLists.txt执行此函数后面的编译选项。

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

4. [链接一个库：link_directories,  LINK_LIBRARIES,  target_link_libraries](https://blog.csdn.net/weixin_38346042/article/details/131069948)

   - ```cmake
     # 1.添加需要链接的库文件目录（https://blog.csdn.net/fengbingchun/article/details/128292359）
     # 相当于g++命令的-L选项，添加编译器可以查找库的文件夹路径，但不会将库链接到target上。
     link_directories(directory1 directory2 ...)
     # 添加路径使链接器可以在其中搜索库。提供给此命令的相对路径被解释为相对于当前源目录。
     # 该命令只适用于在它被调用后创建的target。
     # 还有
     target_link_directories()
     
     # 2.添加需要链接的库文件路径（绝对路径），将库链接到稍后添加的所有目标。
     link_libraries(absPath1 absPath2...)
     link_libraries("/opt/MATLAB/R2012a/bin/glnxa64/libeng.so")
     # 如果target调用了某个库，而没有取link，那么就会报undefined reference to `xxx'
     # 例如：使用mysql.h里的函数，而没有link mysqlclient 就会报undefined reference to `mysql_init'
     
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

5. 包含头：[include_directories](https://blog.csdn.net/sinat_31608641/article/details/121666564)，[target_include_directories](https://blog.csdn.net/sinat_31608641/article/details/121713191)

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

   

6. 



# 2 变量

## 2.1 set

```cmake
#设置普通变量
set(<variable> <value>... [PARENT_SCOPE]) # 不指定PARENT_SCOPE，变量是函数作用域和目录作用域（作用域解释请find），加了PARENT_SCOPE就在父作用域

#设置缓存条目
set(<variable> <value>... CACHE <type> <docstring> [FORCE]) 	# 整个编译生命周期都有效。缓存作用域Persistent Cache（cache变量）

#设置环境变量
set(ENV{<variable>} [<value>])
```

### [set变量作用域](https://blog.csdn.net/weixin_43708622/article/details/108315184)

- 变量是通过set或unset来设置或取消设置的。
- 函数作用域Function Scope（普通变量），类似于javascript变量作用域，有一个作用域链。除非PARENT_SCOPE来改变函数作用域
- 目录作用域Directory Scope（普通变量），
  - 变量拷贝：当前父目录下CMakeLists.txt的变量，在通过add_subdirectory函数后，子目录的CMakeLists.txt会将变量进行全部拷贝，子目录对变量的修改不会影响父目录CMakeLists.txt的变量的值。
  - 向下有效：具有继承关系的子目录才能获取到父目录的变量。
- 缓存作用域Persistent Cache（cache变量）
  - 缓存变量在整个cmake工程的编译生命周期内都有效，工程内的其他任意目录都可以访问缓存变量，注意cmake是从上到下来解析CMakeLists.txt文件的（所以add_subdirectory执行有个先后顺序，没有继承关系的子目录之间需要，在别人设置之后才能读到）。
  - 所有的 Cache 变量都会出现在 CMakeCache.txt 文件中。
  - 有一个与 Cache 变量同名的 Normal 变量出现时，后面使用这个变量的值都是以 Normal 为准，如果没有同名的 Normal 变量，CMake 才会自动使用 Cache 变量。

[CMake 变量、环境变量、持久缓存区别](https://blog.csdn.net/m0_57845572/article/details/118400027)

# 3 [属性](https://blog.csdn.net/jjjstephen/article/details/122464085)

**属性不像变量那样持有独立的值，它提供特定于它所附加的实体的信息。**

它们总是附加到特定的实体上，无论是目录、目标、源文件、测试用例、缓存变量，还是整个构建过程本身。

## set_property

```cmake
set_property(<GLOBAL                      |
              DIRECTORY [<dir>]           |
              TARGET    [<target1> ...]   |
              SOURCE    [<src1> ...]
                        [DIRECTORY <dirs> ...]
                        [TARGET_DIRECTORY <targets> ...] |
              INSTALL   [<file1> ...]     |
              TEST      [<test1> ...]     |
              CACHE     [<entry1> ...]    >
             [APPEND] [APPEND_STRING]
             PROPERTY <name> [<value1> ...])
 
 # 其基本格式为：
 set_property(<Scope> [APPEND] [APPEND_STRING] PROPERTY <name> [value...])
 # Scope：属性的宿主，所属对象，或者说属性的范围
 # [APPEND | APPEND_STRING] 可选，表示属性是可扩展的列表。
 # PROPERTY 是标识
 # name：属性名称
 # value：属性值，其值可选。
 
 add_executable(foo foo.cpp)
 set_target_properties(foo PROPERTIES
    CXX_STANDARD 14
    CXX_EXTENSIONS OFF
)
```

<table><tbody><tr><td> <p style="text-align:center;">Scope</p> </td><td> <p style="text-align:center;">Description</p> </td><td> <p style="text-align:center;">相似命令</p> </td></tr><tr><td> <p>GLOBAL</p> </td><td> <p>属性在全局范围内有效，属性名称需唯一</p> </td><td></td></tr><tr><td> <p>DIRECTORY</p> </td><td> <p>在指定目录内有效，可以是相对路径也可以是绝对路径</p> </td><td> <p>set_directory_properties</p> </td></tr><tr><td> <p>TARGET</p> </td><td> <p>设置指定 TARGET 的属性</p> </td><td> <p>set_target_properties/get_target_property</p> </td></tr><tr><td> <p>SOURCE</p> </td><td> <p>属性对应零个或多个源文件。默认情况下，源文件属性仅对添加在同一目录 (CMakeLists.txt) 中的目标可见。</p> </td><td> <p>set_source_files_properties</p> </td></tr><tr><td> <p>INSTALL</p> </td><td> <p>属性对应零个或多个已安装的文件路径。这些可供 CPack 使用以影响部署。</p> </td><td></td></tr><tr><td> <p>TEST</p> </td><td> <p>属性对应零个或多个现有测试。</p> </td><td> <p>set_tests_properties</p> </td></tr><tr><td> <p>CACHE</p> </td><td> <p>属性对应零个或多个缓存现有条目。</p> </td><td></td></tr></tbody></table>





## get_property

```cmake
# 将属性值存放在变量中
get_property(<variable>
             <GLOBAL             |
              DIRECTORY [<dir>]  |
              TARGET    <target> |
              SOURCE    <source>
                        [DIRECTORY <dir> | TARGET_DIRECTORY <target>] |
              INSTALL   <file>   |
              TEST      <test>   |
              CACHE     <entry>  |
              VARIABLE           >
             PROPERTY <name>
             [SET | DEFINED | BRIEF_DOCS | FULL_DOCS])
# variable：变量名
# GLOBAL/DIRECTORY ... /VARIABLE：表示属性对应的范围，与 set_property() 相同
# PROPERTY <name> : 属性名，同 set_property()

# [SET | DEFINED | BRIEF_DOCS | FULL_DOCS] : 可选参数

# SET : 将变量设置为布尔值，指示是否已设置属性；
# DEFINED : 将变量设置为布尔值，指示属性是否已被定义
# BRIEF_DOCS | FULL_DOCS : 如果给定了 Brief_DOCS 或 FULL_DOCS，则将变量设置为包含所请求属性的文档的字符串。
```



```cmake
# 例子
#CMakeLists.txt
cmake_minimum_required(VERSION 3.10.2)
project(test)

set(GIT_EXECUTABLE "/usr/local/bin/git")
add_executable(Git::Git IMPORTED)
set_property(TARGET Git::Git PROPERTY IMPORTED_LOCATION "${GIT_EXECUTABLE}") 
# https://cmake.org/cmake/help/latest/prop_tgt/IMPORTED_LOCATION.html
get_target_property(git_location Git::Git IMPORTED_LOCATION)
get_target_property(git_imported Git::Git IMPORTED)
message(">>> git location: ${git_location}, ${git_imported}")
```



# 工具函数

1. [获取目录下所有文件：`aux_source_directory，file`](https://blog.csdn.net/a2886015/article/details/126830638)

   ```cmake
   # 搜集所有在指定路径下的源文件的文件名，将输出结果列表储存在指定的变量中。
   aux_source_directory(< dir > < variable >)
   # aux_source_directory 命令生成的是源文件的相对路径，传递到上一层之后无法正常使用。
   
   # file(GLOB) 命令来查找源文件(可以使用通配符)，它会生成文件的绝对路径。注意 file() 可以通过GLOB_RECURSE递归查找的，也就是说子目录下的源代码也会被找到。
   file(GLOB <variable> DIR)
   
   # 文件结构
   |--mat
   |	|---add.cpp
   |	|---sub.cpp
   |	|---he
   |		|---ha
   |			|--multi.cpp
   |--CMakeLists.txt
   # 在当前文件夹下的math文件夹里查找源码
   aux_source_directory(mat SOUR1)				#	SOUR1: mat/add.cpp;mat/sub.cpp
   aux_source_directory(./mat SOUR2)			# 	SOUR2：./mat/add.cpp;./mat/sub.cpp	
   file(GLOB SOUR3 "mat/*.cpp")
   #	SOUR3:/home/buntu/code/ctest/mat/add.cpp;/home/buntu/code/ctest/mat/sub.cpp
   file(GLOB SOUR4 "./mat/*.cpp")
   #	SOUR4 /home/buntu/code/ctest/./mat/add.cpp;/home/buntu/code/ctest/./mat/sub.cpp
   file(GLOB_RECURSE SOUR5 "mat/*.cpp")					#SOUR5：/home/buntu/code/ctest/mat/add.cpp;/home/buntu/code/ctest/mat/he/ha/multi.cpp;/home/buntu/code/ctest/mat/sub.cpp
   file(GLOB_RECURSE SOUR6 "mat/he/*.cpp")		#	SOUR6：/home/buntu/code/ctest/mat/he/ha/multi.cpp
   
   # 如果某个文件下只需要部分源码，可以通过变量来简化书写
   set( SOUR7 mat/add.cpp mat/sub.cpp)			# SOUR7：mat/add.cpp;mat/sub.cpp
   
   add_executable(${PROJECT_NAME}  ${SOURCES})
   ```

2. 打印message

   ```cmake
   message([<mode>] "message text" ...)
   # mode 的值包括 FATAL_ERROR、STATUS、WARNING、AUTHOR_WARNING、VERBOSE等。
   # FATAL_ERROR：产生 CMake Error，会停止编译系统的构建过程；
   # STATUS：project用户可能感兴趣的主要信息。理想情况下，这些信息应该简明扼要，不超过一行，但仍然信息丰富。
   # WARNING：显示警告信息。
   # VERBOSE：针对project用户的详细信息消息。这些消息应提供在大多数情况下不感兴趣的额外详细信息。
   # "message text"为显示在终端的内容。
   message(">>> git location: ${git_location}, ${git_imported}")
   
   # 如果想在clion看到cmake编译信息，可以在底部Messages栏查看
   ```

   ![](./legend/cmake_message.png)

3. [configure_file](https://www.jianshu.com/p/2946eeec2489)

   - 将一个文件(由`input`参数指定)拷贝到指定位置(由`output`参数指定)，并根据`options`修改其内容。

   - **configure_file命令一般用于自定义编译选项或者自定义宏的场景。**

   - `configure_file(<input> <output> [options])`

   - configure_file命令会根据`options`指定的规则，自动对`input`文件中`cmakedefine`关键字及其内容进行转换。

   - 具体来说，会做如下几个替换：

     - 将input文件中的`@var@`或者`${var}`替换成cmake中指定的值。
     -  将input文件中的`#cmakedefine var`关键字替换成`#define var`或者`#undef var`，取决于cmake是否定义了var。

   - c++获取项目路径的两种方式中有应用

4. [file](https://www.jianshu.com/p/ed151fdcf473)

   - **使用cmake 文件操作时不可避免需要操作相关文件，比如读取文件内容，创建新文件，返回文件地址等等**

# 最佳实践

## 1 [c++获取项目路径的两种方式](https://blog.csdn.net/qq_36614557/article/details/120713808)

在某些特定的条件运行时不能使用局部地址，例如ci流程等，这就要求读取文件时必需使用全局地址，但是在项目路径不定的情况下很难知道某个文件的全局地址，目前存在两种获取项目路径的方式，其中一种更适用于ci流程。

###  Cmake传参：适用于简单场景

```cmake
# CMakeLists.txt，/home/buntu/gitRepository/daily/Language/c++/program/RainWarn
set(PROJECT_ROOT_PATH ${PROJECT_SOURCE_DIR})
configure_file(includes/SelfENV.h.in ../includes/SelfENV.h)
message( "view vars: ${CMAKE_CURRENT_SOURCE_DIR}, ${CMAKE_CURRENT_BINARY_DIR}")
# view vars:
# /home/buntu/gitRepository/daily/Language/c++/program/RainWarn,
# /home/buntu/gitRepository/daily/Language/c++/program/RainWarn/cmake-build-debug

# 强调一点：
# configure_file(<input> <output> [options])
# input: 输入文件的路径，它是一个相对路径，以CMAKE_CURRENT_SOURCE_DIR为路径前缀。此外，它必须是一个文件，不能是目录
# output - 输出文件或目录，它也是一个相对路径，以CMAKE_CURRENT_BINARY_DIR为前缀。


# SelfENV.h.in，/home/buntu/gitRepository/daily/Language/c++/program/RainWarn/includes
#define PROJECT_ROOT_PATH "@PROJECT_ROOT_PATH@"


# cmake后，会在includes下生成一个SelfENV.h
#define PROJECT_ROOT_PATH "/home/buntu/gitRepository/daily/Language/c++/program/RainWarn"
```

### 环境变量中获取

C++中自带函数getenv，可以读取指定的环境变量，返回char *。



# 其它

1. [**target_xxx 中的 PUBLIC，PRIVATE，INTERFACE**](https://zhuanlan.zhihu.com/p/82244559)
   - PRIVATE：当前的修饰对象item对target的上层（target的调用层）不可见，target的调用层不使用当前修饰对象的功能。target无需暴露并且隐藏该item
   - INTERFACE：target不使用当前修饰对象item的功能，可target的调用层会使用到当前修饰对象的功能，target需要暴露当前item的接口
   - PUBLIC = INTERFACE + PRIVATE
2. 

# 常用包

1. [读取xml的工具](http://www.firstobject.com/dn_markupmethods.htm)

   - [每个函数的汉语意思](https://dandelioncloud.cn/article/details/1575145342054395905)
   - [CMarkup简介](https://blog.51cto.com/u_2339637/1345249)

   ```c++
   map<string, string> info;
   MCD_STR strName, strAttrib;
   int n = 0;
   while ( xml.GetNthAttrib(n++, strName, strAttrib) )
   {
   
       info.insert(make_pair(strName, strAttrib));
       // do something with strName, strAttrib
   }
   ```

   

2. 

# log

1. [合并静态库](https://zhuanlan.zhihu.com/p/389448385)
1. 设置可执行目标文件的输出目录：`SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY output) `

# 1 从可执行文件到库

## 1.1 生成器

CMake是一个构建系统生成器，可以使用单个CMakeLists.txt为不同平台上的不同工具集配置项目