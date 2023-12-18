# [CMAKE](https://cmake.org)

cmake内置命令是不区分大小写的， 因此`add_subdirectory`与`ADD_SUBDIRECTORY`作用一致。

cmake所有变量是区分大小写的

# 1 基本打包

## 1.1 [add_executable](https://blog.csdn.net/HandsomeHong/article/details/122402395)：指定可执行文件

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



## 1.2 [add_library](https://www.jianshu.com/p/31db466bc4e5)：声明库

声明或手动编译一个库（target）add_library：用指定文件给工程添加一个库（这个库一般在工程的子目录下），子目录有自己的CMakeLists，但不编译库（由上层主动编译），除非手动

- [包括](https://cmake.org/cmake/help/latest/command/add_library.html#command:add_library)：

- ```cmake
  # 1. 普通库
  add_library(
  	<name>
  	[STATIC | SHARED | MODULE]
  	[EXCLUDE_FROM_ALL] 
  	[source1] [source2 ...]				#soruce也可以是头文件
  	)
  	
  # STATIC 静态库
  # SHARED 动态库
  SET(LIBHELLO_SRC hello.c java.c)
  ADD_LIBRARY(hello SHARED ${LIBHELLO_SRC})
  
  # 给库暴露头文件供上层包含
  target_include_directories(ReadXml PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  
  	
  # 2. 导入库 IMPORTED
  #  直接导入已经生成(现有)的库，cmake不会给这类library添加编译规则。
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

- 实际

  ```cmake
  # add_library 没有指定库类型时，会按照BUILD_SHARED_LIBS变量值，来决定是动态库还是静态库
  # 如果未指定BUILD_SHARED_LIBS 变量，则默认为 STATIC。
  if (NOT DEFINED BUILD_SHARED_LIBS)
      set(BUILD_SHARED_LIBS ON)
  endif()
  ```

  

## 1.3 [add_subdirectory](https://www.jianshu.com/p/07acea4e86a3)：编译库

编译一个库：add_subdirectory：为工程（父目录）添加一个子目录（source_dir），并按照子目录下的CMakeLists.txt构建（编译）该子目录，然后将构建产物输入到binary_dir。主要为工程（父目录）编译库（子目录，至于是动态库还是静态库，由子目录下的CMakeLists.txt 里的add_library 参数决定）。

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

[实例：CMake - 子目录构建链接动静态库和动态库](https://blog.csdn.net/Jay_Xio/article/details/122019770)



## 1.4 [link_XXX 链接库](https://blog.csdn.net/weixin_38346042/article/details/131069948)

链接一个库：link_directories,  LINK_LIBRARIES,  target_link_libraries

- link_libraries已被废弃（失效）了，建议使用target_link_libraries替代，存疑

- 如果不指定库后缀，默认优先链接动态库。

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
  
  # 3.添加要连接的库文件名称，默认优先链接动态库
  # 指定链接 给定目标和其依赖项时 要使用的库或标志。
  target_link_libraries(<target>
                        <PRIVATE|PUBLIC|INTERFACE> <item>...
                       [<PRIVATE|PUBLIC|INTERFACE> <item>...]...)
                       
  # target不能是ALIAS target。
  # PUBLIC 在public后面的库会被Link到你的target中，并且里面的符号也会被导出，提供给第三方使用。
  # PRIVATE 在private后面的库仅被link到你的target中，并且终结掉，第三方不能感知你调了啥库
  # INTERFACE 在interface后面引入的库不会被链接到你的target中，只会导出符号。
  target_link_libraries(myProject eng mx)     
  #equals to below 
  #target_link_libraries(myProject -leng -lmx) `
  #target_link_libraries(myProject libeng.so libmx.so)`
  
  # 以下写法都可以： 
  target_link_libraries(myProject comm)       # 连接libhello.so库，默认优先链接动态库
  target_link_libraries(myProject libcomm.a)  # 显示指定链接静态库
  target_link_libraries(myProject libcomm.so) # 显示指定链接动态库
  ```
  
- **target_link_libraries 要在 add_executable '之后'，link_libraries 要在 add_executable '之前'**

## 1.5 include 包含头

包含头：[include_directories](https://blog.csdn.net/sinat_31608641/article/details/121666564)，[target_include_directories](https://blog.csdn.net/sinat_31608641/article/details/121713191)

- **include_directories 的影响范围最大，可以为CMakelists.txt后的所有target添加头文件目录**
- **一般写在最外层CMakelists.txt中影响全局（向下传递，父目录包含，那么其子目录也自动包含）**

```cmake
# 将指定目录添加到编译器的头文件搜索路径之下，指定的目录被解释成当前源码路径的相对路径。
include_directories ([AFTER|BEFORE] [SYSTEM] dir1 [dir2 ...])

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



## 1.6 [.cmake文件](https://blog.csdn.net/qq_38410730/article/details/102677143)

`CmakeLists.txt`才是`cmake`的正统文件，而`.cmake`文件是一个模块文件，可以被`include`到`CMakeLists.txt`中。

.cmake文件里包含了一些cmake命令和一些宏/函数，当CMakeLists.txt包含该.cmake文件时，当编译运行时，该.cmake里的一些命令就会在该包含处得到执行，并且在包含以后的地方能够调用该.cmake里的一些宏和函数。

### include指令

include指令**一般用于语句的复用**，也就是说，如果有一些语句需要在很多CMakeLists.txt文件中使用，为避免重复编写，可以将其写在.cmake文件中，然后在需要的CMakeLists.txt文件中进行include操作就行了。

```cmake
include(<file|module> [OPTIONAL] [RESULT_VARIABLE <var>]
                      [NO_POLICY_SCOPE])
include(file|module)

```

**注意**：为了使`CMakeLists.txt`能够找到该文件，需要指定文件完整路径(绝对路径或相对路径)，当然如果指定了`CMAKE_MODULE_PATH`，就可以直接`include`该目录下的`.cmake`文件了。

**.cmake文件里包含了一些cmake命令和一些宏/函数，当CMakeLists.txt包含该.cmake文件时，当编译运行时，该.cmake里的一些命令就会在该包含处得到执行，并且在包含以后的地方能够调用该.cmake里的一些宏和函数**。

### macro宏和function函数

```cmake
macro(<name> [arg1 [arg2 [arg3 ...]]])
  COMMAND1(ARGS ...)            # 命令语句
  COMMAND2(ARGS ...)
  ...
endmacro()

function(<name> [arg1 [arg2 [arg3 ...]]])
  COMMAND1(ARGS ...)            # 命令语句
  COMMAND2(ARGS ...)
  ...
function()

```

定义一个名称为`name`的宏（函数），`arg1...`是传入的参数。我们除了**可以用`${arg1}`来引用变量**以外，系统为我们提供了一些特殊的变量：

| 变量  | 说明                                                 |
| ----- | ---------------------------------------------------- |
| argv# | #是一个下标，0指向第一个参数，累加                   |
| argv  | 所有的定义时要求传入的参数                           |
| argn  | 定义时要求传入的参数以外的参数                       |
| argc  | 传入的实际参数的个数，也就是调用函数是传入的参数个数 |

其实和`C/C++`里面宏和函数之间的区别差不多

- **当宏和函数调用的时候，如果传递的是经`set`设置的变量，必须通过`${}`取出内容**；
- **在宏的定义过程中，对变量进行的操作必须通过`${}`取出内容，而函数就没有这个必要**。



# 2 变量

## 2.1 set

```cmake
set(hello "good")
include(CMakePrintHelpers)		# 这是一个打印帮助工具
cmake_print_variables(hello)
set(hello)
cmake_print_variables(hello)
set(world "morning")
cmake_print_variables(world)
unset(world)
cmake_print_variables(world)
# 结果
-- hello="good"
-- hello=""
-- world="morning"
-- world=""

# 如果设置多个值，将会连接起来(用";"分隔)作为一个整体赋值给变量。
set(VAR0 1 2 3 4)
set(VAR1 "hello" "good" "evening")
message(STATUS "var0 is: ${VAR0}, var1 is: ${VAR1}")
# 结果
-- var0 is: 1;2;3;4, var1 is: hello;good;evening
```

### 2.1.1 普通变量

```cmake
#设置普通变量
set(<variable> <value>... [PARENT_SCOPE]) # 不指定PARENT_SCOPE，变量是函数作用域和目录作用域，加了PARENT_SCOPE就在父作用域
```

### 2.1.2 缓存变量

```cmake
#设置缓存条目
set(<variable> <value>... CACHE <type> <docstring> [FORCE])
# 默认不会覆盖已存在的缓存变量，通过可选参数FORCE可以强制重写。
# 整个编译生命周期都有效。缓存作用域Persistent Cache（cache变量）
# type值并没有强制作用，只作为给读者的提示。
# type必须是以下之一：
#	BOOL：值为ON/OFF
#	FILEPATH：文件路径
#	PATH：文件所在目录的路径
#	STRING：一行文本
#	INTERNAL：文本，主要在运行过程中存储变量，不对外展示。

set(VAR2 "hello" CACHE BOOL "it is my set BOOL test" FORCE)
set(VAR3 "good" CACHE FILEPATH "it is my set FILEPATH test" FORCE)
set(VAR4 "study" CACHE PATH "it is my set PATH test" FORCE)
set(VAR5 "beautiful" CACHE STRING "it is my set STRING test" FORCE)
set(VAR6 "perfect" CACHE INTERNAL "it is my set INTERNAL test" FORCE)

foreach(var VAR2 VAR3 VAR4 VAR5 VAR6)
    message(STATUS "var is ${${var}}")
endforeach()
# 结果
-- var is hello
-- var is good
-- var is study
-- var is beautiful
-- var is perfect

# 在CMakeCache.txt中会有
//it is my set BOOL test
VAR2:BOOL=hello

//it is my set FILEPATH test
VAR3:FILEPATH=good

//it is my set PATH test
VAR4:PATH=study

//it is my set STRING test
VAR5:STRING=beautiful

//it is my set INTERNAL test
VAR6:INTERNAL=perfect
```

### 2.1.3 环境变量

```cmake
#设置环境变量
set(ENV{<variable>} [<value>])		# 环境变量仅用于cmake编译过程，不能用于目标程序，也不会修改操作系统的操作环境变量
$ENV{<variable>}		# 读取环境变量

# 在cmake中查看linux操作系统的环境变量
message(WARNING "PATH=$ENV{PATH}")
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

## 2.2 [property](https://blog.csdn.net/jjjstephen/article/details/122464085)

**属性不像变量那样持有独立的值，它提供特定于它所附加的实体的信息。**

它们总是附加到特定的实体上，无论是目录、目标、源文件、测试用例、缓存变量，还是整个构建过程本身。

### set_property

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





### get_property

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

## 2.3  [list](https://www.jianshu.com/p/89fb01752d6f)

`list`命令即对列表的一系列操作，cmake中的`列表`变量是用分号`;`分隔的一组字符串

### 创建列表

```cmake
set (list_test a b c d) # 创建列表变量"a;b;c;d"
list (LENGTH list_test length)
message (">>> LENGTH: ${length}")	# LENGTH: 4
```

### 查询

```cmake
# 得到索引元素
# list (GET <list> <element index> [<element index> ...] <output variable>)
# output variable为新创建的变量，也是一个列表
set (list_test a b c d)
list (GET list_test 0 1 -1 -2 list_new)			# 索引：正数——正序索引，负数——倒序索引
message (">>> GET: ${list_new}")			# GET:a;b;d;c

# 子列表
# list (SUBLIST <list> <begin> <length> <output variable>)
# <length>
#		为-1或列表的长度小于<begin>+<length>，那么将列表中从<begin>索引开始的剩余元素返回。
# 		为0，返回空列表。

# 查找
# list (FIND <list> <value> <output variable>)
# 用于查找列表是否存在指定的元素，找到返回元素的索引，找不到返回-1
set (list_test a b c d)
list (FIND list_test d list_index)			
message (">>> FIND 'd': ${list_index}")		# FIND 'd': 3
```

### 修改

```cmake
# 增
# 会改变原来列表的值。
list (APPEND <list> [<element> ...])
list (INSERT <list> <element_index> <element> [<element> ...])	# 如果元素的位置超出列表的范围，会报错。
list (POP_BACK <list> [<out-var>...])		# 将原列表的最后一个元素移除
list (POP_FRONT <list> [<out-var>...])		# 将原列表的第一个元素移除。

# 过滤
list (FILTER <list> <INCLUDE|EXCLUDE> REGEX <regular_expression>)
# 根据模式的匹配结果，将元素添加（INCLUDE选项）到列表或者从列表中排除（EXCLUDE选项）。
# 此命令会改变原来列表的值。模式REGEX表明会对列表进行正则表达式匹配。
set (list_test a b c d 1 2 3 4) 
list (FILTER list_test INCLUDE REGEX [a-z])
message (">>> FILTER: ${list_test}")		# FILTER: a;b;c;d

# 调序
list (REVERSE <list>)
list (SORT <list> [COMPARE <compare>] [CASE <case>] [ORDER <order>])
```

## 2.4 option

CMake中的option用于控制cmake编译流程。在项目中，难免的需要添加一些选项以供下游选择。

**option命令定义的变量不影响c或c++源码中#ifdef或者#ifndef逻辑判断**

**对于同一选项，子项目值遵循主项目的定义。**

```cmake
option(<variable> "<help_text>" [value])


CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
 
#项目信息
PROJECT(main)
 
IF(TEST)
	MESSAGE("TEST is defined,vlaue:${TEST}")
ELSE()
	MESSAGE("TEST is not defined")
ENDIF()
 
#command option
option(TEST "test affect to code" ON) 
 
#可执行文件
ADD_EXECUTABLE(main main.cpp)
 
IF(TEST)
	MESSAGE("TEST is defined,vlaue:${TEST}")
ELSE()
	MESSAGE("TEST is not defined")
ENDIF()
```

### add_definitions

`add_definitions`的功能和`C/C++`中的`#define`是一样的，可以和option配合使用，也可单独使用。

```c++
#include <iostream>
int main()
{
#ifdef TEST_IT_CMAKE
	std::cout<<"in ifdef"<<std::endl;
#endif
	std::cout<<"not in ifdef"<<std::endl;
}
```



```cmake
cmake_minimum_required(VERSION 3.10)
project(optiontest)

add_executable(optiontest main.cpp)
option(TEST_IT_CMAKE "test" ON)
message(${TEST_IT_CMAKE})
if(TEST_IT_CMAKE)
	message("itis" ${TEST_IT_CMAKE})
	add_definitions(-DTEST_IT_CMAKE)		# option名前需要加-D
endif()
```

## 2.5 [条件结构](https://zhuanlan.zhihu.com/p/653282782)

```cmake
if(<condition>)
  <commands>
elseif(<condition>) # optional block, can be repeated
  <commands>
else()              # optional block
  <commands>
endif()
```

在if后面的变量，不需要使用`${Var}`的形式获取Var的值，而是直接使用Var。

`if(P)`的语法看起来非常奇怪: 尝试对一个变量名称**自动求值**。

如果希望处理一个可能是**变量名的字符串**，建议使用双引号`if("${P}")`，这会**抑制if的自动求值**。

### 2.5.1 基本变量

P可以是最基本的常量，字符串或者变量名。

1. P是有意义的常量：`if(<constant>)`
   - true：
     - 1, ON, YES, TRUE, Y
     - 非零的数，甚至浮点数
   - false：
     - 空字符串
     - 0, OFF, NO, FALSE, N, IGNORE, NOTFOUND, *-NOTFOUND
   - 这里的bool量，不区分大小写，true，True，TRUE都是正。
   - 其它情形会被视作一个变量或一个字符串进行处理
2. P是一个变量的名称(而非变量的值)：`if(<variable>)`
   - true：
     - 变量已定义，并且变量的值不是上述False常量的情形
   - false：
     - 变量已定义，但是变量的值是上述False常量的情形
     - 变量未定义
     - 上述规则对宏结构不使用，对环境变量也不使用(环境变量的名称总是得到False)
3. P是字符串：`if(<string>)`
   - true：可以被解析为True常量的字符串
   - false：通常情形下，其它的字符串

### 2.5.2 逻辑运算

P可以是一些简单的逻辑判断

```cmake
# 取反运算
if(NOT <condition>)

# 与运算
if(<cond1> AND <cond2>)

# 或运算
if(<cond1> OR <cond2>)

if((condition1) AND (condition2 OR (condition3)))
```

### 2.5.3 存在性判断

- `if(COMMAND command-name)`: 判断这个command-name是否属于命令、可调用的宏或者函数的名称，则返回True
- `if(TARGET target-name)`: 判断这个target是否已经被`add_executable(), add_library(), add_custom_target()`这类命令创建，即使target不在当前目录下
- `if(DEFINED <name>|CACHE{<name>}|ENV{<name>})`: 判断这个变量是否已定义
- `if(<variable|string> IN_LIST <variable>)`: 判断这个变量或字符串是否在列表中，见下文的列表操作

### 2.5.4 大小比较

```cmake
# 数字比较
# 小于
if(<variable|string> LESS <variable|string>)
# 大于
if(<variable|string> GREATER <variable|string>)
# 等于
if(<variable|string> EQUAL <variable|string>)
# 小于或等于
if(<variable|string> LESS_EQUAL <variable|string>)
# 大于或等于
if(<variable|string> GREATER_EQUAL <variable|string>)

# 字符串比较
if(<variable|string> STRLESS <variable|string>)
if(<variable|string> STRGREATER <variable|string>)
if(<variable|string> STREQUAL <variable|string>)
if(<variable|string> STRLESS_EQUAL <variable|string>)
if(<variable|string> STRGREATER_EQUAL <variable|string>)

# 版本号比较
if(<variable|string> VERSION_LESS <variable|string>)
if(<variable|string> VERSION_GREATER <variable|string>)
if(<variable|string> VERSION_EQUAL <variable|string>)
if(<variable|string> VERSION_LESS_EQUAL <variable|string>)
if(<variable|string> VERSION_GREATER_EQUAL <variable|string>)
```

### 2.5.5 路径与文件判断

细节比较多，用时再查文档

```cmake
# 完整路径是否存在，这里~开头的还不行
if(EXISTS path-to-file-or-directory)

# 两个完整路径下的文件比较时间戳
if(file1 IS_NEWER_THAN file2)

# 完整路径是否是一个目录
if(IS_DIRECTORY path-to-directory)

# 完整路径是不是绝对路径
if(IS_ABSOLUTE path)
# 对于windows，要求路径以盘符开始
# 对于linux，要求路径以~开始
# 空路径视作false
```

## 2.6 [循环结构](https://blog.csdn.net/maizousidemao/article/details/132654835)

### 2.6.1 foreach

```cmake
foreach(<loop_var> <item1> <item2> <item3>...)
  <commands>
endforeach()
# eg:
set(item1 a)
set(item2 b)
set(item3 c)
set(item4 d)
foreach(var ${item1} ${item2} ${item3} ${item4})
    message("var = ${var}")
endforeach()

foreach(<loop_var> RANGE <stop>)
# eg:
foreach(var RANGE 5)
    message("var = ${var}")
endforeach()

foreach(<loop_var> RANGE <start> <stop> [<step>])
#eg:
foreach(var RANGE 2 10 2)
    message("var = ${var}")
endforeach()

foreach(<loop_var> IN [LISTS [<lists>]] [ITEMS [<items>]]) 		# 用LISTS指定列表后不需要用 ${}对列表进行取值。
# eg:
set(myList 1 2 3 4)
foreach(var IN LISTS myList)
    message("var = ${var}")
endforeach()

foreach(<loop_var>... IN ZIP_LISTS <lists>)
eg:
set(myList0 a b c d)
set(myList1 1 2 3 4)
foreach(var0 var1 IN ZIP_LISTS myList0 myList1)
    message("var0 = ${var0}, var1 = ${var1}")
endforeach()
```

### 2.6.2 while

```cmake
while(<condition>)
	<commands>
endwhile()

list(LENGTH myList listLen)
while(listLen GREATER 0)
    message("myList = ${myList}")
    list(POP_FRONT myList)
    list(LENGTH myList listLen)
endwhile()
----------------
myList = 1;2;3;4
myList = 2;3;4
myList = 3;4
myList = 4

```

也可以通过 `break()` 跳出循环，通过 `continue()` 结束本次循环并继续下次循环。



## cmake内置变量

```cmake
CMAKE_SOURCE_DIR			# 源码树的最顶层目录(也就是项目CMakeLists.txt文件所在的地方)
CMAKE_CURRENT_SOURCE_DIR	# CMake正在处理的CMakeLists.txt文件所在的目录。
							# 每次在add_subdirectory()调用的结果中处理新文件时，它都会更新，并在完成对该目录的处理后再次恢复。
CMAKE_CURRENT_LIST_DIR		# 自2.8.3开始，代表当前正在处理的列表文件的完整目录，和CMAKE_CURRENT_SOURCE_DIR几乎一样
							# 只是在CMakeLists.txt里有include(src/CMakeLists.txt)代码时，
							# CMAKE_CURRENT_SOURCE_DIR指向外部的CMakeLists，而CMAKE_CURRENT_LIST_DIR将指向src
							# https://blog.csdn.net/jacke121/article/details/106550720
							
CMAKE_BINARY_DIR			# 构建树的最顶层目录。
CMAKE_CURRENT_BINARY_DIR	# 当前CMake正在处理的CMakeLists.txt文件对应的构建目录。
							# 每次调用add_subdirectory()时它都会改变，并在add_subdirectory()返回时再次恢复。

EXECUTABLE_OUTPUT_PATH		# 指定最终的可执行文件生成的位置
LIBRARY_OUTPUT_PATH			# 指定库文件的输出目录
```



# 3 [依赖](https://blog.csdn.net/zhizhengguan/article/details/118396145)

## 3.1 [add_dependencies](https://blog.csdn.net/BeanGuohui/article/details/120217097) 指定依赖

给目标指定依赖项，依赖项可被先行编译。如此就不会报undefined reference

```cmake
add_dependencies(<target> [<target-dependency>]...)
```

在项目中通常会遇见这样的情况：（例如一个项目中有：main，libhello.a， libworld.a），当项目过小的时候，编译顺序是*.a，然后是main，但是当一个项目的文件过于庞大，就会导致编译的顺序不会按照主CMAKE的add_subdirectory引入的先后顺序，为了解决这一问题，就需要使用add_dependencies进行依赖指定。

```cmake
├── CMakeLists.txt// 下面用主CMAKE表示
├── hello
│   ├── CMakeLists.txt		// 下面用HELLOCMAKE表示
│   ├── hello.c
│   └── hello.h
├── main
│   ├── CMakeLists.txt		// 下面用MAINCMAKE表示
│   └── main.c
└── world
    ├── CMakeLists.txt		// 下面用WORLDCMAKE表示
    ├── world.c
    └── world.h
```

```cmake
# hellocmake
cmake_minimum_required(VERSION 3.5.1)
set(CMAKE_C_STANDARD 99)
add_library(hello STATIC world.c hello.h)

# worldcmake
cmake_minimum_required(VERSION 3.5.1)
set(CMAKE_C_STANDARD 99)
add_library(world STATIC world.c world.h)

# maincmake 
cmake_minimum_required(VERSION 3.5.1)
project(CmakeDemo C)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY /home/lib)
set(CMAKE_C_STANDARD 99)

add_executable(CmakeDemo main.c)
link_directories(/home/lib)			# 依赖hello.a 和 world.a
target_link_libraries(
        CmakeDemo
        hello
        world
)

# 主cmake 最外层CMakeLists.txt
cmake_minimum_required(VERSION 3.5)

add_subdirectory(main)	# main是依赖后二者的，而这里add_subdirectory却写在后二者的前面
						# 所以这里应该让hello和world先编译，故有了add_dependencies
add_subdirectory(hello)
add_subdirectory(world)

add_dependencies(CmakeDemo hello world)
# add_dependencies中所填写的名字应该是其他CMAKE生成目标的名字。
# 该示例中如果写成add_dependencies（CmakeDemo libhello.a libworld.a）则会报错。

# 这样写的好处在于，当一个项目构建的时候，由于依赖关系的存在，所以被依赖的项目总是最先构建，这样就不会出现找不到库而报错。
```



[参考1](https://blog.csdn.net/KingOfMyHeart/article/details/112983922)



## 3.2 查找文件find_file

- **该命令用于查找指定文件的完整路径**。
- 创建一个名为< VAR >的缓存条目(如果指定了NO_CACHE，则是一个普通变量)来存储此命令的结果。
  - 如果找到文件的完整路径，则结果存储在变量中，**并且搜索不会重复，除非该变量被清除。**
  - 如果没有找到，结果将是< VAR >-NOTFOUND。

```cmake
find_file (<VAR> name1 [path1 path2 ...])
find_file (
          <VAR>
          name | NAMES name1 [name2 ...]
          [HINTS [path | ENV var]... ]
          [PATHS [path | ENV var]... ]
          [PATH_SUFFIXES suffix1 [suffix2 ...]]
          [DOC "cache documentation string"]
          [NO_CACHE]
          [REQUIRED]
          [NO_DEFAULT_PATH]
          [NO_PACKAGE_ROOT_PATH]
          [NO_CMAKE_PATH]
          [NO_CMAKE_ENVIRONMENT_PATH]
          [NO_SYSTEM_ENVIRONMENT_PATH]
          [NO_CMAKE_SYSTEM_PATH]
          [CMAKE_FIND_ROOT_PATH_BOTH |
           ONLY_CMAKE_FIND_ROOT_PATH |
           NO_CMAKE_FIND_ROOT_PATH]
         )

```

## 3.3 查找包[find_package](https://blog.csdn.net/zhanghm1995/article/details/105466372)

`find_package`本质上就是一个**搜包的命令**，通过一些特定的规则（路径）找到`<package_name>Config.cmake`或`Find<PackageName>.cmake`包配置文件，通过执行该配置文件，从而定义了一系列的变量（eg：`OpenCV_DIR`、`OpenCV_INCLUDE_DIRS`和`OpenCV_LIBS`），通过这些变量就可以准确定位到**OpenCV库的头文件和库文件**，完成编译。

find_package命令有两种工作模式，这两种工作模式的不同决定了其搜包路径的不同：

- Module模式
  find_package命令基础工作模式(Basic Signature)，也是默认工作模式。

- Config模式
  find_package命令高级工作模式(Full Signature)。 只有在find_package()中指定CONFIG、NO_MODULE等关键字，或者**Module模式查找失败后才会进入到Config模式。**

![](./legend/find_package_工作流程.png)

### module模式

```cmake
find_package(<package> [version] [EXACT] [QUIET] [MODULE]
             [REQUIRED] [[COMPONENTS] [components...]]
             [OPTIONAL_COMPONENTS components...]
             [NO_POLICY_SCOPE])

```

**Module**模式下是要查找到名为`Find<PackageName>.cmake`的配置文件。

Module模式只有两个查找路径：**CMAKE_MODULE_PATH**和cmake安装路径(**CMAKE_ROOT**)下的**Modules**目录

```cmake
# 一定记住是在这两个路径的Modules目录下查找Find<PackageName>.cmake
message(STATUS "CMAKE_MODULE_PATH = ${CMAKE_MODULE_PATH}")		# 默认为空
message(STATUS "CMAKE_ROOT = ${CMAKE_ROOT}")

# 如果直接使用find_package报找不到FindXxxx.cmake, 可以指定文件目录去查找，例如下面这个TensorRT的例子
list (APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
```



### config模式

```cmake
find_package(<package> [version] [EXACT] [QUIET]
             [REQUIRED] [[COMPONENTS] [components...]]
             [CONFIG|NO_MODULE]
             [NO_POLICY_SCOPE]
             [NAMES name1 [name2 ...]]
             [CONFIGS config1 [config2 ...]]
             [HINTS path1 [path2 ... ]]
             [PATHS path1 [path2 ... ]]
             [PATH_SUFFIXES suffix1 [suffix2 ...]]
             [NO_DEFAULT_PATH]
             [NO_CMAKE_ENVIRONMENT_PATH]
             [NO_CMAKE_PATH]
             [NO_SYSTEM_ENVIRONMENT_PATH]
             [NO_CMAKE_PACKAGE_REGISTRY]
             [NO_CMAKE_BUILDS_PATH] # Deprecated; does nothing.
             [NO_CMAKE_SYSTEM_PATH]
             [NO_CMAKE_SYSTEM_PACKAGE_REGISTRY]
             [CMAKE_FIND_ROOT_PATH_BOTH |
              ONLY_CMAKE_FIND_ROOT_PATH |
              NO_CMAKE_FIND_ROOT_PATH])

```

**CMake默认采取Module模式，如果Module模式未找到库，才会采取Config模式。**

**Config**模式下是要查找名为`<PackageName>Config.cmake`或`<lower-case-package-name>-config.cmake`的模块文件。

**Config**模式需要查找的路径非常多，具体查找顺序为：

1. 名为`<PackageName>_DIR`的CMake变量或环境变量路径
2. 名为`CMAKE_PREFIX_PATH`、`CMAKE_FRAMEWORK_PATH`、`CMAKE_APPBUNDLE_PATH`的CMake变量或**环境变量**路径
3. `PATH`环境变量路径

如果没有，CMake会继续**检查或匹配**这些根目录下的以下路径

```cmake
<prefix>/(lib/<arch>|lib|share)/cmake/<name>*/
<prefix>/(lib/<arch>|lib|share)/<name>*/ 
<prefix>/(lib/<arch>|lib|share)/<name>*/(cmake|CMake)/
```

### 查找指定位置的.cmake

如果你明确知道想要查找的库`<PackageName>Config.cmake`或`<lower-case-package-name>-config.cmake`文件所在路径，为了能够准确定位到这个包，可以直接设置变量`<PackageName>_DIR`为具体路径，如：

```cmake
set(OpenCV_DIR "/home/zhanghm/Softwares/enviroment_config/opencv3_4_4/opencv/build")
```

如果你有多个包的配置文件需要查找，可以将这些配置文件都统一放在一个命名为cmake的文件夹下，然后设置变量CMAKE_PREFIX_PATH变量指向这个cmake文件夹路径，需要注意根据上述的匹配规则，此时每个包的配置文件需要单独放置在命名为包名的文件夹下（文件夹名不区分大小写），否则会提示找不到。

```cmake
find_package(OpenCV REQUIRED)

message(WARNING "CMAKE_MODULE_PATH=${CMAKE_MODULE_PATH}")
message(WARNING "CMAKE_ROOT=${CMAKE_ROOT}")
message(WARNING "CMAKE_APPBUNDLE_PATH=${CMAKE_APPBUNDLE_PATH}")
message(WARNING "CMAKE_FRAMEWORK_PATH=${CMAKE_FRAMEWORK_PATH}")
message(WARNING "CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}")
message(WARNING ${OpenCV_DIR})
message(WARNING ${OpenCV_INCLUDE_DIRS})
message(WARNING ${OpenCV_LIBS})
```



其它例子：

```cmake
find_package(CUDA REQUIRED)
list(APPEND ALL_LIBS 
  ${CUDA_LIBRARIES} 
  ${CUDA_cublas_LIBRARY} 
  ${CUDA_nppc_LIBRARY} ${CUDA_nppig_LIBRARY} ${CUDA_nppidei_LIBRARY} ${CUDA_nppial_LIBRARY})
message(${CUDA_INCLUDE_DIRS})
```



## 3.4 查找库find_library

该命令用于查找库（动态库或者静态库，**默认查找动态库**），当构建依赖于第三方库/系统库，可以使用该命令来查找并使用库。

[每个参数的讲解](https://blog.csdn.net/fengbingchun/article/details/127232175)，[搜索路径和优先级](https://blog.csdn.net/u013250861/article/details/127935842)

```cmake
find_library(
          <VAR>
          name | NAMES name1 [name2 ...] [NAMES_PER_DIR]
          [HINTS [path | ENV var]... ]
          [PATHS [path | ENV var]... ]
          [REGISTRY_VIEW (64|32|64_32|32_64|HOST|TARGET|BOTH)]
          [PATH_SUFFIXES suffix1 [suffix2 ...]]
          [DOC "cache documentation string"]
          [NO_CACHE]
          [REQUIRED]
          [NO_DEFAULT_PATH]
          [NO_PACKAGE_ROOT_PATH]
          [NO_CMAKE_PATH]
          [NO_CMAKE_ENVIRONMENT_PATH]
          [NO_SYSTEM_ENVIRONMENT_PATH]
          [NO_CMAKE_SYSTEM_PATH]
          [NO_CMAKE_INSTALL_PREFIX]
          [CMAKE_FIND_ROOT_PATH_BOTH |
           ONLY_CMAKE_FIND_ROOT_PATH |
           NO_CMAKE_FIND_ROOT_PATH]
)
# NO_CACHE：该选项将<var>变量当成一个普通变量而不是一个缓存条目
# REQUIRED：指定该选项后，当找不到库，会输出一条错误信息并终止cmake处理过程；未指定REQUIRED选项，当find_library未找到库时，后续find_library有针对<var>的调用会继续查找。
```



- `<VAR>`

  - `<VAR>`可以是普通变量（需要指定`NO_CACHE`选项），也可以是缓存条目（意味着会存放在`CMakeCache.txt`中，不删除该文件或者用`set`重新设置该变量，其存储的值不会再刷新）；

  - 当库能被找到，`<var>`会被存放正常的库路径，当库未被找到，`<var>`中存放的值为`"<var>-NOTFOUND"`。只要`<var>`中的值`不是"<var>-NOTFOUND"`，那么即使多次调用`find_library`，`<var>`也不会再刷新;

- name

  - `name`用于指定待查找的库名称，库名称可以使用全称，例如`libmymath.a`（优先会当成全名搜索），**全称必须前后缀都在，不然查找不到**；
  - 也可以不带前缀（例如前缀`lib`）和后缀（例如`Linux`中的`.so`、`.a`，`Mac`中的`.dylib`等），直接使用`mymath`（**前后缀都不在**）；、

- NAMES：为要查找的库指定一个或多个可能的名字

  - 默认情况下此命令将一次考虑一个name并在每个目录中搜索它。（外循环多个name，内循环多个目录）
  - NAMES_PER_DIR选项告诉此命令一次考虑一个目录并搜索其中的所有名称。（外循环多个目录，内循环多个name）

- HINTS, PATHS：指定除默认位置外要搜索的目录。下面会有介绍搜索路径

- PATH_SUFFIXES：若在PATHS或HINTS指定的路径中没有找到，则继续会在PATHS/PATH_SUFFIXES或HINTS/PATH_SUFFIXES指定的路径中搜索。

```cmake
# 
unset(var CACHE) # 清除变量,带有CACHE也从缓存文件CMakeCache.txt中清除,若不带CACHE则缓存文件CMakeCache.txt中仍然存在var的值
find_library(var NAMES opencv_core) # 查找默认路径,默认查找动态库?在/usr/lib/x86_64-linux-gnu/目录下既有libopencv_core.so也有libopencv_core.a
message("var: ${var}") # var: /usr/lib/x86_64-linux-gnu/libopencv_core.so
 
# 如果找到库，则结果将存储在变量中，除非清除变量，否则不会重复搜索
find_library(var NAMES opencv_highgui) # 注意:未清除变量，不会重复搜索，最终结果是不对的，并没有查找opencv_highgui
message("var: ${var}") # var: /usr/lib/x86_64-linux-gnu/libopencv_core.so
 
unset(var CACHE) # 若不带CACHE,var是/usr/local/lib/libopencv_core.so而不是/usr/lib/x86_64-linux-gnu/libopencv_highgui.so
find_library(var NAMES opencv_highgui)
message("var: ${var}") # var: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so
 
unset(var CACHE)
find_library(var opencv_highgui) # 最简格式：find_library(<VAR> name)
message("var: ${var}") # var: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so
 
unset(var CACHE)
find_library(var NAMES opencv_xxxx) # 如果没找到库，结果将为<VAR>-NOTFOUND
message("var: ${var}") # var: var-NOTFOUND
if(${var} STREQUAL "var-NOTFOUND")
    message(WARNING "the specified library was not found")
endif()
if(NOT var) # 注意这里是var不是${var}
    message(WARNING "the specified library was not found")
endif()
unset(var) # 不带CACHE则缓存文件CMakeCache.txt中仍然存在var的值
```

库的搜索路径分为两大类： 默认搜索路径 和 附加搜索路径 。

- 默认搜索 路径包含 cmake 定义的以 CMAKE 开头的一些变量、标准的系统环境变量（例如系统环境变量 LIB 和 PATH 定义的路径）、系统的默认的库安装路径（例如 /usr 、 /usr/lib 等）；
  - 通过命令行使用 -D 指定的 CMAKE_XXX_PATH 变量，也就是形如 cmake . -DCMAKE_XXX_PATH=paths 的格式。其中 CMAKE_XXX_PATH （例如 CMAKE_LIBRARY_ARCHITECTURE 、 CMAKE_PREFIX_PATH 、 CMAKE_LIBRARY_PATH 、 CMAKE_FRAMEWORK_PATH ）
- 附加搜索路径 即 find_library 命令中通过 HINTS 或 PATHS 指定的路径；

路径搜索优先级（由高到低）：

- 通过命令行使用`-D`指定的`CMAKE_XXX_PATH`变量
- 通过在**环境变量**中指定`CMAKE_XXX_PATH`变量
-  `HINTS`选项指定的路径
- 系统环境变量指定的目录，默认是`LIB`和`PATH`指定的路径
- 跟当前系统相关的平台文件路径，一般来说指的是当前系统安装软件的标准目录，不同的操作系统对应的路径有所不同
- `PATHS`选项指定的路径。

HINTS与PATHS区别：**HINTS是在搜索系统路径之前先搜索HINTS指定的路径。PATHS是先搜索系统路径，然后再搜索PATHS指定的路径**。

### 循环查找多个库

```cmake
include(CMakePrintHelpers)		# 这是一个打印帮助工具

set(ffmpeg_libs_DIR /usr/lib/x86_64-linux-gnu)
set(ffmpeg_headers_DIR /usr/include/x86_64-linux-gnu)

SET(ffmpeg_LIB_NAME avcodec avformat avutil swresample swscale avfilter)
cmake_print_variables(ffmpeg_LIB_NAME)
SET(FFMPEG_LIBS)

FOREACH (flib IN LISTS ffmpeg_LIB_NAME)

    unset(tmp)
    find_library(tmp ${flib} HINTS ${ffmpeg_libs_DIR} NO_CACHE)
    
    if(tmp)
        LIST(APPEND FFMPEG_LIBS ${tmp})
    else()
        message("${flib} not found")
    endif ()
    
ENDFOREACH ()

cmake_print_variables(FFMPEG_LIBS)
```



## 3.5 查找头文件find_path

find_path 一般用于在某个目录下查找一个或者多个头文件，命令的执行结果会保存到 `<VAR>` 中。同时命令的执行结果也会默认缓存到 CMakeCache.txt 中。

```cmake
find_path (
          <VAR>
          NAMES name1 [name2 ...] 
          [HINTS [path | ENV var]... ]
          [PATHS [path | ENV var]... ]
          [NO_CACHE]
          [REQUIRED]
)
```

- `<VAR>`：用于保存命令的执行结果
- NAMES：要查找的头文件
- HINTS | PATHS
  - HINTS：先搜索指定路径，后搜索系统路径
  - PATHS：先搜索系统路径，后搜索指定路径
- NO_CACHE：搜索结果将存储在普通变量中而不是缓存条目（即CMakeCache.txt）中
- REQUIRED：如果没有找到指定头文件，就出发错误提示，变量会设为` <VAR>-NOTFOUND`

## 3.6 [find_package和find_library的区别](https://juejin.cn/post/7213575951114977341)

在CMake中，`find_package`和`find_library`都是用来找到和链接库的方法，但它们的用法和适用场景略有不同。

- `find_package`主要用于寻找具有CMake配置文件的库，这些库通常遵循CMake的规范，提供了用于导入目标、库路径、头文件路径等的配置文件。这使得使用`find_package`更加简洁，只需指定需要的组件即可自动处理头文件路径、库路径等。`find_package`更适合于较大、更复杂的库，如Boost。在找到库后，`find_package`会生成相关的导入目标（如`Boost::filesystem`）供你在`target_link_libraries`中使用。

- `find_library`则是一个更基本的方法，用于在系统中搜索特定的库文件。它不依赖于库提供的CMake配置文件，而是直接查找库文件。使用`find_library`时，需要手动指定库文件路径、头文件路径等。`find_library`更适合于较小或没有CMake配置文件的库，如Crypto++。比如实际应用中，我们使用`find_library`来找到Crypto++库，因为Crypto++库没有提供CMake配置文件。而对于Boost，我们使用`find_package`，因为Boost库提供了CMake配置文件，使得库的查找和链接更简便。

总之，`find_package`和`find_library`都可以用于在CMake中查找和链接库，但**`find_package`更适用于具有CMake配置文件的库，而`find_library`则适用于没有CMake配置文件的库。**

# 4 cmake命令行参数

- -D：定义CMake变量，-D参数可以用于在CMake中定义变量并将其传递给CMakeLists.txt文件，这些变量可以用于控制构建过程中的行为。

  ```cmake
  # -D参数可以用于：
  
  # 定义变量并设置其值，例如：-DVAR_NAME=VALUE。
  # 定义布尔类型的变量，其值为ON，例如：-DVAR_NAME。
  # 定义路径类型的变量，例如：-DVAR_NAME:PATH=/path/to/dir。
  # 定义配置变量（缓存变量），例如：-DVAR_NAME:STRING=VALUE。
  ```

- -B：指定构建目录。-B参数用于指定生成的构建目录，即将CMake生成的Makefile或项目文件保存到指定的目录中。这个目录可以是相对路径或绝对路径。

  ```cmake
  # cmake将使用它作为构建的根目录，如果这个目录不存在，那么cmake将会创建它
  ```

  

# 工具函数

## [工程声明project](https://www.jianshu.com/p/cdd6e56c2422)

project命令用于指定cmake工程的名称，实际上，它还可以指定cmake工程的版本号（`VERSION`关键字）、简短的描述（`DESCRIPTION`关键字）、主页URL（`HOMEPAGE_URL`关键字）和编译工程使用的语言（`LANGUAGES`关键字）。

```cmake
PROJECT(yolov8 VERSION 1.0.0 LANGUAGES C CXX CUDA DESCRIPTION “This istest project.”)
# 实际上在调用project命令指定当前工程名字的同时，cmake内部会为如下变量赋值
# 1. PROJECT_NAME：工程名，在此为yolov8
# 2. PROJECT_SOURCE_DIR：当前工程的源码路径
# 3. <PROJECT-NAME>_SOURCE_DIR：指定工程的源码路径， 若<PROJECT-NAME>就是当前工程，则该变量和PROJECT_SOURCE_DIR相等。
# 4. PROJECT_BINARY_DIR：当前工程的二进制路径
# 5. <PROJECT-NAME>_BINARY_DIR：指定工程的二进制路径， 若<PROJECT-NAME>就是当前工程，则该变量和PROJECT_BINARY_DIR相等。
# 6. CMAKE_PROJECT_NAME：顶层工程的名称。

# 在调用project命令指定VERSION，LANGUAGES等项时，cmake内部也会生成相关变量

# 有如下结构
/test
/test/CMakeLists.txt
/test/subtest
/test/subtest/CMakeLists.txt


# test/CMakeLists.txt
cmake_minimum_required (VERSION 3.10.2)
set (TOP_PROJECT_NAME "mytest") # 定义了变量TOP_PROJECT_NAME为"mytest"
project (${TOP_PROJECT_NAME}) 

message (">>> top PROJECT_NAME: ${PROJECT_NAME}")
message (">>> top PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
message (">>> top <PROJECT_NAME>_SOURCE_DIR: ${${TOP_PROJECT_NAME}_SOURCE_DIR}") 
message (">>> top PROJECT_BINARY_DIR: ${PROJECT_BINARY_DIR}")
message (">>> top <PROJECT_NAME>_BINARY_DIR: ${${TOP_PROJECT_NAME}_BINARY_DIR}")
message (">>> top CMAKE_PROJECT_NAME: ${CMAKE_PROJECT_NAME}")

add_subdirectory (sub_test) # 调用sub_test下的CMakeList.txt进行构建

# test/sub_test/CMakeLists.txt
cmake_minimum_required (VERSION 3.10.2)
set (SUB_LEVEL_PROJECT_NAME "mysubtest") # 定义了变量SUB_LEVEL_PROJECT_NAME为"mysubtest"
project (${SUB_LEVEL_PROJECT_NAME}) 

message (">>>>>> sub PROJECT_NAME: ${PROJECT_NAME}")
message (">>>>>> sub PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
message (">>>>>> sub <PROJECT_NAME>_SOURCE_DIR: ${${SUB_LEVEL_PROJECT_NAME}_SOURCE_DIR}") 
message (">>>>>> sub <PROJECT_NAME>_SOURCE_DIR(top level): ${${TOP_PROJECT_NAME}_SOURCE_DIR}") 
message (">>>>>> sub PROJECT_BINARY_DIR: ${PROJECT_BINARY_DIR}")
message (">>>>>> sub <PROJECT_NAME>_BINARY_DIR: ${${SUB_LEVEL_PROJECT_NAME}_BINARY_DIR}")
message (">>>>>> sub <PROJECT_NAME>_BINARY_DIR(top level): ${${TOP_PROJECT_NAME}_BINARY_DIR}")
message (">>>>>> sub CMAKE_PROJECT_NAME: ${CMAKE_PROJECT_NAME}")

# 在test/下执行cmake .
>>> top PROJECT_NAME: mytest
>>> top PROJECT_SOURCE_DIR: test/
>>> top <PROJECT_NAME>_SOURCE_DIR: test/
>>> top PROJECT_BINARY_DIR: test/ 
>>> top <PROJECT_NAME>_BINARY_DIR: test/
>>> top CMAKE_PROJECT_NAME: mytest
>>>>>> sub PROJECT_NAME: mysubtest
>>>>>> sub PROJECT_SOURCE_DIR: test/sub_test
>>>>>> sub <PROJECT_NAME>_SOURCE_DIR: test/sub_test
>>>>>> sub <PROJECT_NAME>_SOURCE_DIR(top level): test/
>>>>>> sub PROJECT_BINARY_DIR: test/sub_test
>>>>>> sub <PROJECT_NAME>_BINARY_DIR: test/sub_test
>>>>>> sub <PROJECT_NAME>_BINARY_DIR(top level): test/
>>>>>> sub CMAKE_PROJECT_NAME: mytest
```



## [获取目录下所有文件：`aux_source_directory，file`](https://blog.csdn.net/a2886015/article/details/126830638)

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

## 打印message

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

## [configure_file](https://www.jianshu.com/p/2946eeec2489)

- 将一个文件(由`input`参数指定)拷贝到指定位置(由`output`参数指定)，并根据`options`修改其内容。

- **configure_file命令一般用于自定义编译选项或者自定义宏的场景。**

- `configure_file(<input> <output> [options])`

- configure_file命令会根据`options`指定的规则，自动对`input`文件中`cmakedefine`关键字及其内容进行转换。

- 具体来说，会做如下几个替换：

  - 将input文件中的`@var@`或者`${var}`替换成cmake中指定的值。
  -  将input文件中的`#cmakedefine var`关键字替换成`#define var`或者`#undef var`，取决于cmake是否定义了var。

- c++获取项目路径的两种方式中有应用

## [file](https://www.jianshu.com/p/ed151fdcf473)

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

C++中自带函数getenv/putenv（获取和设置），可以读取指定的环境变量，返回char *。

```c
#include <stdlib.h> 

// 返回指向value的指针，若未找到则为NULL 
char *getenv(const char *name);

// 以 "var_name=value"的形式设置环境变量
int putenv(char *string);
```

## 2 [camke发布release版本](https://blog.csdn.net/weixin_39766005/article/details/122439200)

```bash
# cmake构建的项目默认是debug,可以通过下面命令构建release版本：

cmake --build . --config Release

# 在构建的最后加上 --config Release即可。
```



# 其它

1. [**target_xxx 中的 PUBLIC，PRIVATE，INTERFACE**](https://zhuanlan.zhihu.com/p/82244559)
   - PRIVATE：当前的修饰对象item对target的上层（target的调用层）不可见，target的调用层不使用当前修饰对象的功能。target无需暴露并且隐藏该item
   
   - INTERFACE：target不使用当前修饰对象item的功能，可target的调用层会使用到当前修饰对象的功能，target需要暴露当前item的接口
   
   - PUBLIC = INTERFACE + PRIVATE
   
   - | 情况                             | 使用参数  |
     | -------------------------------- | --------- |
     | 只有源文件（.cpp）中包含了库文件 | PRIVATE   |
     | 只有头文件（.hpp）中包含了库文件 | INTERFACE |
     | 源文件和头文件都包含了库文件     | PUBLIC    |
   
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

2. [读取json](https://blog.csdn.net/luxpity/article/details/116809954)

3. tinyxml

   - [Linux 下配置Tinyxml,将其编译为静态库](https://blog.csdn.net/yasi_xi/article/details/38872467)
   - [Cmake链接tinyxml静态库](https://blog.csdn.net/qq_40089175/article/details/107536133)

4. 


# log

1. [合并静态库](https://zhuanlan.zhihu.com/p/389448385)

1. 设置可执行目标文件的输出目录：`SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY output) `

1. [子目录创建静态库和动态库](https://blog.csdn.net/Jay_Xio/article/details/122019770)

1. [使用子目录](https://blog.csdn.net/jjjstephen/article/details/122455635)

1. [动态库无法链接静态库](https://zhuanlan.zhihu.com/p/631234923)

   - 我们首先添加一个静态库`otherlib`，然后再添加一个动态库`mylib`，但是这个动态库需要链接静态库`otherlib`，此时就会出错，让静态库编译时也**生成位置无关的代码**(PIC)，这样才能装在动态库里

   - ```cmake
     add_library(otherlib STATIC otherlib.cpp)
     set_property(TARGET otherlib PROPERTY POSITION_INDEPENDENT_CODE ON)			# 只针对一个库启用位置无关的代码(PIC)
     
     add_library(mylib SHARED mylib.cpp)
     target_link_libraries(mylib PUBLIC otherlib)
     
     add_executable(main main.cpp)
     target_link_libraries(main PUBLIC mylib)
     ```

   - 

1. [彻底清除cmake产生的缓存](https://blog.csdn.net/kris_fei/article/details/81982565)：删除cmake产生的build和release产物

1. 

   

# 1 从可执行文件到库

## 1.1 生成器

CMake是一个构建系统生成器，可以使用单个CMakeLists.txt为不同平台上的不同工具集配置项目