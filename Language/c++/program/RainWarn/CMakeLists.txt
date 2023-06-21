cmake_minimum_required(VERSION 3.25)
project(RainWarn)

set(CMAKE_CXX_STANDARD 11)
include_directories(/usr/include/mysql  /usr/include)
link_directories(/usr/lib/x86_64-linux-gnu)
include_directories(${PROJECT_SOURCE_DIR}/includes)
link_directories(${PROJECT_SOURCE_DIR}/libs)
link_libraries(tinyxml)

add_executable(${PROJECT_NAME} main.cpp Rain.cpp DataBS.cpp ReadConfigInfo.cpp)
# 如果在add_executable，不加Rain.cpp
# 就会报undefined reference to `Rain::hello()'

target_link_libraries(${PROJECT_NAME} mysqlclient)

#find_package(libtinyxml.a REQUIRED)
#target_link_libraries(${PROJECT_NAME} tinyxml)
