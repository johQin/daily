cmake_minimum_required(VERSION 3.25)
project(RainWarn)

set(CMAKE_CXX_STANDARD 11)

set(PROJECT_ROOT_PATH ${PROJECT_SOURCE_DIR})
configure_file(includes/SelfENV.h.in ../includes/SelfENV.h)

# mysql_client
include_directories(/usr/include/mysql  /usr/include)
link_directories(/usr/lib/x86_64-linux-gnu)

include_directories(${PROJECT_SOURCE_DIR}/includes)             # include_directories, 影响范围最大,可以为CMakelists.txt后的所有target添加头文件目录,
                                                                #一般写在最外层CMakelists.txt中影响全局（向下传递，父目录包含，那么其子目录也自动包含）
link_directories(${PROJECT_SOURCE_DIR}/libs)
link_libraries(tinyxml)

add_subdirectory("utils/xml")
aux_source_directory(src MAIN_SRC_DEP)
aux_source_directory(utils UTILS_SRC_DEP)
#add_executable(${PROJECT_NAME} main.cpp Rain.cpp DataBS.cpp ReadConfigInfo.cpp Markup.cpp Utils.cpp)
add_executable(${PROJECT_NAME} main.cpp ${MAIN_SRC_DEP} ${UTILS_SRC_DEP})

target_link_libraries(${PROJECT_NAME} ReadXml)
target_link_libraries(${PROJECT_NAME} mysqlclient)


#find_package(libtinyxml.a REQUIRED)
#target_link_libraries(${PROJECT_NAME} tinyxml)
