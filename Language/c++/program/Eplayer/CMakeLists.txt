cmake_minimum_required(VERSION 3.25)
project(Eplayer)

set(CMAKE_CXX_STANDARD 17)
include_directories(${PROJECT_SOURCE_DIR}/include/sqlite3)
include_directories(${PROJECT_SOURCE_DIR}/include)
# mysql_client
include_directories(/usr/include/mysql  /usr/include)
link_directories(/usr/lib/x86_64-linux-gnu)

file(GLOB_RECURSE SRC_CPP "src/*.cpp")
file(GLOB_RECURSE INCLUDE_CPP "include/*.cpp" "include/*.c" "include/sqlite3/*.c")
add_executable(Eplayer main.cpp ${INCLUDE_CPP} ${SRC_CPP})

target_link_libraries(${PROJECT_NAME} mysqlclient crypto)