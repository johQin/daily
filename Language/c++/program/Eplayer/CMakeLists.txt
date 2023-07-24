cmake_minimum_required(VERSION 3.25)
project(Eplayer)

set(CMAKE_CXX_STANDARD 17)
include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB_RECURSE SRC_CPP "src/*.cpp")

add_executable(Eplayer main.cpp ${SRC_CPP})
