include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
aux_source_directory(dep JSON_SRC_DEP)
add_library(JsonTrans
        STATIC
        jsoncpp.cpp ${JSON_SRC_DEP})
target_include_directories(JsonTrans PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)