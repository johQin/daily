include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
aux_source_directory(dep XML_SRC_DEP)
add_library(ReadXml
        STATIC
        ReadConfigInfo.cpp ${XML_SRC_DEP})
target_include_directories(ReadXml PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)