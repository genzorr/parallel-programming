cmake_minimum_required(VERSION 3.20)
project(copyin)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenMP)

file(GLOB SOURCES
        copyin.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()
