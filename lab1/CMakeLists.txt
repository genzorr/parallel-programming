cmake_minimum_required(VERSION 3.20)
project(lab1)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenMP)

file(GLOB SOURCES
        task-0.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()
