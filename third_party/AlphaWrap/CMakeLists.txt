cmake_minimum_required(VERSION 3.1)
project(alpha_wrap)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CGAL REQUIRED)
find_package(Boost REQUIRED)

add_executable(alpha_wrap src/alpha_wrap.cpp)
target_link_libraries(alpha_wrap PRIVATE CGAL::CGAL)
