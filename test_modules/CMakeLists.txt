# project
cmake_minimum_required (VERSION 3.12.0)
add_compile_options(-std=c++11)
project (test_support_match)

find_package(PkgConfig)
find_package(Boost COMPONENTS program_options system REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost 1.58.0 REQUIRED COMPONENTS system filesystem)

# directories
set (LIBELAS_SRC_DIR src)

# include directory
include_directories("${LIBELAS_SRC_DIR}")
include_directories(... ${Boost_INCLUDE_DIRS})
# use sse3 instruction set
SET(CMAKE_CXX_FLAGS "-msse3")

# sources
FILE(GLOB LIBELAS_SRC_FILES "src/*.cpp")

# make release version
set(CMAKE_BUILD_TYPE Release)
link_directories(... ${Boost_LIBRARY_DIRS})
# build demo program
add_executable(test_ori_elas ${LIBELAS_SRC_FILES})
target_link_libraries(test_ori_elas ${OpenCV_LIBS}
 ${Boost_LIBRARIES})


