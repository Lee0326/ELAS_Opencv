# project
cmake_minimum_required (VERSION 2.6)
project (ls_elas)
add_compile_options(-std=c++11)

find_package(PkgConfig)
find_package(Boost COMPONENTS program_options system REQUIRED)
find_package(OpenCV REQUIRED)
find_package(catkin REQUIRED cv_bridge tf)

# directories
set (LIBELAS_SRC_DIR src)
set (INC_DIR /usr/local/include/include_fade2d)
set (LINK_DIR /usr/local/lib)

# include directory
include_directories(
  ${LIBELAS_SRC_DIR}
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIR}
  ${INC_DIR})

link_directories(${LINK_DIR})

# use sse3 instruction set
SET(CMAKE_CXX_FLAGS "-msse3")

# sources
FILE(GLOB LIBELAS_SRC_FILES "src/*.cpp")

# make release version
set(CMAKE_BUILD_TYPE Release)

# build demo program
add_executable(ls_elas ${LIBELAS_SRC_FILES})
target_link_libraries(ls_elas ${OpenCV_LIBS} ${catkin_LIBRARIES} libfade2d.so)

