# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lee/ELAS_Opencv/ls_elas

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lee/ELAS_Opencv/ls_elas/build

# Include any dependencies generated for this target.
include CMakeFiles/ls_elas.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ls_elas.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ls_elas.dir/flags.make

CMakeFiles/ls_elas.dir/src/main.cpp.o: CMakeFiles/ls_elas.dir/flags.make
CMakeFiles/ls_elas.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lee/ELAS_Opencv/ls_elas/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ls_elas.dir/src/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ls_elas.dir/src/main.cpp.o -c /home/lee/ELAS_Opencv/ls_elas/src/main.cpp

CMakeFiles/ls_elas.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ls_elas.dir/src/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lee/ELAS_Opencv/ls_elas/src/main.cpp > CMakeFiles/ls_elas.dir/src/main.cpp.i

CMakeFiles/ls_elas.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ls_elas.dir/src/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lee/ELAS_Opencv/ls_elas/src/main.cpp -o CMakeFiles/ls_elas.dir/src/main.cpp.s

CMakeFiles/ls_elas.dir/src/main.cpp.o.requires:

.PHONY : CMakeFiles/ls_elas.dir/src/main.cpp.o.requires

CMakeFiles/ls_elas.dir/src/main.cpp.o.provides: CMakeFiles/ls_elas.dir/src/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/ls_elas.dir/build.make CMakeFiles/ls_elas.dir/src/main.cpp.o.provides.build
.PHONY : CMakeFiles/ls_elas.dir/src/main.cpp.o.provides

CMakeFiles/ls_elas.dir/src/main.cpp.o.provides.build: CMakeFiles/ls_elas.dir/src/main.cpp.o


CMakeFiles/ls_elas.dir/src/elas.cpp.o: CMakeFiles/ls_elas.dir/flags.make
CMakeFiles/ls_elas.dir/src/elas.cpp.o: ../src/elas.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lee/ELAS_Opencv/ls_elas/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/ls_elas.dir/src/elas.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ls_elas.dir/src/elas.cpp.o -c /home/lee/ELAS_Opencv/ls_elas/src/elas.cpp

CMakeFiles/ls_elas.dir/src/elas.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ls_elas.dir/src/elas.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lee/ELAS_Opencv/ls_elas/src/elas.cpp > CMakeFiles/ls_elas.dir/src/elas.cpp.i

CMakeFiles/ls_elas.dir/src/elas.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ls_elas.dir/src/elas.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lee/ELAS_Opencv/ls_elas/src/elas.cpp -o CMakeFiles/ls_elas.dir/src/elas.cpp.s

CMakeFiles/ls_elas.dir/src/elas.cpp.o.requires:

.PHONY : CMakeFiles/ls_elas.dir/src/elas.cpp.o.requires

CMakeFiles/ls_elas.dir/src/elas.cpp.o.provides: CMakeFiles/ls_elas.dir/src/elas.cpp.o.requires
	$(MAKE) -f CMakeFiles/ls_elas.dir/build.make CMakeFiles/ls_elas.dir/src/elas.cpp.o.provides.build
.PHONY : CMakeFiles/ls_elas.dir/src/elas.cpp.o.provides

CMakeFiles/ls_elas.dir/src/elas.cpp.o.provides.build: CMakeFiles/ls_elas.dir/src/elas.cpp.o


CMakeFiles/ls_elas.dir/src/orb_extractor.cpp.o: CMakeFiles/ls_elas.dir/flags.make
CMakeFiles/ls_elas.dir/src/orb_extractor.cpp.o: ../src/orb_extractor.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lee/ELAS_Opencv/ls_elas/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/ls_elas.dir/src/orb_extractor.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ls_elas.dir/src/orb_extractor.cpp.o -c /home/lee/ELAS_Opencv/ls_elas/src/orb_extractor.cpp

CMakeFiles/ls_elas.dir/src/orb_extractor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ls_elas.dir/src/orb_extractor.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lee/ELAS_Opencv/ls_elas/src/orb_extractor.cpp > CMakeFiles/ls_elas.dir/src/orb_extractor.cpp.i

CMakeFiles/ls_elas.dir/src/orb_extractor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ls_elas.dir/src/orb_extractor.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lee/ELAS_Opencv/ls_elas/src/orb_extractor.cpp -o CMakeFiles/ls_elas.dir/src/orb_extractor.cpp.s

CMakeFiles/ls_elas.dir/src/orb_extractor.cpp.o.requires:

.PHONY : CMakeFiles/ls_elas.dir/src/orb_extractor.cpp.o.requires

CMakeFiles/ls_elas.dir/src/orb_extractor.cpp.o.provides: CMakeFiles/ls_elas.dir/src/orb_extractor.cpp.o.requires
	$(MAKE) -f CMakeFiles/ls_elas.dir/build.make CMakeFiles/ls_elas.dir/src/orb_extractor.cpp.o.provides.build
.PHONY : CMakeFiles/ls_elas.dir/src/orb_extractor.cpp.o.provides

CMakeFiles/ls_elas.dir/src/orb_extractor.cpp.o.provides.build: CMakeFiles/ls_elas.dir/src/orb_extractor.cpp.o


CMakeFiles/ls_elas.dir/src/descriptor.cpp.o: CMakeFiles/ls_elas.dir/flags.make
CMakeFiles/ls_elas.dir/src/descriptor.cpp.o: ../src/descriptor.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lee/ELAS_Opencv/ls_elas/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/ls_elas.dir/src/descriptor.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ls_elas.dir/src/descriptor.cpp.o -c /home/lee/ELAS_Opencv/ls_elas/src/descriptor.cpp

CMakeFiles/ls_elas.dir/src/descriptor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ls_elas.dir/src/descriptor.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lee/ELAS_Opencv/ls_elas/src/descriptor.cpp > CMakeFiles/ls_elas.dir/src/descriptor.cpp.i

CMakeFiles/ls_elas.dir/src/descriptor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ls_elas.dir/src/descriptor.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lee/ELAS_Opencv/ls_elas/src/descriptor.cpp -o CMakeFiles/ls_elas.dir/src/descriptor.cpp.s

CMakeFiles/ls_elas.dir/src/descriptor.cpp.o.requires:

.PHONY : CMakeFiles/ls_elas.dir/src/descriptor.cpp.o.requires

CMakeFiles/ls_elas.dir/src/descriptor.cpp.o.provides: CMakeFiles/ls_elas.dir/src/descriptor.cpp.o.requires
	$(MAKE) -f CMakeFiles/ls_elas.dir/build.make CMakeFiles/ls_elas.dir/src/descriptor.cpp.o.provides.build
.PHONY : CMakeFiles/ls_elas.dir/src/descriptor.cpp.o.provides

CMakeFiles/ls_elas.dir/src/descriptor.cpp.o.provides.build: CMakeFiles/ls_elas.dir/src/descriptor.cpp.o


# Object files for target ls_elas
ls_elas_OBJECTS = \
"CMakeFiles/ls_elas.dir/src/main.cpp.o" \
"CMakeFiles/ls_elas.dir/src/elas.cpp.o" \
"CMakeFiles/ls_elas.dir/src/orb_extractor.cpp.o" \
"CMakeFiles/ls_elas.dir/src/descriptor.cpp.o"

# External object files for target ls_elas
ls_elas_EXTERNAL_OBJECTS =

ls_elas: CMakeFiles/ls_elas.dir/src/main.cpp.o
ls_elas: CMakeFiles/ls_elas.dir/src/elas.cpp.o
ls_elas: CMakeFiles/ls_elas.dir/src/orb_extractor.cpp.o
ls_elas: CMakeFiles/ls_elas.dir/src/descriptor.cpp.o
ls_elas: CMakeFiles/ls_elas.dir/build.make
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_stitching3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_superres3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_videostab3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_aruco3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_bgsegm3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_bioinspired3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_ccalib3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_cvv3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_dpm3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_face3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_fuzzy3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_hdf3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_img_hash3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_line_descriptor3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_optflow3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_reg3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_rgbd3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_saliency3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_stereo3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_structured_light3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_surface_matching3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_tracking3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_xfeatures2d3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_ximgproc3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_xobjdetect3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_xphoto3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/libcv_bridge.so
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_core3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_imgproc3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_imgcodecs3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/libtf.so
ls_elas: /opt/ros/kinetic/lib/libtf2_ros.so
ls_elas: /opt/ros/kinetic/lib/libactionlib.so
ls_elas: /opt/ros/kinetic/lib/libmessage_filters.so
ls_elas: /opt/ros/kinetic/lib/libroscpp.so
ls_elas: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
ls_elas: /usr/lib/x86_64-linux-gnu/libboost_signals.so
ls_elas: /opt/ros/kinetic/lib/libxmlrpcpp.so
ls_elas: /opt/ros/kinetic/lib/libtf2.so
ls_elas: /opt/ros/kinetic/lib/libroscpp_serialization.so
ls_elas: /opt/ros/kinetic/lib/librosconsole.so
ls_elas: /opt/ros/kinetic/lib/librosconsole_log4cxx.so
ls_elas: /opt/ros/kinetic/lib/librosconsole_backend_interface.so
ls_elas: /usr/lib/x86_64-linux-gnu/liblog4cxx.so
ls_elas: /usr/lib/x86_64-linux-gnu/libboost_regex.so
ls_elas: /opt/ros/kinetic/lib/librostime.so
ls_elas: /opt/ros/kinetic/lib/libcpp_common.so
ls_elas: /usr/lib/x86_64-linux-gnu/libboost_system.so
ls_elas: /usr/lib/x86_64-linux-gnu/libboost_thread.so
ls_elas: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
ls_elas: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
ls_elas: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
ls_elas: /usr/lib/x86_64-linux-gnu/libpthread.so
ls_elas: /usr/lib/x86_64-linux-gnu/libconsole_bridge.so
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_shape3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_photo3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_datasets3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_plot3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_text3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_dnn3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_ml3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_video3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_calib3d3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_features2d3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_highgui3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_videoio3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_viz3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_phase_unwrapping3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_flann3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_imgcodecs3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_objdetect3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_imgproc3.so.3.3.1
ls_elas: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_core3.so.3.3.1
ls_elas: CMakeFiles/ls_elas.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lee/ELAS_Opencv/ls_elas/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable ls_elas"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ls_elas.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ls_elas.dir/build: ls_elas

.PHONY : CMakeFiles/ls_elas.dir/build

CMakeFiles/ls_elas.dir/requires: CMakeFiles/ls_elas.dir/src/main.cpp.o.requires
CMakeFiles/ls_elas.dir/requires: CMakeFiles/ls_elas.dir/src/elas.cpp.o.requires
CMakeFiles/ls_elas.dir/requires: CMakeFiles/ls_elas.dir/src/orb_extractor.cpp.o.requires
CMakeFiles/ls_elas.dir/requires: CMakeFiles/ls_elas.dir/src/descriptor.cpp.o.requires

.PHONY : CMakeFiles/ls_elas.dir/requires

CMakeFiles/ls_elas.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ls_elas.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ls_elas.dir/clean

CMakeFiles/ls_elas.dir/depend:
	cd /home/lee/ELAS_Opencv/ls_elas/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lee/ELAS_Opencv/ls_elas /home/lee/ELAS_Opencv/ls_elas /home/lee/ELAS_Opencv/ls_elas/build /home/lee/ELAS_Opencv/ls_elas/build /home/lee/ELAS_Opencv/ls_elas/build/CMakeFiles/ls_elas.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ls_elas.dir/depend
