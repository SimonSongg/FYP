# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_SOURCE_DIR = /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build

# Include any dependencies generated for this target.
include g2o/examples/simple_optimize/CMakeFiles/simple_optimize.dir/depend.make

# Include the progress variables for this target.
include g2o/examples/simple_optimize/CMakeFiles/simple_optimize.dir/progress.make

# Include the compile flags for this target's objects.
include g2o/examples/simple_optimize/CMakeFiles/simple_optimize.dir/flags.make

g2o/examples/simple_optimize/CMakeFiles/simple_optimize.dir/simple_optimize.cpp.o: g2o/examples/simple_optimize/CMakeFiles/simple_optimize.dir/flags.make
g2o/examples/simple_optimize/CMakeFiles/simple_optimize.dir/simple_optimize.cpp.o: ../g2o/examples/simple_optimize/simple_optimize.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object g2o/examples/simple_optimize/CMakeFiles/simple_optimize.dir/simple_optimize.cpp.o"
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/examples/simple_optimize && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/simple_optimize.dir/simple_optimize.cpp.o -c /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/g2o/examples/simple_optimize/simple_optimize.cpp

g2o/examples/simple_optimize/CMakeFiles/simple_optimize.dir/simple_optimize.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/simple_optimize.dir/simple_optimize.cpp.i"
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/examples/simple_optimize && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/g2o/examples/simple_optimize/simple_optimize.cpp > CMakeFiles/simple_optimize.dir/simple_optimize.cpp.i

g2o/examples/simple_optimize/CMakeFiles/simple_optimize.dir/simple_optimize.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/simple_optimize.dir/simple_optimize.cpp.s"
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/examples/simple_optimize && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/g2o/examples/simple_optimize/simple_optimize.cpp -o CMakeFiles/simple_optimize.dir/simple_optimize.cpp.s

g2o/examples/simple_optimize/CMakeFiles/simple_optimize.dir/simple_optimize.cpp.o.requires:

.PHONY : g2o/examples/simple_optimize/CMakeFiles/simple_optimize.dir/simple_optimize.cpp.o.requires

g2o/examples/simple_optimize/CMakeFiles/simple_optimize.dir/simple_optimize.cpp.o.provides: g2o/examples/simple_optimize/CMakeFiles/simple_optimize.dir/simple_optimize.cpp.o.requires
	$(MAKE) -f g2o/examples/simple_optimize/CMakeFiles/simple_optimize.dir/build.make g2o/examples/simple_optimize/CMakeFiles/simple_optimize.dir/simple_optimize.cpp.o.provides.build
.PHONY : g2o/examples/simple_optimize/CMakeFiles/simple_optimize.dir/simple_optimize.cpp.o.provides

g2o/examples/simple_optimize/CMakeFiles/simple_optimize.dir/simple_optimize.cpp.o.provides.build: g2o/examples/simple_optimize/CMakeFiles/simple_optimize.dir/simple_optimize.cpp.o


# Object files for target simple_optimize
simple_optimize_OBJECTS = \
"CMakeFiles/simple_optimize.dir/simple_optimize.cpp.o"

# External object files for target simple_optimize
simple_optimize_EXTERNAL_OBJECTS =

../bin/simple_optimize: g2o/examples/simple_optimize/CMakeFiles/simple_optimize.dir/simple_optimize.cpp.o
../bin/simple_optimize: g2o/examples/simple_optimize/CMakeFiles/simple_optimize.dir/build.make
../bin/simple_optimize: ../lib/libg2o_solver_csparse.so
../bin/simple_optimize: ../lib/libg2o_types_slam2d.so
../bin/simple_optimize: ../lib/libg2o_types_slam3d.so
../bin/simple_optimize: ../lib/libg2o_csparse_extension.so
../bin/simple_optimize: /usr/lib/aarch64-linux-gnu/libcxsparse.so
../bin/simple_optimize: ../lib/libg2o_core.so
../bin/simple_optimize: ../lib/libg2o_stuff.so
../bin/simple_optimize: ../lib/libg2o_opengl_helper.so
../bin/simple_optimize: /usr/lib/aarch64-linux-gnu/libGLU.so
../bin/simple_optimize: /usr/lib/aarch64-linux-gnu/libGL.so
../bin/simple_optimize: g2o/examples/simple_optimize/CMakeFiles/simple_optimize.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../../../bin/simple_optimize"
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/examples/simple_optimize && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/simple_optimize.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
g2o/examples/simple_optimize/CMakeFiles/simple_optimize.dir/build: ../bin/simple_optimize

.PHONY : g2o/examples/simple_optimize/CMakeFiles/simple_optimize.dir/build

g2o/examples/simple_optimize/CMakeFiles/simple_optimize.dir/requires: g2o/examples/simple_optimize/CMakeFiles/simple_optimize.dir/simple_optimize.cpp.o.requires

.PHONY : g2o/examples/simple_optimize/CMakeFiles/simple_optimize.dir/requires

g2o/examples/simple_optimize/CMakeFiles/simple_optimize.dir/clean:
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/examples/simple_optimize && $(CMAKE_COMMAND) -P CMakeFiles/simple_optimize.dir/cmake_clean.cmake
.PHONY : g2o/examples/simple_optimize/CMakeFiles/simple_optimize.dir/clean

g2o/examples/simple_optimize/CMakeFiles/simple_optimize.dir/depend:
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2 /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/g2o/examples/simple_optimize /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/examples/simple_optimize /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/examples/simple_optimize/CMakeFiles/simple_optimize.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : g2o/examples/simple_optimize/CMakeFiles/simple_optimize.dir/depend

