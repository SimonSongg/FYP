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
include g2o/examples/ba/CMakeFiles/ba_demo.dir/depend.make

# Include the progress variables for this target.
include g2o/examples/ba/CMakeFiles/ba_demo.dir/progress.make

# Include the compile flags for this target's objects.
include g2o/examples/ba/CMakeFiles/ba_demo.dir/flags.make

g2o/examples/ba/CMakeFiles/ba_demo.dir/ba_demo.cpp.o: g2o/examples/ba/CMakeFiles/ba_demo.dir/flags.make
g2o/examples/ba/CMakeFiles/ba_demo.dir/ba_demo.cpp.o: ../g2o/examples/ba/ba_demo.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object g2o/examples/ba/CMakeFiles/ba_demo.dir/ba_demo.cpp.o"
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/examples/ba && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ba_demo.dir/ba_demo.cpp.o -c /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/g2o/examples/ba/ba_demo.cpp

g2o/examples/ba/CMakeFiles/ba_demo.dir/ba_demo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ba_demo.dir/ba_demo.cpp.i"
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/examples/ba && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/g2o/examples/ba/ba_demo.cpp > CMakeFiles/ba_demo.dir/ba_demo.cpp.i

g2o/examples/ba/CMakeFiles/ba_demo.dir/ba_demo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ba_demo.dir/ba_demo.cpp.s"
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/examples/ba && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/g2o/examples/ba/ba_demo.cpp -o CMakeFiles/ba_demo.dir/ba_demo.cpp.s

g2o/examples/ba/CMakeFiles/ba_demo.dir/ba_demo.cpp.o.requires:

.PHONY : g2o/examples/ba/CMakeFiles/ba_demo.dir/ba_demo.cpp.o.requires

g2o/examples/ba/CMakeFiles/ba_demo.dir/ba_demo.cpp.o.provides: g2o/examples/ba/CMakeFiles/ba_demo.dir/ba_demo.cpp.o.requires
	$(MAKE) -f g2o/examples/ba/CMakeFiles/ba_demo.dir/build.make g2o/examples/ba/CMakeFiles/ba_demo.dir/ba_demo.cpp.o.provides.build
.PHONY : g2o/examples/ba/CMakeFiles/ba_demo.dir/ba_demo.cpp.o.provides

g2o/examples/ba/CMakeFiles/ba_demo.dir/ba_demo.cpp.o.provides.build: g2o/examples/ba/CMakeFiles/ba_demo.dir/ba_demo.cpp.o


# Object files for target ba_demo
ba_demo_OBJECTS = \
"CMakeFiles/ba_demo.dir/ba_demo.cpp.o"

# External object files for target ba_demo
ba_demo_EXTERNAL_OBJECTS =

../bin/ba_demo: g2o/examples/ba/CMakeFiles/ba_demo.dir/ba_demo.cpp.o
../bin/ba_demo: g2o/examples/ba/CMakeFiles/ba_demo.dir/build.make
../bin/ba_demo: ../lib/libg2o_solver_cholmod.so
../bin/ba_demo: ../lib/libg2o_types_sba.so
../bin/ba_demo: /usr/lib/aarch64-linux-gnu/libcholmod.so
../bin/ba_demo: /usr/lib/aarch64-linux-gnu/libamd.so
../bin/ba_demo: /usr/lib/aarch64-linux-gnu/libcolamd.so
../bin/ba_demo: /usr/lib/aarch64-linux-gnu/libcamd.so
../bin/ba_demo: /usr/lib/aarch64-linux-gnu/libccolamd.so
../bin/ba_demo: /usr/lib/aarch64-linux-gnu/libsuitesparseconfig.so
../bin/ba_demo: /usr/lib/aarch64-linux-gnu/libblas.so
../bin/ba_demo: /usr/lib/aarch64-linux-gnu/liblapack.so
../bin/ba_demo: ../lib/libg2o_types_slam3d.so
../bin/ba_demo: ../lib/libg2o_core.so
../bin/ba_demo: ../lib/libg2o_stuff.so
../bin/ba_demo: ../lib/libg2o_opengl_helper.so
../bin/ba_demo: /usr/lib/aarch64-linux-gnu/libGLU.so
../bin/ba_demo: /usr/lib/aarch64-linux-gnu/libGL.so
../bin/ba_demo: g2o/examples/ba/CMakeFiles/ba_demo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../../../bin/ba_demo"
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/examples/ba && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ba_demo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
g2o/examples/ba/CMakeFiles/ba_demo.dir/build: ../bin/ba_demo

.PHONY : g2o/examples/ba/CMakeFiles/ba_demo.dir/build

g2o/examples/ba/CMakeFiles/ba_demo.dir/requires: g2o/examples/ba/CMakeFiles/ba_demo.dir/ba_demo.cpp.o.requires

.PHONY : g2o/examples/ba/CMakeFiles/ba_demo.dir/requires

g2o/examples/ba/CMakeFiles/ba_demo.dir/clean:
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/examples/ba && $(CMAKE_COMMAND) -P CMakeFiles/ba_demo.dir/cmake_clean.cmake
.PHONY : g2o/examples/ba/CMakeFiles/ba_demo.dir/clean

g2o/examples/ba/CMakeFiles/ba_demo.dir/depend:
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2 /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/g2o/examples/ba /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/examples/ba /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/examples/ba/CMakeFiles/ba_demo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : g2o/examples/ba/CMakeFiles/ba_demo.dir/depend

