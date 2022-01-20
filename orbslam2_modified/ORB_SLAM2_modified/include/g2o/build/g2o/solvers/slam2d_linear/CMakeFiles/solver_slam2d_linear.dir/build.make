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
include g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/depend.make

# Include the progress variables for this target.
include g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/progress.make

# Include the compile flags for this target's objects.
include g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/flags.make

g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/slam2d_linear.cpp.o: g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/flags.make
g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/slam2d_linear.cpp.o: ../g2o/solvers/slam2d_linear/slam2d_linear.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/slam2d_linear.cpp.o"
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/solvers/slam2d_linear && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/solver_slam2d_linear.dir/slam2d_linear.cpp.o -c /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/g2o/solvers/slam2d_linear/slam2d_linear.cpp

g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/slam2d_linear.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/solver_slam2d_linear.dir/slam2d_linear.cpp.i"
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/solvers/slam2d_linear && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/g2o/solvers/slam2d_linear/slam2d_linear.cpp > CMakeFiles/solver_slam2d_linear.dir/slam2d_linear.cpp.i

g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/slam2d_linear.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/solver_slam2d_linear.dir/slam2d_linear.cpp.s"
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/solvers/slam2d_linear && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/g2o/solvers/slam2d_linear/slam2d_linear.cpp -o CMakeFiles/solver_slam2d_linear.dir/slam2d_linear.cpp.s

g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/slam2d_linear.cpp.o.requires:

.PHONY : g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/slam2d_linear.cpp.o.requires

g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/slam2d_linear.cpp.o.provides: g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/slam2d_linear.cpp.o.requires
	$(MAKE) -f g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/build.make g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/slam2d_linear.cpp.o.provides.build
.PHONY : g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/slam2d_linear.cpp.o.provides

g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/slam2d_linear.cpp.o.provides.build: g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/slam2d_linear.cpp.o


g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/solver_slam2d_linear.cpp.o: g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/flags.make
g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/solver_slam2d_linear.cpp.o: ../g2o/solvers/slam2d_linear/solver_slam2d_linear.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/solver_slam2d_linear.cpp.o"
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/solvers/slam2d_linear && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/solver_slam2d_linear.dir/solver_slam2d_linear.cpp.o -c /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/g2o/solvers/slam2d_linear/solver_slam2d_linear.cpp

g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/solver_slam2d_linear.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/solver_slam2d_linear.dir/solver_slam2d_linear.cpp.i"
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/solvers/slam2d_linear && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/g2o/solvers/slam2d_linear/solver_slam2d_linear.cpp > CMakeFiles/solver_slam2d_linear.dir/solver_slam2d_linear.cpp.i

g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/solver_slam2d_linear.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/solver_slam2d_linear.dir/solver_slam2d_linear.cpp.s"
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/solvers/slam2d_linear && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/g2o/solvers/slam2d_linear/solver_slam2d_linear.cpp -o CMakeFiles/solver_slam2d_linear.dir/solver_slam2d_linear.cpp.s

g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/solver_slam2d_linear.cpp.o.requires:

.PHONY : g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/solver_slam2d_linear.cpp.o.requires

g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/solver_slam2d_linear.cpp.o.provides: g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/solver_slam2d_linear.cpp.o.requires
	$(MAKE) -f g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/build.make g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/solver_slam2d_linear.cpp.o.provides.build
.PHONY : g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/solver_slam2d_linear.cpp.o.provides

g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/solver_slam2d_linear.cpp.o.provides.build: g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/solver_slam2d_linear.cpp.o


# Object files for target solver_slam2d_linear
solver_slam2d_linear_OBJECTS = \
"CMakeFiles/solver_slam2d_linear.dir/slam2d_linear.cpp.o" \
"CMakeFiles/solver_slam2d_linear.dir/solver_slam2d_linear.cpp.o"

# External object files for target solver_slam2d_linear
solver_slam2d_linear_EXTERNAL_OBJECTS =

../lib/libg2o_solver_slam2d_linear.so: g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/slam2d_linear.cpp.o
../lib/libg2o_solver_slam2d_linear.so: g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/solver_slam2d_linear.cpp.o
../lib/libg2o_solver_slam2d_linear.so: g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/build.make
../lib/libg2o_solver_slam2d_linear.so: ../lib/libg2o_solver_csparse.so
../lib/libg2o_solver_slam2d_linear.so: ../lib/libg2o_types_slam2d.so
../lib/libg2o_solver_slam2d_linear.so: ../lib/libg2o_csparse_extension.so
../lib/libg2o_solver_slam2d_linear.so: /usr/lib/aarch64-linux-gnu/libcxsparse.so
../lib/libg2o_solver_slam2d_linear.so: ../lib/libg2o_core.so
../lib/libg2o_solver_slam2d_linear.so: ../lib/libg2o_stuff.so
../lib/libg2o_solver_slam2d_linear.so: ../lib/libg2o_opengl_helper.so
../lib/libg2o_solver_slam2d_linear.so: /usr/lib/aarch64-linux-gnu/libGLU.so
../lib/libg2o_solver_slam2d_linear.so: /usr/lib/aarch64-linux-gnu/libGL.so
../lib/libg2o_solver_slam2d_linear.so: g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX shared library ../../../../lib/libg2o_solver_slam2d_linear.so"
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/solvers/slam2d_linear && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/solver_slam2d_linear.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/build: ../lib/libg2o_solver_slam2d_linear.so

.PHONY : g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/build

g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/requires: g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/slam2d_linear.cpp.o.requires
g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/requires: g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/solver_slam2d_linear.cpp.o.requires

.PHONY : g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/requires

g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/clean:
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/solvers/slam2d_linear && $(CMAKE_COMMAND) -P CMakeFiles/solver_slam2d_linear.dir/cmake_clean.cmake
.PHONY : g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/clean

g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/depend:
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2 /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/g2o/solvers/slam2d_linear /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/solvers/slam2d_linear /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : g2o/solvers/slam2d_linear/CMakeFiles/solver_slam2d_linear.dir/depend

