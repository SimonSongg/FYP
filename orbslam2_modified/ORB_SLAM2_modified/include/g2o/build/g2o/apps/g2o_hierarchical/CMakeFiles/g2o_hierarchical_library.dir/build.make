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
include g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/depend.make

# Include the progress variables for this target.
include g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/progress.make

# Include the compile flags for this target's objects.
include g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/flags.make

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_labeler.cpp.o: g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/flags.make
g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_labeler.cpp.o: ../g2o/apps/g2o_hierarchical/edge_labeler.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_labeler.cpp.o"
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/apps/g2o_hierarchical && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/g2o_hierarchical_library.dir/edge_labeler.cpp.o -c /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/g2o/apps/g2o_hierarchical/edge_labeler.cpp

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_labeler.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/g2o_hierarchical_library.dir/edge_labeler.cpp.i"
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/apps/g2o_hierarchical && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/g2o/apps/g2o_hierarchical/edge_labeler.cpp > CMakeFiles/g2o_hierarchical_library.dir/edge_labeler.cpp.i

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_labeler.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/g2o_hierarchical_library.dir/edge_labeler.cpp.s"
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/apps/g2o_hierarchical && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/g2o/apps/g2o_hierarchical/edge_labeler.cpp -o CMakeFiles/g2o_hierarchical_library.dir/edge_labeler.cpp.s

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_labeler.cpp.o.requires:

.PHONY : g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_labeler.cpp.o.requires

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_labeler.cpp.o.provides: g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_labeler.cpp.o.requires
	$(MAKE) -f g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/build.make g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_labeler.cpp.o.provides.build
.PHONY : g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_labeler.cpp.o.provides

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_labeler.cpp.o.provides.build: g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_labeler.cpp.o


g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_creator.cpp.o: g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/flags.make
g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_creator.cpp.o: ../g2o/apps/g2o_hierarchical/edge_creator.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_creator.cpp.o"
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/apps/g2o_hierarchical && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/g2o_hierarchical_library.dir/edge_creator.cpp.o -c /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/g2o/apps/g2o_hierarchical/edge_creator.cpp

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_creator.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/g2o_hierarchical_library.dir/edge_creator.cpp.i"
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/apps/g2o_hierarchical && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/g2o/apps/g2o_hierarchical/edge_creator.cpp > CMakeFiles/g2o_hierarchical_library.dir/edge_creator.cpp.i

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_creator.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/g2o_hierarchical_library.dir/edge_creator.cpp.s"
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/apps/g2o_hierarchical && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/g2o/apps/g2o_hierarchical/edge_creator.cpp -o CMakeFiles/g2o_hierarchical_library.dir/edge_creator.cpp.s

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_creator.cpp.o.requires:

.PHONY : g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_creator.cpp.o.requires

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_creator.cpp.o.provides: g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_creator.cpp.o.requires
	$(MAKE) -f g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/build.make g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_creator.cpp.o.provides.build
.PHONY : g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_creator.cpp.o.provides

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_creator.cpp.o.provides.build: g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_creator.cpp.o


g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/star.cpp.o: g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/flags.make
g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/star.cpp.o: ../g2o/apps/g2o_hierarchical/star.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/star.cpp.o"
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/apps/g2o_hierarchical && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/g2o_hierarchical_library.dir/star.cpp.o -c /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/g2o/apps/g2o_hierarchical/star.cpp

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/star.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/g2o_hierarchical_library.dir/star.cpp.i"
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/apps/g2o_hierarchical && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/g2o/apps/g2o_hierarchical/star.cpp > CMakeFiles/g2o_hierarchical_library.dir/star.cpp.i

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/star.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/g2o_hierarchical_library.dir/star.cpp.s"
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/apps/g2o_hierarchical && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/g2o/apps/g2o_hierarchical/star.cpp -o CMakeFiles/g2o_hierarchical_library.dir/star.cpp.s

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/star.cpp.o.requires:

.PHONY : g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/star.cpp.o.requires

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/star.cpp.o.provides: g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/star.cpp.o.requires
	$(MAKE) -f g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/build.make g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/star.cpp.o.provides.build
.PHONY : g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/star.cpp.o.provides

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/star.cpp.o.provides.build: g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/star.cpp.o


g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_types_cost_function.cpp.o: g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/flags.make
g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_types_cost_function.cpp.o: ../g2o/apps/g2o_hierarchical/edge_types_cost_function.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_types_cost_function.cpp.o"
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/apps/g2o_hierarchical && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/g2o_hierarchical_library.dir/edge_types_cost_function.cpp.o -c /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/g2o/apps/g2o_hierarchical/edge_types_cost_function.cpp

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_types_cost_function.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/g2o_hierarchical_library.dir/edge_types_cost_function.cpp.i"
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/apps/g2o_hierarchical && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/g2o/apps/g2o_hierarchical/edge_types_cost_function.cpp > CMakeFiles/g2o_hierarchical_library.dir/edge_types_cost_function.cpp.i

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_types_cost_function.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/g2o_hierarchical_library.dir/edge_types_cost_function.cpp.s"
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/apps/g2o_hierarchical && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/g2o/apps/g2o_hierarchical/edge_types_cost_function.cpp -o CMakeFiles/g2o_hierarchical_library.dir/edge_types_cost_function.cpp.s

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_types_cost_function.cpp.o.requires:

.PHONY : g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_types_cost_function.cpp.o.requires

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_types_cost_function.cpp.o.provides: g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_types_cost_function.cpp.o.requires
	$(MAKE) -f g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/build.make g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_types_cost_function.cpp.o.provides.build
.PHONY : g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_types_cost_function.cpp.o.provides

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_types_cost_function.cpp.o.provides.build: g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_types_cost_function.cpp.o


g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/backbone_tree_action.cpp.o: g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/flags.make
g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/backbone_tree_action.cpp.o: ../g2o/apps/g2o_hierarchical/backbone_tree_action.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/backbone_tree_action.cpp.o"
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/apps/g2o_hierarchical && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/g2o_hierarchical_library.dir/backbone_tree_action.cpp.o -c /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/g2o/apps/g2o_hierarchical/backbone_tree_action.cpp

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/backbone_tree_action.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/g2o_hierarchical_library.dir/backbone_tree_action.cpp.i"
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/apps/g2o_hierarchical && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/g2o/apps/g2o_hierarchical/backbone_tree_action.cpp > CMakeFiles/g2o_hierarchical_library.dir/backbone_tree_action.cpp.i

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/backbone_tree_action.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/g2o_hierarchical_library.dir/backbone_tree_action.cpp.s"
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/apps/g2o_hierarchical && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/g2o/apps/g2o_hierarchical/backbone_tree_action.cpp -o CMakeFiles/g2o_hierarchical_library.dir/backbone_tree_action.cpp.s

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/backbone_tree_action.cpp.o.requires:

.PHONY : g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/backbone_tree_action.cpp.o.requires

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/backbone_tree_action.cpp.o.provides: g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/backbone_tree_action.cpp.o.requires
	$(MAKE) -f g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/build.make g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/backbone_tree_action.cpp.o.provides.build
.PHONY : g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/backbone_tree_action.cpp.o.provides

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/backbone_tree_action.cpp.o.provides.build: g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/backbone_tree_action.cpp.o


g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/simple_star_ops.cpp.o: g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/flags.make
g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/simple_star_ops.cpp.o: ../g2o/apps/g2o_hierarchical/simple_star_ops.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/simple_star_ops.cpp.o"
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/apps/g2o_hierarchical && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/g2o_hierarchical_library.dir/simple_star_ops.cpp.o -c /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/g2o/apps/g2o_hierarchical/simple_star_ops.cpp

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/simple_star_ops.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/g2o_hierarchical_library.dir/simple_star_ops.cpp.i"
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/apps/g2o_hierarchical && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/g2o/apps/g2o_hierarchical/simple_star_ops.cpp > CMakeFiles/g2o_hierarchical_library.dir/simple_star_ops.cpp.i

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/simple_star_ops.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/g2o_hierarchical_library.dir/simple_star_ops.cpp.s"
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/apps/g2o_hierarchical && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/g2o/apps/g2o_hierarchical/simple_star_ops.cpp -o CMakeFiles/g2o_hierarchical_library.dir/simple_star_ops.cpp.s

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/simple_star_ops.cpp.o.requires:

.PHONY : g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/simple_star_ops.cpp.o.requires

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/simple_star_ops.cpp.o.provides: g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/simple_star_ops.cpp.o.requires
	$(MAKE) -f g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/build.make g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/simple_star_ops.cpp.o.provides.build
.PHONY : g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/simple_star_ops.cpp.o.provides

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/simple_star_ops.cpp.o.provides.build: g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/simple_star_ops.cpp.o


g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/g2o_hierarchical_test_functions.cpp.o: g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/flags.make
g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/g2o_hierarchical_test_functions.cpp.o: ../g2o/apps/g2o_hierarchical/g2o_hierarchical_test_functions.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/g2o_hierarchical_test_functions.cpp.o"
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/apps/g2o_hierarchical && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/g2o_hierarchical_library.dir/g2o_hierarchical_test_functions.cpp.o -c /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/g2o/apps/g2o_hierarchical/g2o_hierarchical_test_functions.cpp

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/g2o_hierarchical_test_functions.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/g2o_hierarchical_library.dir/g2o_hierarchical_test_functions.cpp.i"
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/apps/g2o_hierarchical && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/g2o/apps/g2o_hierarchical/g2o_hierarchical_test_functions.cpp > CMakeFiles/g2o_hierarchical_library.dir/g2o_hierarchical_test_functions.cpp.i

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/g2o_hierarchical_test_functions.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/g2o_hierarchical_library.dir/g2o_hierarchical_test_functions.cpp.s"
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/apps/g2o_hierarchical && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/g2o/apps/g2o_hierarchical/g2o_hierarchical_test_functions.cpp -o CMakeFiles/g2o_hierarchical_library.dir/g2o_hierarchical_test_functions.cpp.s

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/g2o_hierarchical_test_functions.cpp.o.requires:

.PHONY : g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/g2o_hierarchical_test_functions.cpp.o.requires

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/g2o_hierarchical_test_functions.cpp.o.provides: g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/g2o_hierarchical_test_functions.cpp.o.requires
	$(MAKE) -f g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/build.make g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/g2o_hierarchical_test_functions.cpp.o.provides.build
.PHONY : g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/g2o_hierarchical_test_functions.cpp.o.provides

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/g2o_hierarchical_test_functions.cpp.o.provides.build: g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/g2o_hierarchical_test_functions.cpp.o


# Object files for target g2o_hierarchical_library
g2o_hierarchical_library_OBJECTS = \
"CMakeFiles/g2o_hierarchical_library.dir/edge_labeler.cpp.o" \
"CMakeFiles/g2o_hierarchical_library.dir/edge_creator.cpp.o" \
"CMakeFiles/g2o_hierarchical_library.dir/star.cpp.o" \
"CMakeFiles/g2o_hierarchical_library.dir/edge_types_cost_function.cpp.o" \
"CMakeFiles/g2o_hierarchical_library.dir/backbone_tree_action.cpp.o" \
"CMakeFiles/g2o_hierarchical_library.dir/simple_star_ops.cpp.o" \
"CMakeFiles/g2o_hierarchical_library.dir/g2o_hierarchical_test_functions.cpp.o"

# External object files for target g2o_hierarchical_library
g2o_hierarchical_library_EXTERNAL_OBJECTS =

../lib/libg2o_hierarchical.so: g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_labeler.cpp.o
../lib/libg2o_hierarchical.so: g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_creator.cpp.o
../lib/libg2o_hierarchical.so: g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/star.cpp.o
../lib/libg2o_hierarchical.so: g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_types_cost_function.cpp.o
../lib/libg2o_hierarchical.so: g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/backbone_tree_action.cpp.o
../lib/libg2o_hierarchical.so: g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/simple_star_ops.cpp.o
../lib/libg2o_hierarchical.so: g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/g2o_hierarchical_test_functions.cpp.o
../lib/libg2o_hierarchical.so: g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/build.make
../lib/libg2o_hierarchical.so: ../lib/libg2o_core.so
../lib/libg2o_hierarchical.so: ../lib/libg2o_stuff.so
../lib/libg2o_hierarchical.so: g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Linking CXX shared library ../../../../lib/libg2o_hierarchical.so"
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/apps/g2o_hierarchical && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/g2o_hierarchical_library.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/build: ../lib/libg2o_hierarchical.so

.PHONY : g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/build

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/requires: g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_labeler.cpp.o.requires
g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/requires: g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_creator.cpp.o.requires
g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/requires: g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/star.cpp.o.requires
g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/requires: g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/edge_types_cost_function.cpp.o.requires
g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/requires: g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/backbone_tree_action.cpp.o.requires
g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/requires: g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/simple_star_ops.cpp.o.requires
g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/requires: g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/g2o_hierarchical_test_functions.cpp.o.requires

.PHONY : g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/requires

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/clean:
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/apps/g2o_hierarchical && $(CMAKE_COMMAND) -P CMakeFiles/g2o_hierarchical_library.dir/cmake_clean.cmake
.PHONY : g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/clean

g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/depend:
	cd /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2 /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/g2o/apps/g2o_hierarchical /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/apps/g2o_hierarchical /home/simon/catkin_ws/src/orbslam2_modified/g2o_with_orbslam2/build/g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : g2o/apps/g2o_hierarchical/CMakeFiles/g2o_hierarchical_library.dir/depend

