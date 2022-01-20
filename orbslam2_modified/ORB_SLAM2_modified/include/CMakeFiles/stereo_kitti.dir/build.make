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
CMAKE_SOURCE_DIR = /home/simon/catkin_ws/src/orbslam2_modified/ORB_SLAM2_modified

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/simon/catkin_ws/src/orbslam2_modified/ORB_SLAM2_modified/include

# Include any dependencies generated for this target.
include CMakeFiles/stereo_kitti.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/stereo_kitti.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/stereo_kitti.dir/flags.make

CMakeFiles/stereo_kitti.dir/Examples/Stereo/stereo_kitti.cc.o: CMakeFiles/stereo_kitti.dir/flags.make
CMakeFiles/stereo_kitti.dir/Examples/Stereo/stereo_kitti.cc.o: ../Examples/Stereo/stereo_kitti.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/simon/catkin_ws/src/orbslam2_modified/ORB_SLAM2_modified/include/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/stereo_kitti.dir/Examples/Stereo/stereo_kitti.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/stereo_kitti.dir/Examples/Stereo/stereo_kitti.cc.o -c /home/simon/catkin_ws/src/orbslam2_modified/ORB_SLAM2_modified/Examples/Stereo/stereo_kitti.cc

CMakeFiles/stereo_kitti.dir/Examples/Stereo/stereo_kitti.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stereo_kitti.dir/Examples/Stereo/stereo_kitti.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/simon/catkin_ws/src/orbslam2_modified/ORB_SLAM2_modified/Examples/Stereo/stereo_kitti.cc > CMakeFiles/stereo_kitti.dir/Examples/Stereo/stereo_kitti.cc.i

CMakeFiles/stereo_kitti.dir/Examples/Stereo/stereo_kitti.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stereo_kitti.dir/Examples/Stereo/stereo_kitti.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/simon/catkin_ws/src/orbslam2_modified/ORB_SLAM2_modified/Examples/Stereo/stereo_kitti.cc -o CMakeFiles/stereo_kitti.dir/Examples/Stereo/stereo_kitti.cc.s

CMakeFiles/stereo_kitti.dir/Examples/Stereo/stereo_kitti.cc.o.requires:

.PHONY : CMakeFiles/stereo_kitti.dir/Examples/Stereo/stereo_kitti.cc.o.requires

CMakeFiles/stereo_kitti.dir/Examples/Stereo/stereo_kitti.cc.o.provides: CMakeFiles/stereo_kitti.dir/Examples/Stereo/stereo_kitti.cc.o.requires
	$(MAKE) -f CMakeFiles/stereo_kitti.dir/build.make CMakeFiles/stereo_kitti.dir/Examples/Stereo/stereo_kitti.cc.o.provides.build
.PHONY : CMakeFiles/stereo_kitti.dir/Examples/Stereo/stereo_kitti.cc.o.provides

CMakeFiles/stereo_kitti.dir/Examples/Stereo/stereo_kitti.cc.o.provides.build: CMakeFiles/stereo_kitti.dir/Examples/Stereo/stereo_kitti.cc.o


# Object files for target stereo_kitti
stereo_kitti_OBJECTS = \
"CMakeFiles/stereo_kitti.dir/Examples/Stereo/stereo_kitti.cc.o"

# External object files for target stereo_kitti
stereo_kitti_EXTERNAL_OBJECTS =

../Examples/Stereo/stereo_kitti: CMakeFiles/stereo_kitti.dir/Examples/Stereo/stereo_kitti.cc.o
../Examples/Stereo/stereo_kitti: CMakeFiles/stereo_kitti.dir/build.make
../Examples/Stereo/stereo_kitti: ../lib/libORB_SLAM2.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libopencv_dnn.so.4.1.1
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libopencv_gapi.so.4.1.1
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libopencv_highgui.so.4.1.1
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libopencv_ml.so.4.1.1
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libopencv_objdetect.so.4.1.1
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libopencv_photo.so.4.1.1
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libopencv_stitching.so.4.1.1
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libopencv_video.so.4.1.1
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libopencv_calib3d.so.4.1.1
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libopencv_features2d.so.4.1.1
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libopencv_flann.so.4.1.1
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libopencv_videoio.so.4.1.1
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.1.1
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.1.1
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libopencv_core.so.4.1.1
../Examples/Stereo/stereo_kitti: /home/simon/Pangolin-master/build/libpango_glgeometry.so
../Examples/Stereo/stereo_kitti: /home/simon/Pangolin-master/build/libpango_geometry.so
../Examples/Stereo/stereo_kitti: /home/simon/Pangolin-master/build/libpango_plot.so
../Examples/Stereo/stereo_kitti: /home/simon/Pangolin-master/build/libpango_python.so
../Examples/Stereo/stereo_kitti: /home/simon/Pangolin-master/build/libpango_scene.so
../Examples/Stereo/stereo_kitti: /home/simon/Pangolin-master/build/libpango_tools.so
../Examples/Stereo/stereo_kitti: /home/simon/Pangolin-master/build/libpango_display.so
../Examples/Stereo/stereo_kitti: /home/simon/Pangolin-master/build/libpango_vars.so
../Examples/Stereo/stereo_kitti: /home/simon/Pangolin-master/build/libpango_video.so
../Examples/Stereo/stereo_kitti: /home/simon/Pangolin-master/build/libpango_packetstream.so
../Examples/Stereo/stereo_kitti: /home/simon/Pangolin-master/build/libpango_windowing.so
../Examples/Stereo/stereo_kitti: /home/simon/Pangolin-master/build/libpango_opengl.so
../Examples/Stereo/stereo_kitti: /home/simon/Pangolin-master/build/libpango_image.so
../Examples/Stereo/stereo_kitti: /home/simon/Pangolin-master/build/libpango_core.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libGLEW.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libOpenGL.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libGLX.so
../Examples/Stereo/stereo_kitti: /home/simon/Pangolin-master/build/libtinyobj.so
../Examples/Stereo/stereo_kitti: ../Thirdparty/DBoW2/lib/libDBoW2.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkDomainsChemistry-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkFiltersGeneric-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkFiltersHyperTree-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkFiltersParallelFlowPaths-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkFiltersFlowPaths-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkFiltersParallelGeometry-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkFiltersParallelImaging-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkFiltersParallelMPI-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkFiltersParallelStatistics-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkFiltersProgrammable-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkFiltersPython-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/libvtkWrappingTools-6.3.a
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkFiltersReebGraph-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkFiltersSMP-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkFiltersSelection-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkFiltersVerdict-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkverdict-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkGUISupportQtOpenGL-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkGUISupportQtSQL-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkGUISupportQtWebkit-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkViewsQt-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkIOAMR-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkIOEnSight-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkIOExport-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkRenderingGL2PS-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkRenderingContextOpenGL-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkIOFFMPEG-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkIOMovie-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkIOGDAL-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkIOGeoJSON-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkIOImport-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkIOInfovis-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkIOMINC-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkIOMPIImage-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkIOMPIParallel-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkIOParallel-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkIONetCDF-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkIOMySQL-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkIOODBC-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkIOPLY-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkIOParallelExodus-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkIOExodus-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkexoIIc-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkIOParallelLSDyna-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkIOLSDyna-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkIOParallelNetCDF-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkIOParallelXML-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkIOPostgreSQL-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkIOVPIC-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkVPIC-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkIOVideo-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkIOXdmf2-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkxdmf2-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkImagingMath-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkImagingMorphological-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkImagingStatistics-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkImagingStencil-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkInteractionImage-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkLocalExample-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkParallelMPI4Py-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkRenderingExternal-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkRenderingFreeTypeFontConfig-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkRenderingImage-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkRenderingLOD-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkRenderingMatplotlib-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkWrappingPython27Core-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkPythonInterpreter-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkRenderingParallel-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkFiltersParallel-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkRenderingParallelLIC-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkParallelMPI-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkRenderingLIC-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkRenderingQt-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkFiltersTexture-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkGUISupportQt-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libQt5Widgets.so.5.9.5
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libQt5Gui.so.5.9.5
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libQt5Core.so.5.9.5
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkRenderingVolumeAMR-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkFiltersAMR-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkParallelCore-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkIOLegacy-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkRenderingVolumeOpenGL-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkRenderingOpenGL-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libGLU.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libSM.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libICE.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libX11.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libXext.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libXt.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkTestingGenericBridge-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkTestingIOSQL-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkIOSQL-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkTestingRendering-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkViewsContext2D-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkViewsGeovis-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkViewsInfovis-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkChartsCore-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkRenderingContext2D-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkFiltersImaging-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkRenderingLabel-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkGeovisCore-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkIOXML-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkIOGeometry-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkIOXMLParser-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkInfovisLayout-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkInfovisBoostGraphAlgorithms-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkInfovisCore-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkViewsCore-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkInteractionWidgets-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkFiltersHybrid-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkImagingGeneral-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkImagingSources-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkFiltersModeling-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkInteractionStyle-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkImagingHybrid-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkIOImage-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkDICOMParser-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkIOCore-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkmetaio-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkRenderingAnnotation-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkRenderingFreeType-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkftgl-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libGL.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkImagingColor-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkRenderingVolume-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkRenderingCore-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkCommonColor-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkFiltersExtraction-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkFiltersStatistics-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkImagingFourier-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkImagingCore-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkalglib-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkFiltersGeometry-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkFiltersSources-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkFiltersGeneral-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkFiltersCore-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkCommonExecutionModel-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkCommonComputationalGeometry-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkCommonDataModel-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkCommonMisc-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkCommonTransforms-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkCommonMath-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkCommonSystem-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtksys-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkWrappingJava-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libvtkCommonCore-6.3.so.6.3.0
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libboost_system.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libboost_filesystem.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libboost_thread.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libboost_date_time.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libboost_iostreams.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libboost_serialization.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libboost_chrono.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libboost_atomic.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libboost_regex.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpthread.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_common.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_octree.so
../Examples/Stereo/stereo_kitti: /usr/lib/libOpenNI.so
../Examples/Stereo/stereo_kitti: /usr/lib/libOpenNI2.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libexpat.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libjpeg.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpng.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libtiff.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libgl2ps.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libjsoncpp.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_io.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libflann_cpp_s.a
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_kdtree.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_search.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_sample_consensus.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_filters.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_features.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_ml.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_segmentation.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_visualization.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libqhull.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_surface.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_registration.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_keypoints.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_tracking.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_recognition.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_stereo.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_apps.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_outofcore.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_people.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libboost_system.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libboost_filesystem.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libboost_thread.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libboost_date_time.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libboost_iostreams.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libboost_serialization.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libboost_chrono.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libboost_atomic.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libboost_regex.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpthread.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_common.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_octree.so
../Examples/Stereo/stereo_kitti: /usr/lib/libOpenNI.so
../Examples/Stereo/stereo_kitti: /usr/lib/libOpenNI2.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libexpat.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libjpeg.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpng.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libtiff.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libgl2ps.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libjsoncpp.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_io.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libflann_cpp_s.a
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_kdtree.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_search.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_sample_consensus.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_filters.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_features.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_ml.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_segmentation.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_visualization.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libqhull.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_surface.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_registration.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_keypoints.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_tracking.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_recognition.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_stereo.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_apps.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_outofcore.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpcl_people.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libfreetype.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libpython2.7.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libproj.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libnetcdf_c++.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libnetcdf.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libtheoraenc.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libtheoradec.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libogg.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libxml2.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/hdf5/openmpi/libhdf5.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libsz.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libz.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libdl.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/libm.so
../Examples/Stereo/stereo_kitti: /usr/lib/aarch64-linux-gnu/openmpi/lib/libmpi.so
../Examples/Stereo/stereo_kitti: CMakeFiles/stereo_kitti.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/simon/catkin_ws/src/orbslam2_modified/ORB_SLAM2_modified/include/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../Examples/Stereo/stereo_kitti"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/stereo_kitti.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/stereo_kitti.dir/build: ../Examples/Stereo/stereo_kitti

.PHONY : CMakeFiles/stereo_kitti.dir/build

CMakeFiles/stereo_kitti.dir/requires: CMakeFiles/stereo_kitti.dir/Examples/Stereo/stereo_kitti.cc.o.requires

.PHONY : CMakeFiles/stereo_kitti.dir/requires

CMakeFiles/stereo_kitti.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/stereo_kitti.dir/cmake_clean.cmake
.PHONY : CMakeFiles/stereo_kitti.dir/clean

CMakeFiles/stereo_kitti.dir/depend:
	cd /home/simon/catkin_ws/src/orbslam2_modified/ORB_SLAM2_modified/include && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/simon/catkin_ws/src/orbslam2_modified/ORB_SLAM2_modified /home/simon/catkin_ws/src/orbslam2_modified/ORB_SLAM2_modified /home/simon/catkin_ws/src/orbslam2_modified/ORB_SLAM2_modified/include /home/simon/catkin_ws/src/orbslam2_modified/ORB_SLAM2_modified/include /home/simon/catkin_ws/src/orbslam2_modified/ORB_SLAM2_modified/include/CMakeFiles/stereo_kitti.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/stereo_kitti.dir/depend

