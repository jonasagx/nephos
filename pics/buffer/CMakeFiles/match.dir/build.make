# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.1

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
CMAKE_SOURCE_DIR = /home/guto/apps/cloudHunter/pics/buffer

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/guto/apps/cloudHunter/pics/buffer

# Include any dependencies generated for this target.
include CMakeFiles/match.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/match.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/match.dir/flags.make

CMakeFiles/match.dir/match.cpp.o: CMakeFiles/match.dir/flags.make
CMakeFiles/match.dir/match.cpp.o: match.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/guto/apps/cloudHunter/pics/buffer/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/match.dir/match.cpp.o"
	/usr/sbin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/match.dir/match.cpp.o -c /home/guto/apps/cloudHunter/pics/buffer/match.cpp

CMakeFiles/match.dir/match.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/match.dir/match.cpp.i"
	/usr/sbin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/guto/apps/cloudHunter/pics/buffer/match.cpp > CMakeFiles/match.dir/match.cpp.i

CMakeFiles/match.dir/match.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/match.dir/match.cpp.s"
	/usr/sbin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/guto/apps/cloudHunter/pics/buffer/match.cpp -o CMakeFiles/match.dir/match.cpp.s

CMakeFiles/match.dir/match.cpp.o.requires:
.PHONY : CMakeFiles/match.dir/match.cpp.o.requires

CMakeFiles/match.dir/match.cpp.o.provides: CMakeFiles/match.dir/match.cpp.o.requires
	$(MAKE) -f CMakeFiles/match.dir/build.make CMakeFiles/match.dir/match.cpp.o.provides.build
.PHONY : CMakeFiles/match.dir/match.cpp.o.provides

CMakeFiles/match.dir/match.cpp.o.provides.build: CMakeFiles/match.dir/match.cpp.o

# Object files for target match
match_OBJECTS = \
"CMakeFiles/match.dir/match.cpp.o"

# External object files for target match
match_EXTERNAL_OBJECTS =

match: CMakeFiles/match.dir/match.cpp.o
match: CMakeFiles/match.dir/build.make
match: /usr/local/lib/libopencv_xphoto.so.3.0.0
match: /usr/local/lib/libopencv_ximgproc.so.3.0.0
match: /usr/local/lib/libopencv_tracking.so.3.0.0
match: /usr/local/lib/libopencv_surface_matching.so.3.0.0
match: /usr/local/lib/libopencv_saliency.so.3.0.0
match: /usr/local/lib/libopencv_rgbd.so.3.0.0
match: /usr/local/lib/libopencv_reg.so.3.0.0
match: /usr/local/lib/libopencv_optflow.so.3.0.0
match: /usr/local/lib/libopencv_line_descriptor.so.3.0.0
match: /usr/local/lib/libopencv_latentsvm.so.3.0.0
match: /usr/local/lib/libopencv_datasets.so.3.0.0
match: /usr/local/lib/libopencv_ccalib.so.3.0.0
match: /usr/local/lib/libopencv_bioinspired.so.3.0.0
match: /usr/local/lib/libopencv_bgsegm.so.3.0.0
match: /usr/local/lib/libopencv_adas.so.3.0.0
match: /usr/local/lib/libopencv_videostab.so.3.0.0
match: /usr/local/lib/libopencv_ts.a
match: /usr/local/lib/libopencv_superres.so.3.0.0
match: /usr/local/lib/libopencv_stitching.so.3.0.0
match: /usr/local/lib/libopencv_photo.so.3.0.0
match: /usr/local/lib/libopencv_objdetect.so.3.0.0
match: /usr/local/lib/libopencv_text.so.3.0.0
match: /usr/local/lib/libopencv_face.so.3.0.0
match: /usr/local/lib/libopencv_xobjdetect.so.3.0.0
match: /usr/local/lib/libopencv_xfeatures2d.so.3.0.0
match: /usr/local/lib/libopencv_shape.so.3.0.0
match: /usr/local/lib/libopencv_video.so.3.0.0
match: /usr/local/lib/libopencv_calib3d.so.3.0.0
match: /usr/local/lib/libopencv_features2d.so.3.0.0
match: /usr/local/lib/libopencv_ml.so.3.0.0
match: /usr/local/lib/libopencv_highgui.so.3.0.0
match: /usr/local/lib/libopencv_videoio.so.3.0.0
match: /usr/local/lib/libopencv_imgcodecs.so.3.0.0
match: /usr/local/lib/libopencv_imgproc.so.3.0.0
match: /usr/local/lib/libopencv_flann.so.3.0.0
match: /usr/local/lib/libopencv_core.so.3.0.0
match: CMakeFiles/match.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable match"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/match.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/match.dir/build: match
.PHONY : CMakeFiles/match.dir/build

CMakeFiles/match.dir/requires: CMakeFiles/match.dir/match.cpp.o.requires
.PHONY : CMakeFiles/match.dir/requires

CMakeFiles/match.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/match.dir/cmake_clean.cmake
.PHONY : CMakeFiles/match.dir/clean

CMakeFiles/match.dir/depend:
	cd /home/guto/apps/cloudHunter/pics/buffer && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/guto/apps/cloudHunter/pics/buffer /home/guto/apps/cloudHunter/pics/buffer /home/guto/apps/cloudHunter/pics/buffer /home/guto/apps/cloudHunter/pics/buffer /home/guto/apps/cloudHunter/pics/buffer/CMakeFiles/match.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/match.dir/depend
