# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/wjc/workspace/file_io

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/wjc/workspace/file_io/build

# Include any dependencies generated for this target.
include CMakeFiles/qcu_io.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/qcu_io.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/qcu_io.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/qcu_io.dir/flags.make

CMakeFiles/qcu_io.dir/src/lqcd_read_write.cu.o: CMakeFiles/qcu_io.dir/flags.make
CMakeFiles/qcu_io.dir/src/lqcd_read_write.cu.o: ../src/lqcd_read_write.cu
CMakeFiles/qcu_io.dir/src/lqcd_read_write.cu.o: CMakeFiles/qcu_io.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wjc/workspace/file_io/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/qcu_io.dir/src/lqcd_read_write.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/qcu_io.dir/src/lqcd_read_write.cu.o -MF CMakeFiles/qcu_io.dir/src/lqcd_read_write.cu.o.d -x cu -c /home/wjc/workspace/file_io/src/lqcd_read_write.cu -o CMakeFiles/qcu_io.dir/src/lqcd_read_write.cu.o

CMakeFiles/qcu_io.dir/src/lqcd_read_write.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/qcu_io.dir/src/lqcd_read_write.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/qcu_io.dir/src/lqcd_read_write.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/qcu_io.dir/src/lqcd_read_write.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/qcu_io.dir/test/test_io.cpp.o: CMakeFiles/qcu_io.dir/flags.make
CMakeFiles/qcu_io.dir/test/test_io.cpp.o: ../test/test_io.cpp
CMakeFiles/qcu_io.dir/test/test_io.cpp.o: CMakeFiles/qcu_io.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wjc/workspace/file_io/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/qcu_io.dir/test/test_io.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/qcu_io.dir/test/test_io.cpp.o -MF CMakeFiles/qcu_io.dir/test/test_io.cpp.o.d -o CMakeFiles/qcu_io.dir/test/test_io.cpp.o -c /home/wjc/workspace/file_io/test/test_io.cpp

CMakeFiles/qcu_io.dir/test/test_io.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/qcu_io.dir/test/test_io.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wjc/workspace/file_io/test/test_io.cpp > CMakeFiles/qcu_io.dir/test/test_io.cpp.i

CMakeFiles/qcu_io.dir/test/test_io.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/qcu_io.dir/test/test_io.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wjc/workspace/file_io/test/test_io.cpp -o CMakeFiles/qcu_io.dir/test/test_io.cpp.s

# Object files for target qcu_io
qcu_io_OBJECTS = \
"CMakeFiles/qcu_io.dir/src/lqcd_read_write.cu.o" \
"CMakeFiles/qcu_io.dir/test/test_io.cpp.o"

# External object files for target qcu_io
qcu_io_EXTERNAL_OBJECTS =

libqcu_io.so: CMakeFiles/qcu_io.dir/src/lqcd_read_write.cu.o
libqcu_io.so: CMakeFiles/qcu_io.dir/test/test_io.cpp.o
libqcu_io.so: CMakeFiles/qcu_io.dir/build.make
libqcu_io.so: CMakeFiles/qcu_io.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/wjc/workspace/file_io/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX shared library libqcu_io.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/qcu_io.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/qcu_io.dir/build: libqcu_io.so
.PHONY : CMakeFiles/qcu_io.dir/build

CMakeFiles/qcu_io.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/qcu_io.dir/cmake_clean.cmake
.PHONY : CMakeFiles/qcu_io.dir/clean

CMakeFiles/qcu_io.dir/depend:
	cd /home/wjc/workspace/file_io/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wjc/workspace/file_io /home/wjc/workspace/file_io /home/wjc/workspace/file_io/build /home/wjc/workspace/file_io/build /home/wjc/workspace/file_io/build/CMakeFiles/qcu_io.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/qcu_io.dir/depend

