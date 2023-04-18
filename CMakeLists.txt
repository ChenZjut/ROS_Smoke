cmake_minimum_required(VERSION 3.0.2)
project(smoke)

## Compile as C++11, supported in ROS Kinetic and newer
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  image_transport
  rospy
  sensor_msgs
  std_msgs
  nodelet
)

find_package(OpenCV REQUIRED)

# set flags for CUDA availability
option(CUDA_AVAIL "CUDA available" OFF)
find_package(CUDA)
if (CUDA_FOUND)
    find_library(CUBLAS_LIBRARIES cublas HINTS
            ${CUDA_TOOLKIT_ROOT_DIR}/lib64
            ${CUDA_TOOLKIT_ROOT_DIR}/lib
            )
    if (CUDA_VERBOSE)
        message(STATUS "CUDA is available!")
        message(STATUS "CUDA Libs: ${CUDA_LIBRARIES}")
        message(STATUS "CUDA Headers: ${CUDA_INCLUDE_DIRS}")
    endif ()
    set(CUDA_AVAIL ON)
else()
    message("CUDA NOT FOUND")
    set(CUDA_AVAIL OFF)
endif (CUDA_FOUND)

# set flags for TensorRT availability
option(TRT_AVAIL "TensorRT available" OFF)
# try to find the tensorRT modules
find_library(NVINFER NAMES nvinfer)
find_library(NVONNXPARSER nvonnxparser)
find_library(NVINFER_PLUGIN NAMES nvinfer_plugin)
if(NVINFER AND NVONNXPARSER AND NVINFER_PLUGIN)
    if (CUDA_VERBOSE)
        message(STATUS "TensorRT is available!")
        message(STATUS "NVINFER: ${NVINFER}")
        message(STATUS "NVPARSERS: ${NVPARSERS}")
        message(STATUS "NVINFER_PLUGIN: ${NVINFER_PLUGIN}")
        message(STATUS "NVONNXPARSER: ${NVONNXPARSER}")
    endif ()
    set(TRT_AVAIL ON)
else()
    message("TensorRT is NOT Available")
    set(TRT_AVAIL OFF)
endif()

# set flags for CUDNN availability
option(CUDNN_AVAIL "CUDNN available" OFF)
# try to find the CUDNN module
find_library(CUDNN_LIBRARY
        NAMES libcudnn.so${__cudnn_ver_suffix} libcudnn${__cudnn_ver_suffix}.dylib ${__cudnn_lib_win_name}
        PATHS $ENV{LD_LIBRARY_PATH} ${__libpath_cudart} ${CUDNN_ROOT_DIR} ${PC_CUDNN_LIBRARY_DIRS} ${CMAKE_INSTALL_PREFIX}
        PATH_SUFFIXES lib lib64 bin
        DOC "CUDNN library." )
set(CUDNN_LIBRARY "/usr/local/cuda/lib64/libcudnn.so")
if(CUDNN_LIBRARY)
    if (CUDA_VERBOSE)
        message(STATUS "CUDNN is available!")
        message(STATUS "CUDNN_LIBRARY: ${CUDNN_LIBRARY}")
    endif ()
    set(CUDNN_AVAIL ON)
else()
    message("CUDNN is NOT Available")
    set(CUDNN_AVAIL OFF)
endif()

if(TRT_AVAIL AND CUDA_AVAIL AND CUDNN_AVAIL)
    catkin_package(
      INCLUDE_DIRS include
      LIBRARIES ${PROJECT_NAME}
      CATKIN_DEPENDS
    )

    include_directories(
      include
      lib/include
      lib/plugins/modulated_deform_conv
      ${OpenCV_INCLUDE_DIRS}
      ${CUDA_INCLUDE_DIRS}
      ${catkin_INCLUDE_DIRS}
    )

    cuda_add_library(trt_modulated_deform_conv_plugin SHARED
      lib/plugins/modulated_deform_conv/trt_modulated_deform_conv_kernel.cu
      lib/plugins/modulated_deform_conv/trt_modulated_deform_conv.cpp
    )

    ### TensorRT lib ###
    add_library(trt_lib
            lib/src/smoke.cpp
    )

    target_link_libraries(trt_lib
      ${NVINFER}
      ${NVONNXPARSER}
      ${NVINFER_PLUGIN}
      ${CUDA_LIBRARIES}
      ${CUBLAS_LIBRARIES}
      ${CUDA_curand_LIBRARY}
      ${CUDNN_LIBRARY}
      trt_modulated_deform_conv_plugin
    )

    add_library(smoke_nodelet SHARED
      src/nodelet.cpp
    )

    add_dependencies(smoke_nodelet
      ${catkin_EXPORTED_TARGETS}
    )

    target_link_libraries(smoke_nodelet
      ${catkin_LIBRARIES}
      ${OpenCV_LIBS}
      trt_lib
      trt_modulated_deform_conv_plugin
    )

    add_executable(smoke_node
      src/node.cpp
    )

    target_link_libraries(smoke_node
      smoke_nodelet
      ${catkin_LIBRARIES}
      ${OpenCV_LIBS}
      trt_lib
      trt_modulated_deform_conv_plugin
    )
    add_dependencies(smoke_node
      ${catkin_EXPORTED_TARGETS}
    )

else()
  find_package(catkin REQUIRED)
  catkin_package()
  message("TensorrtYolo won't be built, CUDA and/or TensorRT were not found.")

  install(
        DIRECTORY
        launch
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  )

endif()
