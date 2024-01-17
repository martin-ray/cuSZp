# Specify the minimum version of CMake required to build the project
cmake_minimum_required(VERSION 3.21)

project(cuSZp
        VERSION 0.0.2
        DESCRIPTION "Error-bounded GPU lossy compression library"
        )
set(namespace "cuSZp")
enable_language(CXX)
enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CUSZP_ENABLE_PYTHON "Enable Python bindings." OFF)

set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_STANDARD "17")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 75)
set(CUDA_PROPAGATE_HOST_FLAGS ON)
set(CUDA_LIBRARY CUDA::cudart)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY VALUE Release)
endif()

# add_library(${PROJECT_NAME} STATIC)
add_library(${PROJECT_NAME} SHARED)

target_sources(${PROJECT_NAME}
        PRIVATE
        src/cuSZp_f32.cu
        src/cuSZp_f64.cu
        src/cuSZp_utility.cu
        src/cuSZp_timer.cu
        src/cuSZp_entry_f32.cu
        src/cuSZp_entry_f64.cu
        )

target_include_directories(${PROJECT_NAME}
        PRIVATE
        # where the library itself will look for its internal headers
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        PUBLIC
        # where top-level project will look for the library's public headers
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        # where external projects will look for the library's public headers
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        )

#target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)

set(public_headers
        include/cuSZp_f32.h
        include/cuSZp_f64.h
        include/cuSZp_utility.h
        include/cuSZp_timer.h
        include/cuSZp_entry_f32.h
        include/cuSZp_entry_f64.h
        )

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Installing)

if(CUSZP_ENABLE_PYTHON)
  find_package(Python COMPONENTS Interpreter Development.Module  REQUIRED) # よく出てくる。ライブラリがあるかを確認。cmakeがいい感じにいろいろやってくれる
  find_package(CUDAToolkit REQUIRED)
  enable_language(CUDA)
  set(CUDA_LIBRARY CUDA::cudart)
  
  include(FetchContent) # 便利
  FetchContent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG        stable
  )
  FetchContent_MakeAvailable(pybind11)

  pybind11_add_module(_cuSZp src/python/bindings.cpp) # ここで作成するライブラリを指定。add_libraryてきなののpybind版
  target_include_directories(_cuSZp PRIVATE "/usr/local/cuda/include")
  target_include_directories(${PROJECT_NAME}
        PRIVATE
        # where the library itself will look for its internal headers
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        PUBLIC
        # where top-level project will look for the library's public headers
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        # where external projects will look for the library's public headers
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        )
  target_link_libraries(_cuSZp PRIVATE cuSZp CUDA::cudart)
  install(TARGETS _cuSZp LIBRARY DESTINATION python/cuSZp)
endif()

option(CUSZP_BUILD_EXAMPLES "Option to enable building example programs" ON)
if (CUSZP_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()