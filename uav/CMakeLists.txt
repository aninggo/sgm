cmake_minimum_required (VERSION 2.8.11)

# project name
project(Tulipp_UC_UAV_SGM)

# project version
set(VERSION_MAJOR 1)
set(VERSION_MINOR 0)
set(VERSION_STRING ${VERSION_MAJOR}.${VERSION_MINOR})

# compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")

# set path to custom cmake modules
set(CUSTOM_CMAKE_MODULE_PATH "$ENV{HOME}/cmake" CACHE PATH "Path to custom cmake modules.")
set(CMAKE_MODULE_PATH ${CUSTOM_CMAKE_MODULE_PATH})

# set build output directorie
set(BUILD_OUTPUT_bin ${PROJECT_SOURCE_DIR}/bin CACHE PATH "Output path of binaries")
if(NOT EXISTS ${BUILD_OUTPUT_bin})
  file(MAKE_DIRECTORY ${BUILD_OUTPUT_bin})
endif()
set(EXECUTABLE_OUTPUT_PATH ${BUILD_OUTPUT_bin})

##################################################

# find OpenCV library
set(OCV_MODULES core imgcodecs imgproc)
find_package( OpenCV REQUIRED COMPONENTS ${OCV_MODULES} )

##################################################
# peprocesser definitions / options

add_definitions(-DVERSION_MAJOR=${VERSION_MAJOR})
add_definitions(-DVERSION_MINOR=${VERSION_MINOR})

# verbosity option
set(WITH_VERBOSITY ON CACHE BOOL "Build with verbosity on.")
if(WITH_VERBOSITY)
  add_definitions(-DVERBOSITY)
endif(WITH_VERBOSITY)

# openmp
set(WITH_OPENMP ON CACHE BOOL "Build with OpenMP support.")
if(WITH_OPENMP)
  add_definitions(-DOPENMP)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif(WITH_OPENMP)

##################################################
# executable target


# set include directories
include_directories(
  ${OpenCV_INCLUDE_DIRS}
)

# set list of soruce files
AUX_SOURCE_DIRECTORY(./src SRC_FILES)
file(GLOB HPP_FILES "${PROJECT_SOURCE_DIR}/src/*.hpp"
                    "${PROJECT_SOURCE_DIR}/src/*/*.hpp")
set(SRC_FILES ${SRC_FILES} ${HPP_FILES})

# set target name
set(TARGET_NAME TULIPP_SGM_v${VERSION_MAJOR}-${VERSION_MINOR})
add_definitions(-DTARGET_NAME=${TARGET_NAME})

# setup target
add_executable(${TARGET_NAME} ${SRC_FILES})
target_link_libraries(${TARGET_NAME} ${OpenCV_LIBRARIES})

############################################
