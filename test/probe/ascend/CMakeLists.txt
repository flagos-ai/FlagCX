cmake_minimum_required(VERSION 3.16)

set(ASCENDC_CMAKE_DIR "$ENV{ASCENDC_CMAKE_DIR}" CACHE PATH
    "Directory containing asc_modules/FindASC.cmake")
set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH
    "CANN toolkit root")
set(SOC_VERSION "$ENV{SOC_VERSION}" CACHE STRING "Ascend SoC version")

if(NOT EXISTS "${ASCENDC_CMAKE_DIR}/asc_modules/FindASC.cmake")
  message(FATAL_ERROR "Set ASCENDC_CMAKE_DIR to the AscendC CMake directory")
endif()
if(NOT EXISTS "${ASCEND_HOME_PATH}/include/acl/acl.h")
  message(FATAL_ERROR "Set ASCEND_HOME_PATH to the CANN toolkit root")
endif()
if(NOT SOC_VERSION)
  message(FATAL_ERROR "Set SOC_VERSION to the target Ascend SoC")
endif()

string(TOLOWER "${SOC_VERSION}" SOC_VERSION_LOWER)
if(SOC_VERSION_LOWER MATCHES "^ascend(950|350)")
  set(CMAKE_ASC_ARCHITECTURES dav-3510)
elseif(SOC_VERSION_LOWER MATCHES "^ascend910(b|c|_93)")
  set(CMAKE_ASC_ARCHITECTURES dav-2201)
else()
  message(FATAL_ERROR "Unsupported SOC_VERSION: ${SOC_VERSION}")
endif()

list(APPEND CMAKE_MODULE_PATH "${ASCENDC_CMAKE_DIR}/asc_modules")
find_package(ASC REQUIRED)
project(Ascend_Stream_Visibility_Probe LANGUAGES ASC CXX)
find_package(MPI REQUIRED COMPONENTS CXX)

# AscendC's static target adds bare -l dependencies. Match the installed CANN
# examples and provide their search directory to the final host link step.
link_directories("${ASCEND_HOME_PATH}/lib64")

ascendc_library(stream_consumer STATIC stream_consumer.cpp)
target_compile_options(stream_consumer PRIVATE
  $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)
target_include_directories(stream_consumer PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(stream_visibility_probe stream_visibility_probe.cpp)
target_compile_features(stream_visibility_probe PRIVATE cxx_std_17)
target_compile_options(stream_visibility_probe PRIVATE -Wall -Wextra -Werror -g)
target_include_directories(stream_visibility_probe PRIVATE
  "${ASCEND_HOME_PATH}/include" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(stream_visibility_probe PRIVATE
  stream_consumer MPI::MPI_CXX "${ASCEND_HOME_PATH}/lib64/libascendcl.so"
  ${CMAKE_DL_LIBS})
set_target_properties(stream_visibility_probe PROPERTIES
  BUILD_RPATH "${ASCEND_HOME_PATH}/lib64")

# The NIC path is optional because many CANN hosts do not ship userspace verbs.
find_path(IBVERBS_INCLUDE_DIR infiniband/verbs.h)
find_library(IBVERBS_LIBRARY ibverbs)
if(IBVERBS_INCLUDE_DIR AND IBVERBS_LIBRARY)
  add_executable(ib_visibility_probe ib_visibility_probe.cpp)
  target_compile_features(ib_visibility_probe PRIVATE cxx_std_17)
  target_compile_options(ib_visibility_probe PRIVATE -Wall -Wextra -Werror -g)
  target_include_directories(ib_visibility_probe PRIVATE
    "${ASCEND_HOME_PATH}/include" "${CMAKE_CURRENT_SOURCE_DIR}"
    "${IBVERBS_INCLUDE_DIR}")
  target_link_libraries(ib_visibility_probe PRIVATE
    stream_consumer MPI::MPI_CXX "${ASCEND_HOME_PATH}/lib64/libascendcl.so"
    "${IBVERBS_LIBRARY}")
  set_target_properties(ib_visibility_probe PROPERTIES
    BUILD_RPATH "${ASCEND_HOME_PATH}/lib64")
endif()
