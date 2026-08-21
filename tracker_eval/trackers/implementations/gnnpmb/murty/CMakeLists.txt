cmake_minimum_required(VERSION 3.16)
project(pedestrian_tracker_murty LANGUAGES CXX)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")

find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

pybind11_add_module(_murty src/murty.cpp)
target_compile_features(_murty PRIVATE cxx_std_14)
target_link_libraries(_murty PRIVATE Eigen3::Eigen)

if(DEFINED MURTY_OUTPUT_DIR)
  set_target_properties(
    _murty PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${MURTY_OUTPUT_DIR}"
  )
endif()
