cmake_minimum_required(VERSION 3.24)
project(nccl_cp VERSION 0.1.0 LANGUAGES NONE)

option(NCCL_CP_BUILD_NATIVE "Build the CUDA/TorchBind shared library" ON)
option(NCCL_CP_BUILD_TESTS "Enable CPU tests through CTest" OFF)
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(cmake/Dependencies.cmake)
set(Python3_FIND_STRATEGY LOCATION)
set(Python3_FIND_VIRTUALENV FIRST)
find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/src/_build_info.py"
  RESULT_VARIABLE _torch_result OUTPUT_VARIABLE _torch_json ERROR_VARIABLE _torch_error
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _torch_result EQUAL 0)
  message(FATAL_ERROR "The selected Python must provide PyTorch: ${_torch_error}")
endif()
string(JSON NCCL_CP_TORCH_ROOT GET "${_torch_json}" torch_root)
string(JSON NCCL_CP_TORCH_VERSION GET "${_torch_json}" torch_version)
string(JSON NCCL_CP_TORCH_CUDA GET "${_torch_json}" torch_cuda_version)
string(JSON NCCL_CP_NCCL_AVAILABLE GET "${_torch_json}" nccl_available)
string(JSON NCCL_CP_CXX11_ABI GET "${_torch_json}" cxx11_abi)
if(NCCL_CP_TORCH_VERSION VERSION_LESS 2.12)
  message(FATAL_ERROR "NCCL CP requires a compatible PyTorch 2.12+ build")
endif()
message(STATUS "NCCL CP Python: ${Python3_EXECUTABLE}")
message(STATUS "NCCL CP PyTorch: ${NCCL_CP_TORCH_VERSION} (${NCCL_CP_TORCH_ROOT})")

set(NCCL_CP_PACKAGE_DIR "${CMAKE_CURRENT_BINARY_DIR}/python/nccl/cp")
set(NCCL_CP_BIN_DIR "${CMAKE_CURRENT_BINARY_DIR}/bin")
file(MAKE_DIRECTORY "${NCCL_CP_PACKAGE_DIR}" "${NCCL_CP_BIN_DIR}")
configure_file(tools/run.py "${NCCL_CP_BIN_DIR}/nccl_cp_run.py" COPYONLY)
file(GLOB NCCL_CP_PYTHON_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.py" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.pyi")
list(APPEND NCCL_CP_PYTHON_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/py.typed")
set(NCCL_CP_NATIVE_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/src/zero_cta_collective.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/zero_cta_kernels.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/zero_cta_kernels.h")

add_custom_target(nccl_cp_python ALL
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
    ${NCCL_CP_PYTHON_SOURCES} ${NCCL_CP_NATIVE_SOURCES}
    "${CMAKE_CURRENT_SOURCE_DIR}/NOTICE" "${NCCL_CP_PACKAGE_DIR}"
  COMMENT "Staging the nccl.cp Python package" VERBATIM)

if(NCCL_CP_BUILD_NATIVE)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "Native NCCL CP requires Linux and CUDA. Use NCCL_CP_BUILD_NATIVE=OFF only for CPU checks.")
  endif()
  if(NOT NCCL_CP_TORCH_CUDA OR NOT NCCL_CP_NCCL_AVAILABLE)
    message(FATAL_ERROR "The selected Python must provide CUDA-enabled PyTorch with NCCL support")
  endif()
  if(NOT CUDAToolkit_ROOT AND DEFINED ENV{CUDA_HOME})
    set(CUDAToolkit_ROOT "$ENV{CUDA_HOME}" CACHE PATH "CUDA toolkit root")
  endif()
  enable_language(CXX)
  find_package(CUDAToolkit REQUIRED)
  nccl_cp_select_architectures("${CUDAToolkit_NVCC_EXECUTABLE}")
  if(NOT CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
    set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}" CACHE FILEPATH "CUDA compiler")
  endif()
  enable_language(CUDA)
  find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
  string(REGEX MATCH "^[0-9]+[.][0-9]+" _cuda_major_minor "${CUDAToolkit_VERSION}")
  if(NOT _cuda_major_minor STREQUAL NCCL_CP_TORCH_CUDA)
    message(FATAL_ERROR "CUDA toolkit ${CUDAToolkit_VERSION} does not match PyTorch CUDA ${NCCL_CP_TORCH_CUDA}")
  endif()

  nccl_cp_find_nccl("${CMAKE_CURRENT_SOURCE_DIR}")
  file(READ "${NCCL_HOME}/include/nccl.h" _nccl_header)
  foreach(_component MAJOR MINOR PATCH)
    string(REGEX MATCH "#define[ \t]+NCCL_${_component}[ \t]+([0-9]+)" _match "${_nccl_header}")
    if(NOT _match)
      message(FATAL_ERROR "Cannot read NCCL_${_component} from nccl.h")
    endif()
    set(NCCL_CP_NCCL_${_component} "${CMAKE_MATCH_1}")
  endforeach()
  set(NCCL_CP_NCCL_HEADER_VERSION "${NCCL_CP_NCCL_MAJOR}.${NCCL_CP_NCCL_MINOR}.${NCCL_CP_NCCL_PATCH}")
  if(NCCL_CP_NCCL_HEADER_VERSION VERSION_LESS 2.30)
    message(FATAL_ERROR "NCCL CP requires NCCL 2.30+ host RMA APIs")
  endif()
  string(JSON _runtime_nccl_major GET "${_torch_json}" nccl_version 0)
  string(JSON _runtime_nccl_minor GET "${_torch_json}" nccl_version 1)
  string(JSON _runtime_nccl_patch GET "${_torch_json}" nccl_version 2)
  set(_runtime_nccl_version "${_runtime_nccl_major}.${_runtime_nccl_minor}.${_runtime_nccl_patch}")
  if(NOT NCCL_CP_NCCL_HEADER_VERSION VERSION_EQUAL _runtime_nccl_version)
    message(FATAL_ERROR "NCCL headers ${NCCL_CP_NCCL_HEADER_VERSION} do not match PyTorch's NCCL ${_runtime_nccl_version}")
  endif()
  message(STATUS "NCCL CP CUDA: ${CUDAToolkit_VERSION} (${CMAKE_CUDA_COMPILER})")
  message(STATUS "NCCL CP CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
  message(STATUS "NCCL CP NCCL: ${NCCL_CP_NCCL_HEADER_VERSION} (${NCCL_CP_NCCL_LIBRARY})")

  foreach(_header
    torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp
    torch/csrc/distributed/c10d/symm_mem/NCCLSymmetricMemory.hpp)
    if(NOT EXISTS "${NCCL_CP_TORCH_ROOT}/include/${_header}")
      message(FATAL_ERROR "The selected PyTorch build is missing ${_header}")
    endif()
  endforeach()
  set(_torch_libraries "")
  foreach(_library c10 torch torch_cpu torch_python c10_cuda torch_cuda)
    find_library(NCCL_CP_${_library}_LIBRARY NAMES "${_library}"
      PATHS "${NCCL_CP_TORCH_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
    list(APPEND _torch_libraries "${NCCL_CP_${_library}_LIBRARY}")
  endforeach()

  add_library(nccl_cp SHARED ${NCCL_CP_NATIVE_SOURCES})
  set_target_properties(nccl_cp PROPERTIES
    OUTPUT_NAME nccl_cp
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION OFF
    BUILD_WITH_INSTALL_RPATH TRUE)
  get_filename_component(_nccl_library_dir "${NCCL_CP_NCCL_LIBRARY}" DIRECTORY)
  set_target_properties(nccl_cp PROPERTIES
    INSTALL_RPATH "$ORIGIN;${NCCL_CP_TORCH_ROOT}/lib;${_nccl_library_dir}")
  target_include_directories(nccl_cp PRIVATE
    "${NCCL_CP_TORCH_ROOT}/include"
    "${NCCL_CP_TORCH_ROOT}/include/torch/csrc/api/include"
    "${NCCL_HOME}/include")
  target_compile_definitions(nccl_cp PRIVATE
    USE_NCCL=1 USE_C10D_NCCL=1 TORCH_API_INCLUDE_EXTENSION_H
    TORCH_EXTENSION_NAME=nccl_cp_native _GLIBCXX_USE_CXX11_ABI=${NCCL_CP_CXX11_ABI})
  foreach(_field compiler_type stdlib build_abi)
    string(JSON _value GET "${_torch_json}" pybind_${_field})
    if(_value)
      string(TOUPPER "${_field}" _name)
      target_compile_definitions(nccl_cp PRIVATE "PYBIND11_${_name}=\"${_value}\"")
    endif()
  endforeach()
  target_compile_options(nccl_cp PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-D__CUDA_NO_HALF_OPERATORS__>
    $<$<COMPILE_LANGUAGE:CUDA>:-D__CUDA_NO_HALF_CONVERSIONS__>
    $<$<COMPILE_LANGUAGE:CUDA>:-D__CUDA_NO_BFLOAT16_CONVERSIONS__>
    $<$<COMPILE_LANGUAGE:CUDA>:-D__CUDA_NO_HALF2_OPERATORS__>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  target_link_libraries(nccl_cp PRIVATE
    ${_torch_libraries} "${NCCL_CP_NCCL_LIBRARY}"
    Python3::Module CUDA::cudart CUDA::cuda_driver)
  add_custom_command(TARGET nccl_cp POST_BUILD
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/src/_build_info.py"
      --library "$<TARGET_FILE:nccl_cp>"
      --cuda-toolkit "${CUDAToolkit_VERSION}"
      --nccl-header-version "${NCCL_CP_NCCL_HEADER_VERSION}"
      --architectures "${CMAKE_CUDA_ARCHITECTURES}"
    VERBATIM)
  add_dependencies(nccl_cp_python nccl_cp)
  add_custom_command(TARGET nccl_cp_python POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "$<TARGET_FILE:nccl_cp>"
      "$<TARGET_FILE_DIR:nccl_cp>/libnccl_cp.build.json" "${NCCL_CP_PACKAGE_DIR}"
    VERBATIM)
  install(TARGETS nccl_cp LIBRARY DESTINATION lib)
  install(FILES "$<TARGET_FILE_DIR:nccl_cp>/libnccl_cp.build.json" DESTINATION lib)
  install(FILES "$<TARGET_FILE:nccl_cp>"
    "$<TARGET_FILE_DIR:nccl_cp>/libnccl_cp.build.json" DESTINATION python/nccl/cp)
else()
  message(STATUS "NCCL_CP_BUILD_NATIVE=OFF: staging and CPU tests only; no communication library is built")
endif()

install(FILES ${NCCL_CP_PYTHON_SOURCES} ${NCCL_CP_NATIVE_SOURCES} NOTICE DESTINATION python/nccl/cp)
install(PROGRAMS "${NCCL_CP_BIN_DIR}/nccl_cp_run.py" DESTINATION bin)
install(FILES LICENSE NOTICE DESTINATION share/licenses/nccl_cp)
install(FILES tests/run_distributed.py tests/run_public.py tests/route_cases.py DESTINATION share/nccl_cp/tests)
install(FILES examples/basic.py DESTINATION share/nccl_cp/examples)

if(NCCL_CP_BUILD_TESTS)
  execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import pytest"
    RESULT_VARIABLE _pytest_result ERROR_QUIET)
  if(NOT _pytest_result EQUAL 0)
    message(FATAL_ERROR "NCCL_CP_BUILD_TESTS=ON requires pytest in the selected Python environment")
  endif()
  enable_testing()
  add_test(NAME nccl_cp_cpu
    COMMAND "${Python3_EXECUTABLE}" "${NCCL_CP_BIN_DIR}/nccl_cp_run.py"
      --module pytest "${CMAKE_CURRENT_SOURCE_DIR}/tests"
      -c "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" -p no:cacheprovider -q)
  set_tests_properties(nccl_cp_cpu PROPERTIES LABELS cpu
    ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1;NCCL_CP_CMAKE_COMMAND=${CMAKE_COMMAND}")
endif()
