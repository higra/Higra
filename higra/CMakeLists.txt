############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/lib/include ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})

message("Python include dirs: ${Python_INCLUDE_DIRS}")
message("Python numpy include dirs: ${Python_NumPy_INCLUDE_DIRS}")

set(PY_FILES
        __init__.py
        concept.py
        data_cache.py
        hg_utils.py
        sorting.py)

REGISTER_PYTHON_MODULE_FILES("${PY_FILES}")

set(PYMODULE_COMPONENTS
        py_sorting.cpp
        pymodule.cpp)

set_source_files_properties(
        pymodule.cpp PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON
)

add_subdirectory(accumulator)
add_subdirectory(algo)
add_subdirectory(assessment)
add_subdirectory(attribute)
add_subdirectory(detail)
add_subdirectory(hierarchy)
add_subdirectory(image)
add_subdirectory(interop)
add_subdirectory(io_utils)
add_subdirectory(plot)
add_subdirectory(structure)

pybind11_add_module(higram ${PYMODULE_COMPONENTS})
#target_include_directories(higram PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/lib/include ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})


if (HG_USE_TBB)
    add_definitions("-DXTENSOR_USE_TBB")
    target_compile_definitions(higram PRIVATE HG_USE_TBB)
    target_include_directories(higram PRIVATE ${TBB_INCLUDE_DIRS})
    if(HG_BUILD_WHEEL AND WIN32) # hack for windows wheels to get unique names
        get_filename_component(LIB_PATH tbb.lib ABSOLUTE)
        set(TBB_LIBFILE ${LIB_PATH})
    else()
        set(TBB_LIBFILE ${TBB_LIBRARIES})
    endif()
    target_link_libraries(higram PRIVATE ${TBB_LIBFILE})
    message("TBBFILE system library: ${TBB_LIBFILE}")
    message("TBBFILE library used: ${TBB_LIBFILE}")
endif ()

get_target_property(_higram_file_name higram OUTPUT_NAME)
set(HIGRA_CPP_MODULE_NAME $<TARGET_FILE:${higram}> PARENT_SCOPE)
set(UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} higram CACHE INTERNAL "" FORCE)

if (HG_UNITY_BUILD)
    message("Enabling unity build for higram")
    set_target_properties(higram PROPERTIES UNITY_BUILD ON)
    set_target_properties(higram PROPERTIES UNITY_BUILD_BATCH_SIZE ${HG_UNITY_BUILD_BATCH_SIZE})
endif ()

message("Python CMAKE_CXX_FLAGS" ${CMAKE_CXX_FLAGS})