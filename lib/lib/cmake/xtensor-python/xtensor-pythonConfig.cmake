############################################################################
# Copyright (c) Wolf Vollprecht, Johan Mabille and Sylvain Corlay          #
# Copyright (c) QuantStack                                                 #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

# xtensor-python cmake module
# This module sets the following variables in your project::
#
#   xtensor-python_FOUND - true if xtensor-python found on the system
#   xtensor-python_INCLUDE_DIRS - the directory containing xtensor-python headers
#   xtensor-python_LIBRARY - empty


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was xtensor-pythonConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

if(NOT TARGET xtensor-python)
  include("${CMAKE_CURRENT_LIST_DIR}/xtensor-pythonTargets.cmake")
  get_target_property(xtensor-python_INCLUDE_DIRS xtensor-python INTERFACE_INCLUDE_DIRECTORIES)
endif()
