#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Catch2::Catch2" for configuration ""
set_property(TARGET Catch2::Catch2 APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(Catch2::Catch2 PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libCatch2.a"
  )

list(APPEND _cmake_import_check_targets Catch2::Catch2 )
list(APPEND _cmake_import_check_files_for_Catch2::Catch2 "${_IMPORT_PREFIX}/lib/libCatch2.a" )

# Import target "Catch2::Catch2WithMain" for configuration ""
set_property(TARGET Catch2::Catch2WithMain APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(Catch2::Catch2WithMain PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libCatch2Main.a"
  )

list(APPEND _cmake_import_check_targets Catch2::Catch2WithMain )
list(APPEND _cmake_import_check_files_for_Catch2::Catch2WithMain "${_IMPORT_PREFIX}/lib/libCatch2Main.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
