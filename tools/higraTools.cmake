function(REGISTER_PYTHON_MODULE_FILES PYTHON_FILES)
    foreach (FILE ${PYTHON_FILES})
        configure_file(${FILE} ${CMAKE_CURRENT_BINARY_DIR}/${FILE} COPYONLY)
    endforeach ()

    # Remove copies of Python files that are no longer registered in this directory (removed or renamed
    # sources), so that test discovery and imports from the build tree do not pick up obsolete files.
    if (NOT CMAKE_CURRENT_BINARY_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        file(GLOB COPIED_PYTHON_FILES RELATIVE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/*.py)
        foreach (FILE ${COPIED_PYTHON_FILES})
            if (NOT FILE IN_LIST PYTHON_FILES)
                message(STATUS "Removing obsolete copy ${CMAKE_CURRENT_BINARY_DIR}/${FILE}")
                file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/${FILE})
            endif ()
        endforeach ()
    endif ()
endfunction(REGISTER_PYTHON_MODULE_FILES)
