if (DO_TEST)

    set(PYTHON_MODULE_PATH ${CMAKE_BINARY_DIR}/python)

    set(PYTHON_TEST_FILES
            test_undirected_graph.py)

    foreach (FILE ${PYTHON_TEST_FILES})
        configure_file(${FILE} ${CMAKE_CURRENT_BINARY_DIR}/${FILE})
        add_test(NAME ${FILE}
                COMMAND ${PYTHON_EXECUTABLE}
                ${CMAKE_CURRENT_BINARY_DIR}/${FILE}
                )
    endforeach ()


    #set(UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} _mortof CACHE INTERNAL "" FORCE)
endif (DO_TEST)