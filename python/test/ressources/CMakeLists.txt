if (DO_TEST)

    set(RESSOURCES_FILES
            test.graph)

    foreach (FILE ${RESSOURCES_FILES})
        configure_file(${FILE} ${CMAKE_CURRENT_BINARY_DIR}/${FILE})

    endforeach ()

endif (DO_TEST)