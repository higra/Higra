if (DO_TEST)

    set(PYTHON_MODULE_PATH ${CMAKE_BINARY_DIR}/python/)

    set(PYTHON_TEST_FILES
            test_undirected_graph.py
            test_embedding.py
            test_regular_graph.py
            test_tree.py
            test_graph_weights.py
            test_graph_image.py
            test_lca_fast.py
            test_hierarchy_core.py
            test_pink_graph_io.py
            test_data_cache.py
            test_algo_tree.py
            test_attributes.py
            test_rag.py
            test_watershed.py)

    foreach (FILE ${PYTHON_TEST_FILES})
        configure_file(${FILE} ${CMAKE_CURRENT_BINARY_DIR}/${FILE})
        add_test(NAME ${FILE}
                COMMAND ${PYTHON_EXECUTABLE}
                ${CMAKE_CURRENT_BINARY_DIR}/${FILE}
                )

    endforeach ()


endif (DO_TEST)