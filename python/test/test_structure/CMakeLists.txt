############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

set(PY_FILES
        __init__.py
        test_embedding.py
        test_lca_fast.py
        test_regular_graph.py
        test_tree.py
        test_undirected_graph.py)

REGISTER_PYTHON_MODULE_FILES("${PY_FILES}")
