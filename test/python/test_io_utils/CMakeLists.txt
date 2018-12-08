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
        test_pink_graph_io.py
        test_tree_io.py)

REGISTER_PYTHON_MODULE_FILES("${PY_FILES}")
