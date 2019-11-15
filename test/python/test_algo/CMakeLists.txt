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
        test_alignment.py
        test_graph_core.py
        test_graph_weights.py
        test_horizontal_cuts.py
        test_rag.py
        test_tree.py
        test_tree_energy_optimization.py
        test_tree_fusion.py
        test_watershed.py)

REGISTER_PYTHON_MODULE_FILES("${PY_FILES}")
