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
        plot_graph.py
        plot_tree.py
        utils.py)

set(PYMODULE_COMPONENTS ${PYMODULE_COMPONENTS}

        PARENT_SCOPE)

REGISTER_PYTHON_MODULE_FILES("${PY_FILES}")
