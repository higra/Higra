############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################


set(FILES
        py_binary_partition_tree.cpp
        py_common.cpp
        py_hierarchy_core.cpp
        py_watershed_hierarchy.cpp)

set(PATH_PREFIX cpp/hierarchy/)

foreach (FILE ${FILES})
    set(PYMODULE_COMPONENTS ${PYMODULE_COMPONENTS}
            ${PATH_PREFIX}${FILE}
            CACHE INTERNAL "" FORCE)
endforeach ()