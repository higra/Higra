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
        py_contour_2d.cpp
        py_graph_image.cpp
        py_hierarchy_mean_pb.cpp)

set(PATH_PREFIX cpp/image/)

foreach (FILE ${FILES})
    set(PYMODULE_COMPONENTS ${PYMODULE_COMPONENTS}
            ${PATH_PREFIX}${FILE}
            CACHE INTERNAL "" FORCE)
endforeach ()