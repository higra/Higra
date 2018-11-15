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
        test_contour2d.cpp
        test_graph_image.cpp)

set(PATH_PREFIX image/)

foreach (FILE ${FILES})
    set(TEST_CPP_COMPONENTS ${TEST_CPP_COMPONENTS}
            ${PATH_PREFIX}${FILE}
            PARENT_SCOPE)
endforeach ()

