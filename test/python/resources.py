############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

import os
import os.path

__my_path = os.path.dirname(os.path.abspath(__file__))


def get_ressource_path(filename):
    global __my_path
    res = os.path.join(__my_path, "resources", filename)
    print(res)
    return res
