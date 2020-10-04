############################################################################
# Copyright ESIEE Paris (2020)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

import higra as hg


@hg.extend_class(hg.RegularGraph1d, method_name="__reduce__")
def ____reduce__(self):
    return self.__class__, (self.shape(), self.neighbour_list()), self.__dict__


@hg.extend_class(hg.RegularGraph2d, method_name="__reduce__")
def ____reduce__(self):
    return self.__class__, (self.shape(), self.neighbour_list()), self.__dict__


@hg.extend_class(hg.RegularGraph3d, method_name="__reduce__")
def ____reduce__(self):
    return self.__class__, (self.shape(), self.neighbour_list()), self.__dict__


@hg.extend_class(hg.RegularGraph4d, method_name="__reduce__")
def ____reduce__(self):
    return self.__class__, (self.shape(), self.neighbour_list()), self.__dict__


@hg.extend_class(hg.RegularGraph5d, method_name="__reduce__")
def ____reduce__(self):
    return self.__class__, (self.shape(), self.neighbour_list()), self.__dict__
