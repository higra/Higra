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


@hg.extend_class(hg.EmbeddingGrid1d, method_name="__reduce__")
@hg.extend_class(hg.EmbeddingGrid2d, method_name="__reduce__")
@hg.extend_class(hg.EmbeddingGrid3d, method_name="__reduce__")
@hg.extend_class(hg.EmbeddingGrid4d, method_name="__reduce__")
@hg.extend_class(hg.EmbeddingGrid5d, method_name="__reduce__")
def ____reduce__(self):
    return self.__class__, (self.shape(),), self.__dict__
