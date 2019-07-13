############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

# pre-declaration of globals
globals()["__data_providers"] = {}
globals()["__higra_global_cache"] = None
globals()["__provider_caching"] = True

# extension module
from .higram import *
# required to access member names starting with underscore
from . import higram as cpp

# modules that have to be loaded before the others
# (required for correct file parsing, eg. defining top level decorators)
from . import data_cache
from .data_cache import *
from .concept import *
from .hg_utils import *

data_cache._data_cache__init()

# normal modules
from .accumulator import *
from .algo import *
from .assessment import *
from .attribute import *
from .hierarchy import *
from .image import *
from .interop import *
from .io_utils import *
from .plot import *
from .structure import *


def __logger_printer(m):
    print(m)
