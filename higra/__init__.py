############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

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

# normal modules
from .sorting import *
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


def __set_num_threads_from_environment():
    # HG_NUM_THREADS: decimal digits, surrounding whitespace allowed, value between 1 and 2**63 - 1
    import os
    import warnings
    value = os.environ.get("HG_NUM_THREADS")
    whitespace = " \t\n\r\f\v"
    if value is None or value.strip(whitespace) == "":
        return
    digits = value.strip(whitespace)
    if digits.isascii() and digits.isdigit() and 0 < int(digits) < 2 ** 63:
        try:
            set_num_threads(int(digits))
            return
        except Exception:
            pass
    warnings.warn("Ignoring the environment variable HG_NUM_THREADS='" + value +
                  "': it must be a positive integer smaller than 2**63.", RuntimeWarning)

__set_num_threads_from_environment()
