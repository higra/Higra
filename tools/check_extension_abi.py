# Copyright ESIEE Paris (2026)
# Distributed under the terms of the CECILL-B License.
# The full license is in the file LICENSE, distributed with this software.

"""Check installed Higra object exchange with an independently compiled module.

Run outside the source checkout using the Python environment containing the
wheel to test. This Unix compiler probe uses that wheel's exported headers.
"""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import sysconfig
import tempfile


SOURCE = r"""
#include <pybind11/pybind11.h>
#include <higra/structure/tree_graph.hpp>
#include <higra/structure/undirected_graph.hpp>

namespace py = pybind11;
PYBIND11_MODULE(higra_abi_probe, m) {
    m.attr("internals_id") = PYBIND11_INTERNALS_ID;
    m.attr("platform_abi_id") = PYBIND11_PLATFORM_ABI_ID;
    m.def("tree_size", [](const hg::tree& t) { return t.num_vertices(); });
    m.def("copy_tree", [](const hg::tree& t) { return hg::tree(t); });
    m.def("same_tree", [](hg::tree& t) -> hg::tree& { return t; },
          py::return_value_policy::reference);
    m.def("copy_graph", [](const hg::ugraph& g) { return hg::ugraph(g); });
    m.def("add_edge", [](hg::ugraph& g) { g.add_edge(0, 2); });
}
"""

CHECK = r"""
import gc
import sys
import numpy as np
if sys.argv[1] == "probe_first":
    import higra_abi_probe as probe
    import higra as hg
else:
    import higra as hg
    import higra_abi_probe as probe
print(hg.__file__, hg.cpp.__file__, probe.__file__)
print(probe.internals_id, probe.platform_abi_id)
tree = hg.Tree(np.array([2, 2, 2], dtype=np.int64), hg.TreeCategory.ComponentTree)
assert probe.tree_size(tree) == 3
assert probe.same_tree(tree) is tree
copy = probe.copy_tree(tree)
assert isinstance(copy, hg.Tree)
assert copy.category() == tree.category()
np.testing.assert_array_equal(copy.parents(), tree.parents())
parents = copy.parents()
del tree, copy
gc.collect()
np.testing.assert_array_equal(parents, [2, 2, 2])
graph = hg.UndirectedGraph(3)
graph.add_edge(0, 1)
probe.add_edge(graph)
assert graph.num_edges() == 2
copy = probe.copy_graph(graph)
assert isinstance(copy, hg.UndirectedGraph)
assert copy.num_vertices() == 3
np.testing.assert_array_equal(copy.edge_list(), graph.edge_list())
del graph
gc.collect()
np.testing.assert_array_equal(copy.edge_list(), [[0, 0], [1, 2]])
print("tree/graph exchange passed:", sys.argv[1])
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    parser.add_argument("--extra-flag", action="append", default=[])
    args = parser.parse_args()
    if sys.platform == "win32":
        parser.error("this probe uses Unix compiler flags")

    import higra as hg

    includes = [sysconfig.get_path("include"), hg.get_include(), hg.get_lib_include()]
    if not all(Path(p).is_dir() for p in includes):
        parser.error("use an installed wheel with exported headers")
    with tempfile.TemporaryDirectory(prefix="higra-abi-probe-") as directory:
        root = Path(directory)
        source = root / "probe.cpp"
        source.write_text(SOURCE)
        module = root / ("higra_abi_probe" + sysconfig.get_config_var("EXT_SUFFIX"))
        command = shlex.split(args.cxx) + [
            "-std=c++17", "-shared", "-fPIC", "-fvisibility=hidden", "-O0",
            *["-I" + p for p in includes], *args.extra_flag,
            str(source), "-o", str(module),
        ]
        if sys.platform == "darwin":
            command += ["-undefined", "dynamic_lookup"]
        print(shlex.join(command), flush=True)
        subprocess.run(command, check=True)
        for order in ("higra_first", "probe_first"):
            subprocess.run([sys.executable, "-c", CHECK, order], cwd=root, check=True)


if __name__ == "__main__":
    main()
