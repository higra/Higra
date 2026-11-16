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
#include <higra/python/py_array.hpp>

namespace py = pybind11;
PYBIND11_MODULE(higra_abi_probe, m) {
    m.attr("internals_id") = PYBIND11_INTERNALS_ID;
    m.attr("platform_abi_id") = PYBIND11_PLATFORM_ABI_ID;
    m.attr("compiler") = __VERSION__;
#ifdef _GLIBCXX_USE_CXX11_ABI
    m.attr("libstdcxx_cxx11_abi") = _GLIBCXX_USE_CXX11_ABI;
    m.attr("libstdcxx_header_date") = __GLIBCXX__;
#endif
    m.def("tree_size", [](const hg::tree& t) { return t.num_vertices(); });
    m.def("make_tree", []() {
        return hg::tree(hg::array_1d<hg::index_t>{2, 2, 2}, hg::tree_category::component_tree);
    });
    m.def("copy_tree", [](const hg::tree& t) { return hg::tree(t); });
    m.def("same_tree", [](hg::tree& t) -> hg::tree& { return t; },
          py::return_value_policy::reference);
    m.def("copy_graph", [](const hg::ugraph& g) { return hg::ugraph(g); });
    m.def("same_graph", [](hg::ugraph& g) -> hg::ugraph& { return g; },
          py::return_value_policy::reference);
    m.def("add_edge", [](hg::ugraph& g) { g.add_edge(0, 2); });
    m.def("scaled", [](hg::array_view<const double> a, double f) {
        return hg::map([f](double v) { return v * f; }, a);
    });
    m.def("scaled", [](hg::array_view<const float> a, float f) {
        return hg::map([f](float v) { return v * f; }, a);
    });
    m.def("increment", [](hg::array_view<hg::index_t> a) {
        for (auto& v : a) { ++v; }
    });
    m.def("parents_sum", [](const hg::tree& t) { return hg::sum(t.parents()); });
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
print("Downstream compiler:", probe.compiler)
if hasattr(probe, "libstdcxx_cxx11_abi"):
    print("libstdc++ ABI/header date:", probe.libstdcxx_cxx11_abi,
          probe.libstdcxx_header_date)
tree = hg.Tree(np.array([2, 2, 2], dtype=np.int64), hg.TreeCategory.ComponentTree)
assert probe.tree_size(tree) == 3
assert probe.same_tree(tree) is tree
created = probe.make_tree()
assert isinstance(created, hg.Tree)
assert created.category() == hg.TreeCategory.ComponentTree
np.testing.assert_array_equal(hg.attribute_area(created), [1, 1, 2])
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
assert probe.same_graph(graph) is graph
sources, targets = graph.edge_list()
probe.add_edge(graph)
assert graph.num_edges() == 2
np.testing.assert_array_equal(sources, [0])
np.testing.assert_array_equal(targets, [1])
copy = probe.copy_graph(graph)
assert isinstance(copy, hg.UndirectedGraph)
assert copy.num_vertices() == 3
np.testing.assert_array_equal(copy.edge_list(), graph.edge_list())
del graph, sources, targets
gc.collect()
np.testing.assert_array_equal(copy.edge_list(), [[0, 0], [1, 2]])
np.testing.assert_array_equal(copy.out_degree(np.arange(3)), [2, 1, 1])
x = np.arange(12, dtype=np.float64).reshape(3, 4)[:, ::-2]
r = probe.scaled(x, 2.0)
np.testing.assert_array_equal(r, x * 2)
assert r.dtype == np.float64 and r.flags.writeable and r.flags.c_contiguous
assert probe.scaled(np.ones(3, np.float32), 2.0).dtype == np.float32
# arrays allocated by each module are passed to the other one, then released
area = hg.attribute_area(created)
assert area.dtype == np.float64 and type(area.base).__name__ == "PyCapsule"
scaled_area = probe.scaled(area, 2.0)
assert type(scaled_area.base).__name__ == "PyCapsule"
del area
gc.collect()
np.testing.assert_array_equal(scaled_area, [2, 2, 4])
np.testing.assert_array_equal(hg.accumulate_parallel(created, scaled_area, hg.Accumulators.sum), [0, 0, 4])
depth = hg.attribute_depth(created)
assert depth.dtype == np.int64 and type(depth.base).__name__ == "PyCapsule"
probe.increment(depth)
np.testing.assert_array_equal(depth, [2, 2, 1])
del scaled_area, depth
gc.collect()
y = np.zeros(4, dtype=np.int64)
probe.increment(y)
np.testing.assert_array_equal(y, [1, 1, 1, 1])
try:
    probe.increment(np.zeros(4, dtype=np.int64)[::2])
    raise AssertionError("in-place update of a strided array must fail")
except TypeError:
    pass
assert probe.parents_sum(created) == 6
print("tree/graph/array exchange passed:", sys.argv[1])
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    parser.add_argument("--extra-flag", action="append", default=[])
    args = parser.parse_args()
    if sys.platform == "win32":
        parser.error("this probe uses Unix compiler flags")

    import higra as hg

    print("Python:", sys.version, flush=True)
    print("Installed Higra:", hg.__file__, hg.cpp.__file__, flush=True)
    subprocess.run(shlex.split(args.cxx) + ["--version"], check=True)
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
