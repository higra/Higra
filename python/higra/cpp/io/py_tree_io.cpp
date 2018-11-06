/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_tree_io.hpp"
#include "higra/io/tree_io.hpp"
#include "../py_common.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include <fstream>

template<typename T>
using pyarray = xt::pyarray<T>;

void py_init_tree_io(pybind11::module &m) {
    xt::import_numpy();

    m.def("_read_tree", [](const std::string &filename) {
              std::ifstream file(filename);
              return hg::read_tree(file);
          },
          "Read tree from mixed ascii/binary format. Return a pair with the tree and a map of attributes (tree, dict[string => 1d array[double] ])",
          pybind11::arg("filename"));

    m.def("save_tree", [](const std::string &filename, const hg::tree &tree,
                          const std::map<std::string, pyarray<double>> &attributes) {
              std::ofstream file(filename);
              auto s = hg::save_tree(file, tree);
              for (auto e: attributes) {
                  s.add_attribute(e.first, e.second);
              }
              s.finalize();
          },
          "Save a tree and scalar attributes to mixed ascii/binary format. "
          "Attributes must be numpy 1d arrays stored in a dictionary with string keys (attribute names).",
          pybind11::arg("filename"),
          pybind11::arg("tree"),
          pybind11::arg("attributes") = std::map<std::string, pyarray<double>>());
}

