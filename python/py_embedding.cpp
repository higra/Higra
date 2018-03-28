//
// Created by user on 3/24/18.
//
#include "pybind11/stl.h"
#include "py_embedding.hpp"
#include "embedding.hpp"
#include "xtensor/xexpression.hpp"
#include <xtensor-python/pyarray.hpp>

using namespace hg;
namespace py = pybind11;

void py_init_embedding(pybind11::module &m) {

    py::class_<embedding_grid>(m, "EmbeddingGrid")
            .def(py::init<const std::vector<long> &>(),
                 "Create a new grid embedding. Shape must be a 1d array with striclty positive values.",
                 py::arg("shape"))
            .def(py::init<const xt::pyarray<long> &>(),
                 "Create a new grid embedding. Shape must be a 1d array with striclty positive values.",
                 py::arg("shape"))
            .def("shape", &embedding_grid::getShape,
                 "Get the shape/dimensions of the grid embedding")
            .def("size", &embedding_grid::size, "Get the total number of points contained in the embedding.")
            .def("dimension", &embedding_grid::dimension,
                 "Get the dimension of the embedding (aka self.shape().size()).");
    /* .def("contains", &embedding_grid::containsV<xt::xexpression<xt::pyarray<int>>>,
          "Takes a n1 x n2 x ... nk  array, with nk = self.dimension(), and returns a boolean array of dimension n1 x n2 x ... n(k-1) "
                  "indicating wether each point is contained in the embedding.",
          py::arg("points"));
*/

}