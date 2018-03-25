//
// Created by user on 3/24/18.
//
#include "pybind11/stl.h"
#include "py_embedding.hpp"
#include "embedding.hpp"
#include "xtensor/xarray.hpp"
#include <vector>

using namespace hg;
namespace py = pybind11;

void py_init_embedding(pybind11::module &m) {

    py::class_<embedding_grid>(m, "EmbeddingGrid")
            .def(py::init([](std::vector<std::size_t> shape) { return embedding_grid::make_embedding_grid(shape); }),
                 "Create a new grid embedding.",
                 py::arg("shape"))
            .def(py::init([](xt::xarray<std::size_t> shape) { return embedding_grid::make_embedding_grid(shape); }),
                 "Create a new grid embedding.",
                 py::arg("shape"))
            .def("shape", &embedding_grid::getShape,
                 "Get the shape/dimensions of the grid embedding")
            .def("size", &embedding_grid::size, "Get the total number of points contained in the embedding.");
    //.def("contains")

}