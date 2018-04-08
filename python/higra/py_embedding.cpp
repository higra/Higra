//
// Created by user on 3/24/18.
//

#include "xtensor-python/pyarray.hpp"
#include "pybind11/stl.h"
#include "py_embedding.hpp"
#include "embedding.hpp"
#include "utils.hpp"
#include "xtensor/xeval.hpp"
#include <string>

using namespace hg;
namespace py = pybind11;


#define LIN2GRID(rawXKCD, dataXKCD, type) \
    .def("lin2grid", \
    [](const class_t &e, const xt::pyarray<type> &a) { return xt::eval(e.lin2grid(a)); }, \
    "Takes a n1 x n2 x ... nk  array of type " HG_XSTR(type) ", and returns an array of dimension n1 x n2 x ... nk x self.dimension() " \
    "where each value, seen as the linear coordinates of a point, has been replaced by the corresponding nd coordinates.", \
    py::arg("points"))

#define CONTAINS(rawXKCD, dataXKCD, type) \
    .def("contains", \
    [](const class_t &e, const xt::pyarray<type> &a) { return xt::eval(e.contains(a)); },\
    "Takes a n1 x n2 x ... nk  array of type " HG_XSTR(type) ", with nk = self.dimension(), and returns a boolean array of dimension n1 x n2 x ... n(k-1) "\
    "indicating if each point is contained in the embedding.",\
    py::arg("points"))

#define GRID2LIN(rawXKCD, dataXKCD, type) \
    .def("grid2lin", \
    [](const class_t &e, const xt::pyarray<type> &a) { return xt::eval(e.grid2lin(a)); },\
    "Takes a n1 x n2 x ... nk  array of type " HG_XSTR(type) ", with nk = self.dimension(), and returns a uint64 array of dimension n1 x n2 x ... n(k-1) "\
    "giving the linear coordinate of each point.",\
    py::arg("points"))

template<int dim>
void py_init_embedding_impl(pybind11::module &m) {

    using class_t = embedding_grid<dim>;
    auto c = py::class_<class_t>(m, ("EmbeddingGrid" + std::to_string(dim) + "d").c_str())
            .def(py::init<const std::vector<long> &>(),
                 "Create a new grid embedding. Shape must be a 1d array with striclty positive values.",
                 py::arg("shape"))
            .def(py::init<const xt::pyarray<long> &>(),
                 "Create a new grid embedding. Shape must be a 1d array with striclty positive values.",
                 py::arg("shape"))
            .def("shape", [](const class_t &e) { return xt::pyarray<long>(e.shape()); },
                 "Get the shape/dimensions of the grid embedding")
            .def("size", &class_t::size, "Get the total number of points contained in the embedding.")
            .def("dimension", &class_t::dimension,
                 "Get the dimension of the embedding (aka self.shape().size()).")
            .def("contains", [](const class_t &e, const std::vector<long> &a) { return e.contains(a); },
                 "Takes a list or tuple representing the coordinates of a point and returns true if the point is contained in the embedding.",
                 py::arg("coordinates"))
                    HG_FOREACH(CONTAINS, HG_SINTEGRAL_TYPES)
            .def("lin2grid", [](const class_t &e, const std::size_t a) { return xt::pyarray<long>(e.lin2grid(a)); },
                 "Compute the nd coordinates of a point given its linear coordinate.",
                 py::arg("index"))
                    HG_FOREACH(LIN2GRID, HG_INTEGRAL_TYPES)
            .def("grid2lin", [](const class_t &e, const std::vector<long> &a) { return e.grid2lin(a); },
                 "Compute the linear coordinate of a point given its nd coordinates.",
                 py::arg("coordinates"))
                    HG_FOREACH(GRID2LIN, HG_SINTEGRAL_TYPES);


}

void py_init_embedding(pybind11::module &m) {
    xt::import_numpy();
    //py_init_embedding_impl<1>(m);
    py_init_embedding_impl<2>(m);
    py_init_embedding_impl<3>(m);
    //py_init_embedding_impl<4>(m);
    //py_init_embedding_impl<5>(m);
}

#undef CONTAINS
#undef LIN2GRID
#undef GRID2LIN