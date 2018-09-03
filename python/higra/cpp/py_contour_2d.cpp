/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_contour_2d.hpp"

#include "py_common.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "higra/image/contour_2d.hpp"

using namespace hg;
namespace py = pybind11;


void py_init_contour_segment_2d(pybind11::module &m) {

    using class_t = contour_segment_2d;
    auto c = py::class_<class_t>(m, "ContourSegment2d", "A contour segment is an ordered list of contour elements that "
            "are considered to form a straight line segment."
            "A contour element is a pair (edge_index, coordinates) where coordinates is a 2d point.");
    c.def("__iter__", [](class_t &c) {

            auto transform = [](const std::pair<index_t, point_2d_f> &e){
                return std::make_pair(e.first, std::make_pair(e.second[0], e.second[1]));
            };
              // ::TODO:: remove this stupid transorm iterator when xtensor python supports it
        using transform_it_t = transform_forward_iterator<
                decltype(transform),
                contour_2d_internal::contour_segment_2d_iterator<point_2d_f >,
                std::pair<index_t, std::pair<double, double> > >;
              return py::make_iterator(transform_it_t(c.begin(), transform), transform_it_t(c.end(), transform));
              //return py::make_iterator(c.begin(), c.end());
          },
          "Iterator on the contour elements of the contour segment. ");
    c.def("__len__", &class_t::size, "Number of elements in the contour segment.");
    c.def("__getitem__", [](class_t &c, index_t i) {
        // ::TODO:: remove this stupid thing when xtensor python supports it
        const auto &e = c[i];
        return std::make_pair(e.first, std::make_pair(e.second[0], e.second[1]));
    });
    // ::TODO:: uncomment when xtensor python supports it
    //c.def("first", &class_t::first, "First element of the contour segment.");
    //c.def("last", &class_t::last, "First element of the contour segment.");
    c.def("norm", &class_t::norm, "Distance between first and last elements of the contour segment.");
    c.def("angle", &class_t::angle, "Angle of the vector (last()-first()) in [-pi; pi].");
    c.def("distance_to_point", &class_t::distance_to_point, "Distance between the given point and the line "
                  "defined by the fist and last element of the contour segment.",
          py::arg("point"));
}

void py_init_polyline_contour_2d(pybind11::module &m) {

    using class_t = polyline_contour_2d;
    auto c = py::class_<class_t>(m, "PolylineContour2d", "A polyline contour is an ordered list of contour segment "
            "that form a continuous piece of frontier between 2 regions."
            "Note that consecutive segments share their extremities: given two consecutive contour segments s1 and s2, "
            "the last element of s1 is equal to the first element of s2.");
    c.def("__iter__", [](class_t &c) {
              return py::make_iterator(c.begin(), c.end());
          },
          "Iterator on the contour segments of the polyline contour. ");
    c.def("__len__", &class_t::size, "Number of segments in the polyline contour.");
    c.def("__getitem__", [](class_t &c, index_t i) { return c[i]; });
    c.def("subdivide", &class_t::subdivide, "Subdivide teh line such that the distance between the line\n"
                  "joining the extremities of the contour segment and each of its elements is lower than the threshold (\n"
                  "Ramer–Douglas–Peucker algorithm)\n"
                  "\n"
                  "The threshold is equal to\n"
                  "- epsilon if relative_epsilon is false\n"
                  "- epsilon times the distance between the segment extremities if relative_epsilon is true\n"
                  "\n"
                  "If the distance between the segment extremities is smaller than min_size, the segment is never subdivided.",
          py::arg("epsilon") = 0.05,
          py::arg("relative_epsilon") = true,
          py::arg("min_size") = 2
    );
}

void py_init_contour_2d_impl(pybind11::module &m) {

    using class_t = contour_2d;
    auto c = py::class_<class_t>(m, "Contour2d", "A contour2d is a set of polyline contours. "
            "Each polyline contour represent a continuous piece of contour between two regions.");
    c.def("__iter__", [](class_t &c) {
              return py::make_iterator(c.begin(), c.end());
          },
          "Iterator on the polyline contours. ");
    c.def("__len__", &class_t::size, "Number of polyline contours.");
    c.def("__getitem__", [](class_t &c, index_t i) { return c[i]; });
    c.def("subdivide", &class_t::subdivide, "Subdivide teh line such that the distance between the line\n"
                  "joining the extremities of the contour segment and each of its elements is lower than the threshold (\n"
                  "Ramer–Douglas–Peucker algorithm)\n"
                  "\n"
                  "The threshold is equal to\n"
                  "- epsilon if relative_epsilon is false\n"
                  "- epsilon times the distance between the segment extremities if relative_epsilon is true\n"
                  "\n"
                  "If the distance between the segment extremities is smaller than min_size, the segment is never subdivided.\n"
                  "\n"
                  "Implementation note: simply call subdivide on each polyline of the contour.",
          py::arg("epsilon") = 0.05,
          py::arg("relative_epsilon") = true,
          py::arg("min_size") = 2
    );
}


struct def_fit_contour_2d {
    template<typename T>
    static
    void def(pybind11::module &m, const char *doc) {
        m.def("fit_contour_2d", [](const hg::ugraph &graph,
                                   const std::vector<size_t> &shape,
                                   const xt::pyarray<T> &weights) {
                  hg::embedding_grid_2d embedding(shape);
                  return hg::fit_contour_2d(graph, embedding, weights);
              },
              doc,
              py::arg("graph"),
              py::arg("shape"),
              py::arg("edgeWeights"));
    }
};

void py_init_contour_2d(pybind11::module &m) {
    xt::import_numpy();
    py_init_contour_segment_2d(m);
    py_init_polyline_contour_2d(m);
    py_init_contour_2d_impl(m);

    add_type_overloads<def_fit_contour_2d, HG_TEMPLATE_NUMERIC_TYPES>
            (m,
             "Construct a contour_2d object from a graph cut of a 2d image with a 4 adjacency (non zero edges are part of the cut).");

}
