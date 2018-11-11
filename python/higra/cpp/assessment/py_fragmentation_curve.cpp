/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_fragmentation_curve.hpp"
#include "../py_common.hpp"
#include "higra/assessment/fragmentation_curve.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

using namespace hg;
namespace py = pybind11;

struct def_assesser_optimal_cut_ctr {
    template<typename value_type, typename C>
    static
    void def(C &c, const char *doc) {
        c.def(py::init<const hg::tree &,
                      const xt::pyarray<value_type> &,
                      optimal_cut_measure,
                      size_t>(),
              doc,
              py::arg("tree"),
              py::arg("ground_truth"),
              py::arg("optimal_cut_measure") = hg::optimal_cut_measure::BCE,
              py::arg("max_regions") = 200);
    }
};


void py_init_fragmentation_curve(pybind11::module &m) {
    xt::import_numpy();

    py::enum_<optimal_cut_measure>(m, "OptimalCutMeasure")
            .value("BCE", optimal_cut_measure::BCE)
            .value("DHamming", optimal_cut_measure::DHamming)
            .value("Covering", optimal_cut_measure::Covering);

    auto c = py::class_<assesser_optimal_cut>(m, "AssesserOptimalCut");
    add_type_overloads<def_assesser_optimal_cut_ctr, HG_TEMPLATE_INTEGRAL_TYPES>
            (c,
             "Create an assesser for hierarchy optimal cuts w.r.t. a given ground-truth partition of hierarchy"
             "leaves and the given optimal cut measure (see OptimalCutMeasure). The algorithms will explore optimal cuts containing at most "
             "max_regions regions.");

    c.def("fragmentation_curve",
          [](const assesser_optimal_cut &assesser, bool normalize) {
              auto curve = assesser.fragmentation_curve(normalize);
              return py::make_tuple(std::move(curve.k), std::move(curve.scores));
          },
          "Fragmentation curve, i.e. for each number of region k between 1 and max_regions, "
          "the score of the optimal cut with k regions. The curve is given by a pair of arrays "
          "(number_of_regions/number_of_region_ground_truth, scores) (or (number_of_regions, scores) if "
          "normalize is False), ready to be plotted:"
          "plot(x=number_of_regions, y=scores)",
          py::arg("normalize")=true);

    c.def("number_of_region_ground_truth",
          &assesser_optimal_cut::number_of_region_ground_truth,
          "Number of regions in the ground truth.");

    c.def("optimal_number_of_regions",
          &assesser_optimal_cut::optimal_number_of_regions,
          "Number of regions in the optimal cut.");

    c.def("optimal_score",
          &assesser_optimal_cut::optimal_score,
          "Score of the optimal cut.");

    c.def("optimal_partition",
          &assesser_optimal_cut::optimal_partition,
          "Labelisation of the tree vertices that corresponds to the optimal cut with"
          "the given number of regions. If the number of regions is equal to 0 (default), the "
          "global optimal cut it returned (it will contain get_optimal_number_of_regions regions).",
          py::arg("num_regions") = 0);
}
