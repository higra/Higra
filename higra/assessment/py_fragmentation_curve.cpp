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
#include "higra/assessment/partition.hpp"
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
                      const xt::pytensor<index_t, 1> &,
                      size_t>(),
              doc,
              py::arg("tree"),
              py::arg("ground_truth"),
              py::arg("optimal_cut_measure") = hg::optimal_cut_measure::BCE,
              py::arg("vertex_map") = xt::pytensor<index_t, 1>{},
              py::arg("max_regions") = 200);
    }
};

template<typename tree_t>
struct def_assesse_horizontal_cut {
    template<typename value_type, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("_assess_fragmentation_horizontal_cut",
              [](const tree_t &tree,
                 const xt::pyarray<value_type> &altitudes,
                 const xt::pyarray<index_t> &ground_truth,
                 hg::partition_measure measure,
                 const xt::pytensor<index_t, 1> &vertex_map,
                 size_t max_regions
              ) {
                  switch (measure) {
                      case partition_measure::DHamming:
                          return hg::assess_fragmentation_horizontal_cut(tree, altitudes, ground_truth,
                                                                         hg::scorer_partition_DHamming(), vertex_map,
                                                                         max_regions);
                      case partition_measure::DCovering:
                          return hg::assess_fragmentation_horizontal_cut(tree, altitudes, ground_truth,
                                                                         hg::scorer_partition_DCovering(), vertex_map,
                                                                         max_regions);
                      case partition_measure::BCE:
                          return hg::assess_fragmentation_horizontal_cut(tree, altitudes, ground_truth,
                                                                         hg::scorer_partition_BCE(), vertex_map,
                                                                         max_regions);
                      default:
                          throw std::runtime_error(
                                  "Partition measure is not known, see enumeration PartitionMeasure for legal values.");

                  }
              },
              doc,
              py::arg("tree"),
              py::arg("altitudes"),
              py::arg("ground_truth"),
              py::arg("optimal_cut_measure"),
              py::arg("vertex_map") = xt::pytensor<index_t, 1>{},
              py::arg("max_regions") = 200);
    }
};

void py_init_fragmentation_curve(pybind11::module &m) {
    xt::import_numpy();

    using fg_t = fragmentation_curve<double>;
    py::class_<fg_t>(m,
                     "FragmentationCurve",
                     "This class represents a fragmentation curve, ie the evolution of the "
                     "scores of the partitions of a hierarchy with respect to the number of regions "
                     "in those partitions.\n\n"
                     ""
                     "Example:\n\n"
                     "\tplt.plot(x=fg.num_regions(), y=fg.scores())")
            .def("num_regions", &fg_t::num_regions,
                 "Array of number of regions in the different cuts")
            .def("num_regions_normalized", &fg_t::num_regions_normalized,
                 "Array of number of regions in the different cuts divided by the number of regions in the ground-truth")
            .def("num_regions_ground_truth", &fg_t::num_regions_ground_truth,
                 "Array of number of regions in the different cuts")
            .def("scores", &fg_t::scores,
                 "Array of scores of the different cuts");


    py::enum_<optimal_cut_measure>(m, "OptimalCutMeasure",
            "Quality measures usable with optimal cut assessment")
            .value("BCE", optimal_cut_measure::BCE)
            .value("DHamming", optimal_cut_measure::DHamming)
            .value("DCovering", optimal_cut_measure::DCovering);

    auto c = py::class_<assesser_fragmentation_optimal_cut>(m, "AssesserFragmentationOptimalCut");
    add_type_overloads<def_assesser_optimal_cut_ctr, HG_TEMPLATE_INTEGRAL_TYPES>
            (c,
             "Create an assesser for hierarchy optimal cuts w.r.t. a given ground-truth partition of hierarchy"
             "leaves and the given optimal cut measure (see OptimalCutMeasure). The algorithms will explore optimal cuts containing at most "
             "max_regions regions.");

    c.def("fragmentation_curve",
          &assesser_fragmentation_optimal_cut::fragmentation_curve,
          "Fragmentation curve, i.e. for each number of region k between 1 and max_regions, "
          "the score of the optimal cut with k regions.");

    c.def("optimal_number_of_regions",
          &assesser_fragmentation_optimal_cut::optimal_number_of_regions,
          "Number of regions in the optimal cut.");

    c.def("optimal_score",
          &assesser_fragmentation_optimal_cut::optimal_score,
          "Score of the optimal cut.");

    c.def("optimal_partition",
          &assesser_fragmentation_optimal_cut::optimal_partition,
          "Labelisation of the tree vertices that corresponds to the optimal cut with"
          "the given number of regions. If the number of regions is equal to 0 (default), the "
          "global optimal cut it returned (it will contain get_optimal_number_of_regions regions).",
          py::arg("num_regions") = 0);

    add_type_overloads<def_assesse_horizontal_cut<hg::tree>, HG_TEMPLATE_NUMERIC_TYPES>
            (m,
             "Compute the fragmentation curve of the horizontal cuts in a hierarchy w.r.t. a given measure.");
}
