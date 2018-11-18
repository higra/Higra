/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_partition.hpp"
#include "../py_common.hpp"
#include "higra/assessment/partition.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

using namespace hg;
namespace py = pybind11;

struct def_assess_partition {
    template<typename value_type, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("assess_partition",
              [](const xt::pyarray <value_type> &candidate,
                 const xt::pyarray <value_type> &ground_truth,
                 partition_measure measure) {
                  switch (measure) {
                      case partition_measure::DHamming:
                          return assess_partition(candidate, ground_truth, scorer_partition_DHamming());
                      case partition_measure::DCovering:
                          return assess_partition(candidate, ground_truth, scorer_partition_DCovering());
                      case partition_measure::BCE:
                          return assess_partition(candidate, ground_truth, scorer_partition_BCE());
                      default:
                          throw std::runtime_error("Partition measure is not known, see enumeration PartitionMeasure for legal values.");
                  }
              },
              doc,
              py::arg("candidate"),
              py::arg("ground_truth"),
              py::arg("partition_measure"));
    }
};

void py_init_assessment_partition(pybind11::module &m) {
    xt::import_numpy();

    py::enum_<partition_measure>(m, "PartitionMeasure")
            .value("BCE", partition_measure::BCE)
            .value("DHamming", partition_measure::DHamming)
            .value("DCovering", partition_measure::DCovering);

    add_type_overloads<def_assess_partition, HG_TEMPLATE_INTEGRAL_TYPES>
            (m,
             "Assess a given candidate partition with a ground-truth segmentation and an evaluation measure "
             "(see enumeration PartitionMeasure). The candidate and ground-truth partitions must be given as "
             "labelisations with integers values between 0 (included) and the number of regions in the partition "
             "(excluded).");
}
