/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "higra/assessment/dendrogram_purity.hpp"
#include "../test_utils.hpp"

using namespace hg;

namespace assessment_dendrogram_purity {

    TEST_CASE("dendrogram purity", "[dendrogram purity]") {
        SECTION("binary"){
            tree t(array_1d <index_t>{5,5,6,7,7,6,8,8,8});
            array_1d<int> labels{1,1,0,1,0};

            auto p = dendrogram_purity(t, labels);

            REQUIRE(almost_equal(p, 0.65));
        }
        SECTION("non binary"){
            tree t(array_1d <index_t>{5,5,5,6,6,7,7,7});
            array_1d<int> labels{1,1,0,1,0};

            auto p = dendrogram_purity(t, labels);

            REQUIRE(almost_equal(p, 0.5666666666666667));
        }
    }

    TEST_CASE("dendrogram purity brute force", "[dendrogram purity]") {
        // non binary tree, unused label 2
        tree t(array_1d<index_t>{9, 9, 10, 10, 10, 11, 11, 12, 12, 13, 13, 13, 14, 14, 14});
        array_1d<int> labels{0, 3, 3, 0, 1, 1, 3, 1, 0};

        // pur(T) = 1/Z sum_{x < y, label(x) = label(y)} |lca(x, y) cap C_label(x)| / |lca(x, y)|
        auto area = attribute_area(t);
        double total = 0;
        double z = 0;
        for (index_t x = 0; x < 9; x++) {
            for (index_t y = x + 1; y < 9; y++) {
                if (labels(x) == labels(y)) {
                    auto l = lowest_common_ancestor(x, y, t);
                    double count = 0;
                    for (index_t v = 0; v < 9; v++) {
                        if (labels(v) == labels(x) && lowest_common_ancestor(v, l, t) == l) {
                            count++;
                        }
                    }
                    total += count / area(l);
                    z++;
                }
            }
        }
        REQUIRE(almost_equal(dendrogram_purity(t, labels), total / z, 1e-14));
        array_1d<char> labels_char = labels;
        REQUIRE(almost_equal(dendrogram_purity(t, labels_char), total / z, 1e-14));
    }
}
