/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Wonder Alexandre Luz Alves                              *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "../test_utils.hpp"
#include "higra/detail/hierarchy/dynamic_component_tree.hpp"
#include "higra/detail/hierarchy/dynamic_component_tree_attribute_computers.hpp"
#include "higra/hierarchy/component_tree.hpp"
#include "higra/image/graph_image.hpp"

/*
* Reference dynamic maxtree:
*      
*               2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1,
*               2, 5, 5, 5, 3, 3, 3, 2, 2, 2, 3, 1,
*               2, 5, 6, 5, 3, 3, 3, 2, 5, 2, 3, 1,
*               2, 5, 6, 5, 3, 3, 3, 2, 6, 2, 3, 1,
*               2, 5, 6, 5, 3, 3, 3, 2, 5, 2, 3, 1,
*    image  =   2, 5, 5, 5, 3, 3, 3, 2, 2, 2, 3, 1,
*               3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1,
*               2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 1,
*               6, 6, 6, 6, 6, 2, 4, 4, 4, 4, 4, 1,
*               7, 3, 6, 4, 6, 2, 4, 8, 4, 8, 4, 1,
*               7, 8, 6, 6, 6, 2, 4, 4, 4, 4, 4, 1,
*               2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1
* 
* Dynamic maxtree:
* └──ID: 158, Parent: 158, Altitude: 1, NumChildren: 1,
*    ProperParts: [6, 7, 8, 9, 10, 11, 23, 35, 47, 59, 71, 83, 95, 107, 119, 131, 138, 139, 140, 141, 142, 143], ProperPartCount: 22
*    └──ID: 157, Parent: 158, Altitude: 2, NumChildren: 3,
*       ProperParts: [0, 1, 2, 3, 4, 5, 12, 19, 20, 21, 24, 31, 33, 36, 43, 45, 48, 55, 57, 60, 67, 68, 69, 84, 85, 86, 87, 88, 89, 101, 113, 125, 132, 133, 134, 135, 136, 137], ProperPartCount: 38
*       ├──ID: 152, Parent: 157, Altitude: 5, NumChildren: 1, ProperParts: [32, 56], ProperPartCount: 2
*       │  └──ID: 150, Parent: 152, Altitude: 6, NumChildren: 0, ProperParts: [44], ProperPartCount: 1
*       ├──ID: 155, Parent: 157, Altitude: 3, NumChildren: 1, ProperParts: [109], ProperPartCount: 1
*       │  └──ID: 154, Parent: 155, Altitude: 4, NumChildren: 1, ProperParts: [111], ProperPartCount: 1
*       │     └──ID: 148, Parent: 154, Altitude: 6, NumChildren: 1,
*       │        ProperParts: [96, 97, 98, 99, 100, 110, 112, 122, 123, 124], ProperPartCount: 10
*       │        └──ID: 147, Parent: 148, Altitude: 7, NumChildren: 1, ProperParts: [108, 120], ProperPartCount: 2
*       │           └──ID: 144, Parent: 147, Altitude: 8, NumChildren: 0, ProperParts: [121], ProperPartCount: 1
*       └──ID: 156, Parent: 157, Altitude: 3, NumChildren: 2,
*          ProperParts: [16, 17, 18, 22, 28, 29, 30, 34, 40, 41, 42, 46, 52, 53, 54, 58, 64, 65, 66, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 90, 91, 92, 93, 94], ProperPartCount: 36
*          ├──ID: 151, Parent: 156, Altitude: 5, NumChildren: 1,
*          │  ProperParts: [13, 14, 15, 25, 27, 37, 39, 49, 51, 61, 62, 63], ProperPartCount: 12
*          │  └──ID: 149, Parent: 151, Altitude: 6, NumChildren: 0, ProperParts: [26, 38, 50], ProperPartCount: 3
*          └──ID: 153, Parent: 156, Altitude: 4, NumChildren: 2,
*             ProperParts: [102, 103, 104, 105, 106, 114, 116, 118, 126, 127, 128, 129, 130], ProperPartCount: 13
*             ├──ID: 145, Parent: 153, Altitude: 8, NumChildren: 0, ProperParts: [117], ProperPartCount: 1
*             └──ID: 146, Parent: 153, Altitude: 8, NumChildren: 0, ProperParts: [115], ProperPartCount: 1
*/

using namespace hg;

namespace dynamic_component_tree_attribute_computers {

    namespace {

        array_1d<uint8_t> make_demo_image() {
            constexpr index_t numPixels = 144;
            uint8_t raw[numPixels] = {
                    2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1,
                    2, 5, 5, 5, 3, 3, 3, 2, 2, 2, 3, 1,
                    2, 5, 6, 5, 3, 3, 3, 2, 5, 2, 3, 1,
                    2, 5, 6, 5, 3, 3, 3, 2, 6, 2, 3, 1,
                    2, 5, 6, 5, 3, 3, 3, 2, 5, 2, 3, 1,
                    2, 5, 5, 5, 3, 3, 3, 2, 2, 2, 3, 1,
                    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1,
                    2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 1,
                    6, 6, 6, 6, 6, 2, 4, 4, 4, 4, 4, 1,
                    7, 3, 6, 4, 6, 2, 4, 8, 4, 8, 4, 1,
                    7, 8, 6, 6, 6, 2, 4, 4, 4, 4, 4, 1,
                    2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1};

            auto image = array_1d<uint8_t>::from_shape({numPixels});
            for (index_t i = 0; i < numPixels; ++i) {
                image(i) = raw[i];
            }
            return image;
        }

        template<typename value_t>
        void require_equal_buffers(const std::vector<value_t> &lhs, const std::vector<value_t> &rhs) {
            // Bounding-box attributes are scalar doubles in the current tests, so Approx keeps the helper
            // usable for width/height and diagonal-length checks with the same code path.
            REQUIRE(lhs.size() == rhs.size());
            for (size_t i = 0; i < lhs.size(); ++i) {
                REQUIRE(lhs[i] == Approx(rhs[i]));
            }
        }

    } // namespace

    TEST_CASE("dynamic component tree scalar attribute computers") {
        // Full bottom-up recomputation must recover the expected area and bounding-box measures on the
        // untouched reference maxtree.
        embedding_grid_2d embedding({12, 12});
        auto graph = get_4_adjacency_graph(embedding);
        auto image = make_demo_image();
        const auto staticTree = component_tree_max_tree(graph, image);
        detail::hierarchy::DynamicComponentTree maxtree;
        maxtree.reset(staticTree.tree);

        detail::hierarchy::DynamicComponentTreeAreaAttributeComputer<> areaComputer;
        std::vector<double> area;
        areaComputer.computeAttribute(maxtree, area);

        REQUIRE(area[(size_t) maxtree.getRoot()] == 144.0);
        REQUIRE(area[(size_t) 152] == 3.0);
        REQUIRE(area[(size_t) 150] == 1.0);

        auto pixelToPoint = [](index_t pixelId) {
            return std::pair<index_t, index_t>{pixelId / 12, pixelId % 12};
        };

        detail::hierarchy::DynamicComponentTreeBoundingBoxAttributeComputer<decltype(pixelToPoint)>
                diagonalComputer(pixelToPoint, detail::hierarchy::DynamicComponentTreeBoundingBoxMeasure::diagonal_length);
        std::vector<double> diagonal;
        diagonalComputer.computeAttribute(maxtree, diagonal);

        detail::hierarchy::DynamicComponentTreeBoundingBoxAttributeComputer<decltype(pixelToPoint)>
                widthComputer(pixelToPoint, detail::hierarchy::DynamicComponentTreeBoundingBoxMeasure::width);
        std::vector<double> width;
        widthComputer.computeAttribute(maxtree, width);

        detail::hierarchy::DynamicComponentTreeBoundingBoxAttributeComputer<decltype(pixelToPoint)>
                heightComputer(pixelToPoint, detail::hierarchy::DynamicComponentTreeBoundingBoxMeasure::height);
        std::vector<double> height;
        heightComputer.computeAttribute(maxtree, height);

        REQUIRE(width[(size_t) maxtree.getRoot()] == 12.0);
        REQUIRE(height[(size_t) maxtree.getRoot()] == 12.0);
        REQUIRE(diagonal[(size_t) maxtree.getRoot()] == Approx(std::sqrt(288.0)));

        REQUIRE(width[(size_t) 152] == 1.0);
        REQUIRE(height[(size_t) 152] == 3.0);
        REQUIRE(diagonal[(size_t) 152] == Approx(std::sqrt(10.0)));
    }

    TEST_CASE("dynamic component tree bounding-box computer matches full recomputation after a local edit") {
        // computeAttributeOnNode only refreshes one node from its current children and proper parts. After a
        // local edit, callers must therefore recompute the affected path bottom-up. This test checks that this
        // incremental refresh strategy matches a full recomputation for the bounding-box attribute.
        embedding_grid_2d embedding({12, 12});
        auto graph = get_4_adjacency_graph(embedding);
        auto image = make_demo_image();
        const auto staticTree = component_tree_max_tree(graph, image);
        detail::hierarchy::DynamicComponentTree maxtree;
        maxtree.reset(staticTree.tree);

        auto pixelToPoint = [](index_t pixelId) {
            return std::pair<index_t, index_t>{pixelId / 12, pixelId % 12};
        };

        using measure_t = detail::hierarchy::DynamicComponentTreeBoundingBoxMeasure;
        detail::hierarchy::DynamicComponentTreeBoundingBoxAttributeComputer<decltype(pixelToPoint)>
                diagonalComputer(pixelToPoint, measure_t::diagonal_length);
        detail::hierarchy::DynamicComponentTreeBoundingBoxAttributeComputer<decltype(pixelToPoint)>
                widthComputer(pixelToPoint, measure_t::width);
        detail::hierarchy::DynamicComponentTreeBoundingBoxAttributeComputer<decltype(pixelToPoint)>
                heightComputer(pixelToPoint, measure_t::height);

        std::vector<double> diagonalIncremental;
        std::vector<double> widthIncremental;
        std::vector<double> heightIncremental;
        diagonalComputer.computeAttribute(maxtree, diagonalIncremental);
        widthComputer.computeAttribute(maxtree, widthIncremental);
        heightComputer.computeAttribute(maxtree, heightIncremental);

        // Move one direct proper part from node 152 to its parent 157. This changes the source box, the
        // target box, and the root box, so the affected path is exactly 152 -> 157 -> 158.
        maxtree.moveProperPart(157, 152, 32);

        for (const index_t nodeId: std::vector<index_t>{152, 157, 158}) {
            diagonalComputer.computeAttributeOnNode(maxtree, nodeId, diagonalIncremental);
            widthComputer.computeAttributeOnNode(maxtree, nodeId, widthIncremental);
            heightComputer.computeAttributeOnNode(maxtree, nodeId, heightIncremental);
        }

        std::vector<double> diagonalFull;
        std::vector<double> widthFull;
        std::vector<double> heightFull;
        diagonalComputer.computeAttribute(maxtree, diagonalFull);
        widthComputer.computeAttribute(maxtree, widthFull);
        heightComputer.computeAttribute(maxtree, heightFull);

        require_equal_buffers(diagonalIncremental, diagonalFull);
        require_equal_buffers(widthIncremental, widthFull);
        require_equal_buffers(heightIncremental, heightFull);

        REQUIRE(widthIncremental[(size_t) 152] == 1.0);
        REQUIRE(heightIncremental[(size_t) 152] == 2.0);
        REQUIRE(diagonalIncremental[(size_t) 152] == Approx(std::sqrt(5.0)));
        REQUIRE(widthIncremental[(size_t) 157] == 11.0);
        REQUIRE(heightIncremental[(size_t) 157] == 12.0);
    }

} // namespace dynamic_component_tree_attribute_computers
