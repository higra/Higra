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
#include "higra/detail/hierarchy/dual_min_max_tree_incremental_filter.hpp"
#include "higra/detail/hierarchy/dynamic_component_tree.hpp"
#include "higra/detail/hierarchy/dynamic_component_tree_attribute_computers.hpp"
#include "higra/hierarchy/component_tree.hpp"
#include "higra/image/graph_image.hpp"
#include <map>

using namespace hg;

namespace dual_min_max_tree_incremental_filter {

using detail::hierarchy::testing::DualMinMaxTreeIncrementalFilterTestAccess;

namespace {

        using tree_t = detail::hierarchy::DynamicComponentTree;

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

        array_1d<float> make_demo_image_non_integral_float() {
            const auto image = make_demo_image();
            auto out = array_1d<float>::from_shape({image.size()});
            for (index_t i = 0; i < (index_t) image.size(); ++i) {
                out(i) = (float) image(i) + 0.25f;
            }
            return out;
        }

        template<typename graph_t>
        std::pair<tree_t, std::vector<uint8_t>> make_component_tree_with_altitude(const graph_t &graph,
                                                                                  const array_1d<uint8_t> &image,
                                                                                  bool isMaxTree) {
            const auto staticTree = isMaxTree ? component_tree_max_tree(graph, image) : component_tree_min_tree(graph, image);
            tree_t tree;
            tree.reset(staticTree.tree);
            return {std::move(tree), std::vector<uint8_t>(staticTree.altitudes.begin(), staticTree.altitudes.end())};
        }

        template<typename altitude_t>
        array_1d<altitude_t> reconstruct_image_from_dynamic_tree(const detail::hierarchy::DynamicComponentTree &tree,
                                                                 const std::vector<altitude_t> &altitude) {
            // Reconstructs the current image by assigning each pixel the altitude of its current smallest component.
            auto image = array_1d<altitude_t>::from_shape({(size_t) tree.getNumTotalProperParts()});
            for (index_t p = 0; p < tree.getNumTotalProperParts(); ++p) {
                const auto nodeId = tree.getSmallestComponent(p);
                REQUIRE(nodeId != invalid_index);
                image(p) = altitude[(size_t) nodeId];
            }
            return image;
        }

        template<typename altitude_t>
        void require_tree_matches_rebuilt_image_baseline(const detail::hierarchy::DynamicComponentTree &updatedTree,
                                                         const std::vector<altitude_t> &updatedAltitude,
                                                         const detail::hierarchy::DynamicComponentTree &prunedDualTree,
                                                         const std::vector<altitude_t> &prunedDualAltitude) {
            // The updated dual tree must represent exactly the same filtered image as the baseline obtained
            // by pruning the source tree and reconstructing the filtered image from that modified hierarchy.
            const auto updatedImage = reconstruct_image_from_dynamic_tree(updatedTree, updatedAltitude);
            const auto expectedImage = reconstruct_image_from_dynamic_tree(prunedDualTree, prunedDualAltitude);
            REQUIRE((updatedImage == expectedImage));
        }

        /*
         * Reference maxtree and mintree for the 12x12 demo image.
         *
         * == Dynamic maxtree ==
         * └──ID: 158, Parent: 158, Altitude: 1, NumChildren: 1,
         *    ProperParts: [6, 7, 8, 9, 10, 11, 23, 35, 47, 59, 71, 83, 95, 107, 119, 131, 138, 139, 140, 141, 142, 143],
         *    ProperPartCount: 22
         *    └──ID: 157, Parent: 158, Altitude: 2, NumChildren: 3,
         *       ProperParts: [0, 1, 2, 3, 4, 5, 12, 19, 20, 21, 24, 31, 33, 36, 43, 45, 48, 55, 57, 60, 67, 68, 69, 84, 85, 86, 87, 88, 89, 101, 113, 125, 132, 133, 134, 135, 136, 137],
         *       ProperPartCount: 38
         *       ├──ID: 152, Parent: 157, Altitude: 5, NumChildren: 1, ProperParts: [32, 56], ProperPartCount: 2
         *       │  └──ID: 150, Parent: 152, Altitude: 6, NumChildren: 0, ProperParts: [44], ProperPartCount: 1
         *       ├──ID: 155, Parent: 157, Altitude: 3, NumChildren: 1, ProperParts: [109], ProperPartCount: 1
         *       │  └──ID: 154, Parent: 155, Altitude: 4, NumChildren: 1, ProperParts: [111], ProperPartCount: 1
         *       │     └──ID: 148, Parent: 154, Altitude: 6, NumChildren: 1,
         *       │        ProperParts: [96, 97, 98, 99, 100, 110, 112, 122, 123, 124], ProperPartCount: 10
         *       │        └──ID: 147, Parent: 148, Altitude: 7, NumChildren: 1, ProperParts: [108, 120], ProperPartCount: 2
         *       │           └──ID: 144, Parent: 147, Altitude: 8, NumChildren: 0, ProperParts: [121], ProperPartCount: 1
         *       └──ID: 156, Parent: 157, Altitude: 3, NumChildren: 2,
         *          ProperParts: [16, 17, 18, 22, 28, 29, 30, 34, 40, 41, 42, 46, 52, 53, 54, 58, 64, 65, 66, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 90, 91, 92, 93, 94],
         *          ProperPartCount: 36
         *          ├──ID: 151, Parent: 156, Altitude: 5, NumChildren: 1,
         *          │  ProperParts: [13, 14, 15, 25, 27, 37, 39, 49, 51, 61, 62, 63], ProperPartCount: 12
         *          │  └──ID: 149, Parent: 151, Altitude: 6, NumChildren: 0, ProperParts: [26, 38, 50], ProperPartCount: 3
         *          └──ID: 153, Parent: 156, Altitude: 4, NumChildren: 2,
         *             ProperParts: [102, 103, 104, 105, 106, 114, 116, 118, 126, 127, 128, 129, 130], ProperPartCount: 13
         *             ├──ID: 145, Parent: 153, Altitude: 8, NumChildren: 0, ProperParts: [117], ProperPartCount: 1
         *             └──ID: 146, Parent: 153, Altitude: 8, NumChildren: 0, ProperParts: [115], ProperPartCount: 1
         *
         * == Dynamic mintree ==
         * └──ID: 153, Parent: 153, Altitude: 8, NumChildren: 1, ProperParts: [115, 117, 121], ProperPartCount: 3
         *    └──ID: 152, Parent: 153, Altitude: 7, NumChildren: 1, ProperParts: [108, 120], ProperPartCount: 2
         *       └──ID: 151, Parent: 152, Altitude: 6, NumChildren: 3,
         *          ProperParts: [26, 38, 44, 50, 96, 97, 98, 99, 100, 110, 112, 122, 123, 124], ProperPartCount: 14
         *          ├──ID: 146, Parent: 151, Altitude: 3, NumChildren: 0, ProperParts: [109], ProperPartCount: 1
         *          ├──ID: 149, Parent: 151, Altitude: 4, NumChildren: 0, ProperParts: [111], ProperPartCount: 1
         *          └──ID: 150, Parent: 151, Altitude: 5, NumChildren: 1,
         *             ProperParts: [13, 14, 15, 25, 27, 32, 37, 39, 49, 51, 56, 61, 62, 63], ProperPartCount: 14
         *             └──ID: 148, Parent: 150, Altitude: 4, NumChildren: 1,
         *                ProperParts: [102, 103, 104, 105, 106, 114, 116, 118, 126, 127, 128, 129, 130], ProperPartCount: 13
         *                └──ID: 147, Parent: 148, Altitude: 3, NumChildren: 1,
         *                   ProperParts: [16, 17, 18, 22, 28, 29, 30, 34, 40, 41, 42, 46, 52, 53, 54, 58, 64, 65, 66, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 90, 91, 92, 93, 94],
         *                   ProperPartCount: 36
         *                   └──ID: 145, Parent: 147, Altitude: 2, NumChildren: 1,
         *                      ProperParts: [0, 1, 2, 3, 4, 5, 12, 19, 20, 21, 24, 31, 33, 36, 43, 45, 48, 55, 57, 60, 67, 68, 69, 84, 85, 86, 87, 88, 89, 101, 113, 125, 132, 133, 134, 135, 136, 137],
         *                      ProperPartCount: 38
         *                      └──ID: 144, Parent: 145, Altitude: 1, NumChildren: 0,
         *                         ProperParts: [6, 7, 8, 9, 10, 11, 23, 35, 47, 59, 71, 83, 95, 107, 119, 131, 138, 139, 140, 141, 142, 143],
         *                         ProperPartCount: 22
         */

        template<typename tree_t>
        bool has_unreachable_alive_nodes(const tree_t &tree) {
            // Detects alive internal nodes left outside the rooted final hierarchy.
            std::vector<bool> reachable((size_t) tree.getGlobalIdSpaceSize(), false);
            if (tree.getRoot() != invalid_index && tree.isAlive(tree.getRoot())) {
                for (auto nodeId: tree.getNodeSubtree(tree.getRoot())) {
                    reachable[(size_t) nodeId] = true;
                }
            }

            for (auto nodeId: tree.getAliveNodeIds()) {
                if (tree.isAlive(nodeId) && !reachable[(size_t) nodeId]) {
                    return true;
                }
            }
            return false;
        }

        template<typename tree_t>
        bool area_buffers_equal(const tree_t &tree,
                                const std::vector<double> &incrementalArea,
                                const std::vector<double> &fullArea) {
            // Compares incremental and full recomputation only on alive internal nodes.
            for (auto nodeId: tree.getAliveNodeIds()) {
                if (!tree.isAlive(nodeId)) {
                    continue;
                }
                if (incrementalArea[(size_t) nodeId] != fullArea[(size_t) nodeId]) {
                    return false;
                }
            }
            return true;
        }

        template<typename tree_t>
        void require_tree_consistency(const tree_t &tree) {
            // Rechecks the same structural invariants expected from a valid adjusted tree.
            REQUIRE(tree.getRoot() != invalid_index);
            REQUIRE(tree.isAlive(tree.getRoot()));
            REQUIRE(!has_unreachable_alive_nodes(tree));

            index_t totalProperParts = 0;
            std::vector<bool> seenProperParts((size_t) tree.getNumTotalProperParts(), false);
            for (auto nodeId: tree.getAliveNodeIds()) {
                if (!tree.isAlive(nodeId)) {
                    continue;
                }

                for (auto childId: tree.getChildren(nodeId)) {
                    REQUIRE(tree.isAlive(childId));
                    REQUIRE(tree.getNodeParent(childId) == nodeId);
                    REQUIRE(tree.hasChild(nodeId, childId));
                }

                index_t countedProperParts = 0;
                for (auto p: tree.getProperParts(nodeId)) {
                    REQUIRE(tree.isProperPart(p));
                    REQUIRE(tree.getSmallestComponent(p) == nodeId);
                    REQUIRE(!seenProperParts[(size_t) p]);
                    seenProperParts[(size_t) p] = true;
                    countedProperParts++;
                }
                REQUIRE(countedProperParts == tree.getNumProperParts(nodeId));
                totalProperParts += countedProperParts;
            }

            REQUIRE(totalProperParts == tree.getNumTotalProperParts());
            REQUIRE(std::all_of(seenProperParts.begin(), seenProperParts.end(), [](bool v) { return v; }));
        }

        template<typename tree_t, typename altitude_t>
        std::map<int, index_t> collect_alive_nodes_by_level(const tree_t &tree, const std::vector<altitude_t> &altitude) {
            // Builds a compact structural signature that is stable across reruns of the same input.
            std::map<int, index_t> hist;
            for (auto nodeId: tree.getAliveNodeIds()) {
                if (!tree.isAlive(nodeId)) {
                    continue;
                }
                hist[(int) altitude[(size_t) nodeId]]++;
            }
            return hist;
        }

        template<typename tree_t>
        void require_area_consistency(const tree_t &tree,
                                      const std::vector<double> &incrementalArea,
                                      detail::hierarchy::DynamicComponentTreeAreaAttributeComputer<> &areaComputer) {
            // Recomputes area from scratch and checks it against the incremental buffer.
            std::vector<double> fullArea;
            areaComputer.computeAttribute(tree, fullArea);
            REQUIRE(area_buffers_equal(tree, incrementalArea, fullArea));
            REQUIRE(fullArea[(size_t) tree.getRoot()] == 144.0);
            REQUIRE(incrementalArea[(size_t) tree.getRoot()] == 144.0);
        }

    } // namespace

    TEST_CASE("dual min max tree incremental filter maxtree update matches the rebuild image baseline after mintree pruning", "[dual_min_max_tree_incremental_filter]") {
            // The updated maxtree must represent the same filtered image as a prune-then-rebuild baseline.
            auto graph = get_4_adjacency_implicit_graph({12, 12});
            auto image = make_demo_image();
            auto [maxtree, maxAltitude] = make_component_tree_with_altitude(graph, image, true);
            auto [mintree, minAltitude] = make_component_tree_with_altitude(graph, image, false);
            detail::hierarchy::DualMinMaxTreeIncrementalFilter<uint8_t, decltype(graph)> adjust(&mintree, &maxtree, graph);
            adjust.setAltitudeBuffers(minAltitude, maxAltitude);

            const index_t rootNodeC = mintree.getSmallestComponent(32);
            REQUIRE(rootNodeC == 150);

            DualMinMaxTreeIncrementalFilterTestAccess::updateTree(adjust, &maxtree, rootNodeC);
            mintree.pruneNode(rootNodeC);

            require_tree_matches_rebuilt_image_baseline(maxtree, maxAltitude, mintree, minAltitude);
    }

    TEST_CASE("dual min max tree incremental filter mintree update matches the rebuild image baseline after maxtree pruning", "[dual_min_max_tree_incremental_filter]") {
            // The symmetric update must represent the same filtered image as a prune-then-rebuild baseline.
            auto graph = get_4_adjacency_implicit_graph({12, 12});
            auto image = make_demo_image();
            auto [maxtree, maxAltitude] = make_component_tree_with_altitude(graph, image, true);
            auto [mintree, minAltitude] = make_component_tree_with_altitude(graph, image, false);
            detail::hierarchy::DualMinMaxTreeIncrementalFilter<uint8_t, decltype(graph)> adjust(&mintree, &maxtree, graph);
            adjust.setAltitudeBuffers(minAltitude, maxAltitude);

            const index_t rootNodeC = maxtree.getSmallestComponent(44);
            REQUIRE(rootNodeC == 150);

            DualMinMaxTreeIncrementalFilterTestAccess::updateTree(adjust, &mintree, rootNodeC);
            maxtree.pruneNode(rootNodeC);

            require_tree_matches_rebuilt_image_baseline(mintree, minAltitude, maxtree, maxAltitude);
    }

    TEST_CASE("dual min max tree incremental filter matches the rebuild image baseline across sequential mintree prunes", "[dual_min_max_tree_incremental_filter]") {
            // Reusing the same adjustment helper across multiple rounds must stay equivalent to repeated prune-then-rebuild baselines.
            auto graph = get_4_adjacency_implicit_graph({12, 12});
            auto image = make_demo_image();
            auto [maxtree, maxAltitude] = make_component_tree_with_altitude(graph, image, true);
            auto [mintree, minAltitude] = make_component_tree_with_altitude(graph, image, false);
            detail::hierarchy::DualMinMaxTreeIncrementalFilter<uint8_t, decltype(graph)> adjust(&mintree, &maxtree, graph);
            adjust.setAltitudeBuffers(minAltitude, maxAltitude);

            for (const index_t pixelId: std::vector<index_t>{32, 82}) {
                const index_t rootNodeC = mintree.getSmallestComponent(pixelId);
                INFO("pixelId=" << pixelId << ", rootNodeC=" << rootNodeC);
                REQUIRE(rootNodeC != invalid_index);
                REQUIRE(mintree.isAlive(rootNodeC));
                REQUIRE(!mintree.isRoot(rootNodeC));

                DualMinMaxTreeIncrementalFilterTestAccess::updateTree(adjust, &maxtree, rootNodeC);
                mintree.pruneNode(rootNodeC);

                require_tree_matches_rebuilt_image_baseline(maxtree, maxAltitude, mintree, minAltitude);
            }
    }

    TEST_CASE("dual min max tree incremental filter sparse backend matches the rebuild image baseline on non-integral float altitudes", "[dual_min_max_tree_incremental_filter]") {
            // The sparse backend must represent the same filtered image as a prune-then-rebuild baseline on non-integral levels.
            auto graph = get_4_adjacency_implicit_graph({12, 12});
            auto image = make_demo_image_non_integral_float();

            const auto staticMaxTree = component_tree_max_tree(graph, image);
            const auto staticMinTree = component_tree_min_tree(graph, image);
            tree_t maxtree;
            tree_t mintree;
            maxtree.reset(staticMaxTree.tree);
            mintree.reset(staticMinTree.tree);
            std::vector<float> maxAltitude(staticMaxTree.altitudes.begin(), staticMaxTree.altitudes.end());
            std::vector<float> minAltitude(staticMinTree.altitudes.begin(), staticMinTree.altitudes.end());

            detail::hierarchy::DualMinMaxTreeIncrementalFilter<float, decltype(graph)> adjust(&mintree, &maxtree, graph);
            adjust.setAltitudeBuffers(minAltitude, maxAltitude);

            const index_t rootNodeC = mintree.getSmallestComponent(32);
            REQUIRE(rootNodeC == 150);

            DualMinMaxTreeIncrementalFilterTestAccess::updateTree(adjust, &maxtree, rootNodeC);
            mintree.pruneNode(rootNodeC);

            require_tree_matches_rebuilt_image_baseline(maxtree, maxAltitude, mintree, minAltitude);
    }

    TEST_CASE("dual min max tree incremental filter keeps final tree connected and area-consistent", "[dual_min_max_tree_incremental_filter]") {
            // The subtree adjustment must preserve a valid maxtree and keep incremental area exact.
            auto graph = get_4_adjacency_implicit_graph({12, 12});
            auto image = make_demo_image();
    
            const std::vector<index_t> referencePixels = {32, 37, 63};
            for (auto pixelId: referencePixels) {
                auto [maxtree, maxAltitude] = make_component_tree_with_altitude(graph, image, true);
                auto [mintree, minAltitude] = make_component_tree_with_altitude(graph, image, false);
                detail::hierarchy::DualMinMaxTreeIncrementalFilter<uint8_t, decltype(graph)> adjust(&mintree, &maxtree, graph);
                adjust.setAltitudeBuffers(minAltitude, maxAltitude);
    
                detail::hierarchy::DynamicComponentTreeAreaAttributeComputer<> areaComputer;
                std::vector<double> areaBufferMin;
                std::vector<double> areaBufferMax;
                adjust.setAttributeComputer(areaComputer, areaBufferMin, areaBufferMax);
                areaComputer.computeAttribute(mintree, areaBufferMin);
                areaComputer.computeAttribute(maxtree, areaBufferMax);
    
                const index_t rootNodeC = mintree.getSmallestComponent(pixelId);
                REQUIRE(rootNodeC == 150);
                const index_t oldNumNodes = maxtree.getNumNodes();

                DualMinMaxTreeIncrementalFilterTestAccess::updateTree(adjust, &maxtree, rootNodeC);
    
                std::vector<double> fullArea;
                areaComputer.computeAttribute(maxtree, fullArea);
    
                INFO("pixelId=" << pixelId);
                REQUIRE(maxtree.getNumNodes() < oldNumNodes);
                REQUIRE(maxtree.getNumNodes() == 6);
                REQUIRE(maxtree.getRoot() == 155);
                REQUIRE(maxAltitude[(size_t) maxtree.getRoot()] == 3);
                REQUIRE(area_buffers_equal(maxtree, areaBufferMax, fullArea));
                REQUIRE(fullArea[(size_t) maxtree.getRoot()] == 144.0);
                REQUIRE(areaBufferMax[(size_t) maxtree.getRoot()] == 144.0);
                REQUIRE(maxtree.getNumChildren(155) == 1);
                REQUIRE(maxtree.getNumChildren(154) == 1);
                REQUIRE(maxtree.getNumChildren(149) == 2);
                REQUIRE(maxtree.getNumProperParts(149) == 137);
                REQUIRE(collect_alive_nodes_by_level(maxtree, maxAltitude) == std::map<int, index_t>{{3, 1}, {4, 1}, {6, 1}, {7, 1}, {8, 2}});
                require_tree_consistency(maxtree);
    
            }
    }

    TEST_CASE("dual min max tree incremental filter also preserves the symmetric maxtree-to-mintree case", "[dual_min_max_tree_incremental_filter]") {
            // The symmetric adjustment must reconnect the final node union under nodeCa when nodeCa survives.
            auto graph = get_4_adjacency_implicit_graph({12, 12});
            auto image = make_demo_image();
    
            auto [maxtree, maxAltitude] = make_component_tree_with_altitude(graph, image, true);
            auto [mintree, minAltitude] = make_component_tree_with_altitude(graph, image, false);
            detail::hierarchy::DualMinMaxTreeIncrementalFilter<uint8_t, decltype(graph)> adjust(&mintree, &maxtree, graph);
            adjust.setAltitudeBuffers(minAltitude, maxAltitude);
    
            detail::hierarchy::DynamicComponentTreeAreaAttributeComputer<> areaComputer;
            std::vector<double> areaBufferMin;
            std::vector<double> areaBufferMax;
            adjust.setAttributeComputer(areaComputer, areaBufferMin, areaBufferMax);
            areaComputer.computeAttribute(mintree, areaBufferMin);
            areaComputer.computeAttribute(maxtree, areaBufferMax);
    
            const index_t rootNodeC = maxtree.getSmallestComponent(44);
            REQUIRE(rootNodeC == 150);

            DualMinMaxTreeIncrementalFilterTestAccess::updateTree(adjust, &mintree, rootNodeC);
    
            std::vector<double> fullArea;
            areaComputer.computeAttribute(mintree, fullArea);
    
            REQUIRE(mintree.getNumNodes() == 10);
            REQUIRE(mintree.getRoot() == 153);
            REQUIRE(minAltitude[(size_t) mintree.getRoot()] == 8);
            REQUIRE(area_buffers_equal(mintree, areaBufferMin, fullArea));
            REQUIRE(fullArea[(size_t) mintree.getRoot()] == 144.0);
            REQUIRE(areaBufferMin[(size_t) mintree.getRoot()] == 144.0);
            REQUIRE(mintree.getNumChildren(153) == 1);
            REQUIRE(mintree.getNumChildren(152) == 1);
            REQUIRE(mintree.getNumChildren(151) == 3);
            REQUIRE(mintree.hasChild(151, 146));
            REQUIRE(mintree.hasChild(151, 149));
            REQUIRE(mintree.hasChild(151, 150));
            REQUIRE(mintree.getNumChildren(150) == 1);
            REQUIRE(mintree.hasChild(150, 148));
            REQUIRE(mintree.getNumChildren(148) == 1);
            REQUIRE(mintree.hasChild(148, 147));
            REQUIRE(mintree.getNumChildren(147) == 1);
            REQUIRE(mintree.hasChild(147, 145));
            REQUIRE(mintree.getNumChildren(145) == 1);
            REQUIRE(mintree.hasChild(145, 144));
            REQUIRE(collect_alive_nodes_by_level(mintree, minAltitude) == std::map<int, index_t>{{1, 1}, {2, 1}, {3, 2}, {4, 2}, {5, 1}, {6, 1}, {7, 1}, {8, 1}});
            require_tree_consistency(mintree);
    }

    TEST_CASE("dual min max tree incremental filter remains structurally valid on all shared mintree subtree roots", "[dual_min_max_tree_incremental_filter]") {
            // Every shared non-root mintree subtree root should keep the adjusted maxtree connected and area-consistent.
            auto graph = get_4_adjacency_implicit_graph({12, 12});
            auto image = make_demo_image();
    
            const std::vector<index_t> sharedMintRoots = {144, 145, 146, 147, 148, 149, 150, 151, 152};
            for (auto rootNodeC: sharedMintRoots) {
                auto [maxtree, maxAltitude] = make_component_tree_with_altitude(graph, image, true);
                auto [mintree, minAltitude] = make_component_tree_with_altitude(graph, image, false);
                detail::hierarchy::DualMinMaxTreeIncrementalFilter<uint8_t, decltype(graph)> adjust(&mintree, &maxtree, graph);
                adjust.setAltitudeBuffers(minAltitude, maxAltitude);
    
                detail::hierarchy::DynamicComponentTreeAreaAttributeComputer<> areaComputer;
                std::vector<double> areaBufferMin;
                std::vector<double> areaBufferMax;
                adjust.setAttributeComputer(areaComputer, areaBufferMin, areaBufferMax);
                areaComputer.computeAttribute(mintree, areaBufferMin);
                areaComputer.computeAttribute(maxtree, areaBufferMax);
    
                INFO("mint rootNodeC=" << rootNodeC);
                REQUIRE(mintree.isAlive(rootNodeC));
                REQUIRE(maxtree.isAlive(rootNodeC));
                REQUIRE(!mintree.isRoot(rootNodeC));
    
                DualMinMaxTreeIncrementalFilterTestAccess::updateTree(adjust, &maxtree, rootNodeC);

                require_area_consistency(maxtree, areaBufferMax, areaComputer);
                require_tree_consistency(maxtree);
                REQUIRE(maxtree.isAlive(maxtree.getRoot()));
            }
    }

    TEST_CASE("dual min max tree incremental filter remains structurally valid on all shared maxtree subtree roots", "[dual_min_max_tree_incremental_filter]") {
            // Every shared non-root maxtree subtree root should keep the adjusted mintree connected and area-consistent.
            auto graph = get_4_adjacency_implicit_graph({12, 12});
            auto image = make_demo_image();
    
            const std::vector<index_t> sharedMaxRoots = {144, 145, 146, 147, 148, 149, 150, 151, 152, 153};
            for (auto rootNodeC: sharedMaxRoots) {
                auto [maxtree, maxAltitude] = make_component_tree_with_altitude(graph, image, true);
                auto [mintree, minAltitude] = make_component_tree_with_altitude(graph, image, false);
                detail::hierarchy::DualMinMaxTreeIncrementalFilter<uint8_t, decltype(graph)> adjust(&mintree, &maxtree, graph);
                adjust.setAltitudeBuffers(minAltitude, maxAltitude);
    
                detail::hierarchy::DynamicComponentTreeAreaAttributeComputer<> areaComputer;
                std::vector<double> areaBufferMin;
                std::vector<double> areaBufferMax;
                adjust.setAttributeComputer(areaComputer, areaBufferMin, areaBufferMax);
                areaComputer.computeAttribute(mintree, areaBufferMin);
                areaComputer.computeAttribute(maxtree, areaBufferMax);
    
                INFO("max rootNodeC=" << rootNodeC);
                REQUIRE(maxtree.isAlive(rootNodeC));
                REQUIRE(mintree.isAlive(rootNodeC));
                REQUIRE(!maxtree.isRoot(rootNodeC));
    
                DualMinMaxTreeIncrementalFilterTestAccess::updateTree(adjust, &mintree, rootNodeC);

                require_area_consistency(mintree, areaBufferMin, areaComputer);
                require_tree_consistency(mintree);
                REQUIRE(mintree.isAlive(mintree.getRoot()));
            }
    }

    TEST_CASE("dual min max tree incremental filter rejects subtree roots that are outside complementary-tree bounds", "[dual_min_max_tree_incremental_filter]") {
            // The subtree root id belongs to the source/complementary tree, so its bounds are checked there.
            auto graph = get_4_adjacency_implicit_graph({12, 12});
            auto image = make_demo_image();
    
            auto [maxtree, maxAltitude] = make_component_tree_with_altitude(graph, image, true);
            auto [mintree, minAltitude] = make_component_tree_with_altitude(graph, image, false);
            detail::hierarchy::DualMinMaxTreeIncrementalFilter<uint8_t, decltype(graph)> adjust(&mintree, &maxtree, graph);
            adjust.setAltitudeBuffers(minAltitude, maxAltitude);
    
            const index_t outOfBounds = maxtree.getGlobalIdSpaceSize();
            bool threw = false;
            try {
                DualMinMaxTreeIncrementalFilterTestAccess::updateTree(adjust, &mintree, outOfBounds);
            } catch (const std::runtime_error &e) {
                threw = true;
                REQUIRE(std::string(e.what()).find("complementary-tree bounds") != std::string::npos);
            }
            REQUIRE(threw);
    }

    TEST_CASE("dual min max tree incremental filter rejects subtree roots that are not alive in the complementary tree", "[dual_min_max_tree_incremental_filter]") {
            // A source-tree node id may still map to a dead slot in the complementary tree; this must fail explicitly.
            auto graph = get_4_adjacency_implicit_graph({12, 12});
            auto image = make_demo_image();
    
            auto [maxtree, maxAltitude] = make_component_tree_with_altitude(graph, image, true);
            auto [mintree, minAltitude] = make_component_tree_with_altitude(graph, image, false);
            detail::hierarchy::DualMinMaxTreeIncrementalFilter<uint8_t, decltype(graph)> adjust(&mintree, &maxtree, graph);
            adjust.setAltitudeBuffers(minAltitude, maxAltitude);
    
            REQUIRE(maxtree.isAlive(144));
            REQUIRE(mintree.isAlive(144));
            mintree.pruneNode(144);
            REQUIRE(!mintree.isAlive(144));
    
            bool threw = false;
            try {
                DualMinMaxTreeIncrementalFilterTestAccess::updateTree(adjust, &maxtree, 144);
            } catch (const std::runtime_error &e) {
                threw = true;
                REQUIRE(std::string(e.what()).find("valid/alive in complementary tree") != std::string::npos);
            }
            REQUIRE(threw);
    }

} // namespace dual_min_max_tree_incremental_filter
