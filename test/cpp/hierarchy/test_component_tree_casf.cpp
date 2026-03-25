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
#include <cmath>
#include <limits>

#include "higra/attribute/tree_attribute.hpp"
#include "higra/hierarchy/component_tree.hpp"
#include "higra/hierarchy/component_tree_casf.hpp"
#include "higra/image/graph_image.hpp"

using namespace hg;

namespace component_tree_casf {

    namespace {

        // Returns the shared 12x12 grayscale fixture used by the functional CASF tests.
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

        // Structured synthetic image matching the benchmark generator used in the area stress test.
        array_1d<uint8_t> make_structured_benchmark_image(index_t numRows, index_t numCols) {
            const index_t numPixels = numRows * numCols;
            auto image = array_1d<uint8_t>::from_shape({(size_t) numPixels});
            for (index_t r = 0; r < numRows; ++r) {
                for (index_t c = 0; c < numCols; ++c) {
                    const index_t p = r * numCols + c;
                    const auto value = (37 * r + 19 * c + 11 * ((r / 4) % 7) + 23 * ((c / 8) % 5) + ((r * c) % 29)) % 256;
                    image(p) = (uint8_t) value;
                }
            }
            return image;
        }

        template<typename value_t>
        // Casts the shared fixture to another altitude type while preserving pixel values.
        array_1d<value_t> make_demo_image_as() {
            const auto image = make_demo_image();
            auto out = array_1d<value_t>::from_shape({image.size()});
            for (index_t i = 0; i < (index_t) image.size(); ++i) {
                out(i) = (value_t) image(i);
            }
            return out;
        }

        template<typename graph_t>
        // Builds a dynamic min-tree or max-tree from the given image for internal-selection tests.
        detail::hierarchy::DynamicComponentTree
        make_dynamic_tree_from_image(const graph_t &graph, const array_1d<uint8_t> &image, bool isMaxTree) {
            const auto staticTree = isMaxTree ? component_tree_max_tree(graph, image) : component_tree_min_tree(graph, image);
            detail::hierarchy::DynamicComponentTree tree;
            tree.reset(staticTree.tree);
            return tree;
        }

        // Tests whether `ancestorId` belongs to the path from `nodeId` to the root.
        bool is_ancestor(const detail::hierarchy::DynamicComponentTree &tree, index_t ancestorId, index_t nodeId) {
            for (auto current: tree.getPathToRootNodes(nodeId)) {
                if (current == ancestorId) {
                    return true;
                }
            }
            return false;
        }

        // Checks that a selected candidate set is maximal, non-root, and pairwise non-ancestral.
        void require_candidate_set_valid(const detail::hierarchy::DynamicComponentTree &tree,
                                         const std::vector<double> &area,
                                         const std::vector<index_t> &candidates,
                                         double threshold) {
            for (auto nodeId: candidates) {
                REQUIRE(tree.isAlive(nodeId));
                REQUIRE(!tree.isRoot(nodeId));
                REQUIRE(area[(size_t) nodeId] <= threshold);

                const auto parentId = tree.getNodeParent(nodeId);
                REQUIRE(parentId != invalid_index);
                if (!tree.isRoot(parentId)) {
                    REQUIRE(area[(size_t) parentId] > threshold);
                }
            }

            for (size_t i = 0; i < candidates.size(); ++i) {
                for (size_t j = i + 1; j < candidates.size(); ++j) {
                    REQUIRE(!is_ancestor(tree, candidates[i], candidates[j]));
                    REQUIRE(!is_ancestor(tree, candidates[j], candidates[i]));
                }
            }
        }

        template<typename altitude_t>
        // Exported trees must respect Higra's static topological ordering convention.
        void require_higra_topological_order(const tree &staticTree, const xt::xexpression<altitude_t> &xaltitude) {
            const auto &alt = xaltitude.derived_cast();
            const index_t n = (index_t) staticTree.num_vertices();
            const index_t numLeaves = (index_t) staticTree.num_leaves();

            REQUIRE((index_t) alt.size() == n);
            REQUIRE((index_t) staticTree.root() == n - 1);
            REQUIRE(staticTree.parent(n - 1) == n - 1);

            for (index_t i = 0; i < n; ++i) {
                const auto p = staticTree.parent(i);
                REQUIRE(p >= i);
                REQUIRE(p < n);
                if (i < numLeaves) {
                    REQUIRE(p >= numLeaves);
                }
                if (i != n - 1) {
                    REQUIRE(p > i);
                }
            }
        }

        template<typename altitude_t, typename graph_t>
        // Exported trees must match a fresh static component-tree rebuild from the filtered image.
        void require_export_matches_component_tree(const typename ComponentTreeCasf<altitude_t, graph_t>::ExportedTree &exported,
                                                   const graph_t &graph,
                                                   const array_1d<altitude_t> &filteredImage,
                                                   bool isMaxTree) {
            const auto expected = isMaxTree ? component_tree_max_tree(graph, filteredImage)
                                            : component_tree_min_tree(graph, filteredImage);
            require_higra_topological_order(exported.tree, exported.altitudes);
            REQUIRE((exported.tree.parents() == expected.tree.parents()));
            REQUIRE((exported.altitudes == expected.altitudes));
        }

        template<typename image_t, typename graph_t>
        // Applies one naive area-based CASF step by rebuilding the max-tree and min-tree from scratch.
        image_t apply_naive_threshold_by_area(const graph_t &graph, const image_t &image, double threshold) {
            auto maxTree = component_tree_max_tree(graph, image);
            auto maxArea = attribute_area(maxTree.tree);
            auto filtered = reconstruct_leaf_data(maxTree.tree, maxTree.altitudes, maxArea <= threshold);

            auto minTree = component_tree_min_tree(graph, filtered);
            auto minArea = attribute_area(minTree.tree);
            return reconstruct_leaf_data(minTree.tree, minTree.altitudes, minArea <= threshold);
        }

        template<typename image_t, typename graph_t>
        // Runs the full naive area-based threshold sequence used as a functional reference.
        image_t run_naive_area_sequence(const graph_t &graph, const image_t &image, const std::vector<double> &thresholds) {
            image_t current = image;
            for (double threshold: thresholds) {
                current = apply_naive_threshold_by_area(graph, current, threshold);
            }
            return current;
        }

        // Computes a static-tree bounding-box attribute with an explicit bottom-up reference implementation.
        template<typename tree_t, typename graph_t>
        auto compute_naive_bounding_box_attribute(const tree_t &tree,
                                                  const graph_t &graph,
                                                  detail::hierarchy::DynamicComponentTreeBoundingBoxMeasure measure) {
            auto treeWithChildren = tree;
            treeWithChildren.compute_children();

            std::vector<index_t> xmin((size_t) treeWithChildren.num_vertices(), std::numeric_limits<index_t>::max());
            std::vector<index_t> xmax((size_t) treeWithChildren.num_vertices(), std::numeric_limits<index_t>::lowest());
            std::vector<index_t> ymin((size_t) treeWithChildren.num_vertices(), std::numeric_limits<index_t>::max());
            std::vector<index_t> ymax((size_t) treeWithChildren.num_vertices(), std::numeric_limits<index_t>::lowest());
            std::vector<bool> empty((size_t) treeWithChildren.num_vertices(), true);
            array_1d<double> attribute = xt::zeros<double>({treeWithChildren.num_vertices()});

            for (index_t leaf = 0; leaf < (index_t) treeWithChildren.num_leaves(); ++leaf) {
                const auto point = graph.embedding().lin2grid(leaf);
                const auto y = (index_t) point[0];
                const auto x = (index_t) point[1];
                xmin[(size_t) leaf] = x;
                xmax[(size_t) leaf] = x;
                ymin[(size_t) leaf] = y;
                ymax[(size_t) leaf] = y;
                empty[(size_t) leaf] = false;
            }

            for (auto node: leaves_to_root_iterator(treeWithChildren, leaves_it::exclude)) {
                for (auto child: children_iterator(node, treeWithChildren)) {
                    if (empty[(size_t) child]) {
                        continue;
                    }
                    if (empty[(size_t) node]) {
                        xmin[(size_t) node] = xmin[(size_t) child];
                        xmax[(size_t) node] = xmax[(size_t) child];
                        ymin[(size_t) node] = ymin[(size_t) child];
                        ymax[(size_t) node] = ymax[(size_t) child];
                        empty[(size_t) node] = false;
                    } else {
                        xmin[(size_t) node] = std::min(xmin[(size_t) node], xmin[(size_t) child]);
                        xmax[(size_t) node] = std::max(xmax[(size_t) node], xmax[(size_t) child]);
                        ymin[(size_t) node] = std::min(ymin[(size_t) node], ymin[(size_t) child]);
                        ymax[(size_t) node] = std::max(ymax[(size_t) node], ymax[(size_t) child]);
                    }
                }
            }

            for (index_t node = 0; node < (index_t) treeWithChildren.num_vertices(); ++node) {
                if (empty[(size_t) node]) {
                    attribute(node) = 0.0;
                    continue;
                }
                const double width = (double) (xmax[(size_t) node] - xmin[(size_t) node] + 1);
                const double height = (double) (ymax[(size_t) node] - ymin[(size_t) node] + 1);
                switch (measure) {
                    case detail::hierarchy::DynamicComponentTreeBoundingBoxMeasure::width:
                        attribute(node) = width;
                        break;
                    case detail::hierarchy::DynamicComponentTreeBoundingBoxMeasure::height:
                        attribute(node) = height;
                        break;
                    case detail::hierarchy::DynamicComponentTreeBoundingBoxMeasure::diagonal_length:
                        attribute(node) = std::sqrt(width * width + height * height);
                        break;
                }
            }

            return attribute;
        }

        template<typename image_t, typename graph_t>
        // Applies one naive bounding-box-based CASF step by rebuilding both trees from scratch.
        image_t apply_naive_threshold_by_bounding_box(const graph_t &graph,
                                                      const image_t &image,
                                                      double threshold,
                                                      detail::hierarchy::DynamicComponentTreeBoundingBoxMeasure measure) {
            auto maxTree = component_tree_max_tree(graph, image);
            auto maxAttribute = compute_naive_bounding_box_attribute(maxTree.tree, graph, measure);
            auto filtered = reconstruct_leaf_data(maxTree.tree, maxTree.altitudes, maxAttribute <= threshold);

            auto minTree = component_tree_min_tree(graph, filtered);
            auto minAttribute = compute_naive_bounding_box_attribute(minTree.tree, graph, measure);
            return reconstruct_leaf_data(minTree.tree, minTree.altitudes, minAttribute <= threshold);
        }

        template<typename image_t, typename graph_t>
        // Runs the full naive bounding-box threshold sequence used as a functional reference.
        image_t run_naive_bounding_box_sequence(const graph_t &graph,
                                                const image_t &image,
                                                const std::vector<double> &thresholds,
                                                detail::hierarchy::DynamicComponentTreeBoundingBoxMeasure measure) {
            image_t current = image;
            for (double threshold: thresholds) {
                current = apply_naive_threshold_by_bounding_box(graph, current, threshold, measure);
            }
            return current;
        }

        // Generates increasing area thresholds spanning the image domain for the stress test.
        std::vector<double> make_area_thresholds(index_t numPixels, int numThresholds) {
            std::vector<double> thresholds;
            thresholds.reserve((size_t) numThresholds);
            for (int step = 1; step <= numThresholds; ++step) {
                const double ratio = (double) step / (double) (numThresholds + 1);
                const auto threshold = (double) std::max<index_t>(1, (index_t) std::llround(ratio * (double) numPixels));
                if (thresholds.empty() || threshold > thresholds.back()) {
                    thresholds.push_back(threshold);
                }
            }
            return thresholds;
        }

    } // namespace

    TEST_CASE("component tree CASF filter exports max-tree and min-tree in Higra static form", "[component_tree_casf]") {
        // Export must coincide with rebuilding the static min-tree and max-tree from the filtered image.
        auto graph = get_4_adjacency_implicit_graph({12, 12});
        auto image = make_demo_image();
        const std::vector<double> thresholds{15.0, 100.0};

        ComponentTreeCasf<uint8_t, decltype(graph)> casf(graph, image);
        const auto filtered = casf.filter(thresholds);
        const auto exportedMax = casf.exportMaxTree();
        const auto exportedMin = casf.exportMinTree();

        require_export_matches_component_tree<uint8_t>(exportedMax, graph, filtered, true);
        require_export_matches_component_tree<uint8_t>(exportedMin, graph, filtered, false);
    }

    TEST_CASE("component tree CASF export remains valid for all supported attribute choices", "[component_tree_casf]") {
        // Changing the selection attribute must not break the structural validity of the exported trees.
        auto graph = get_4_adjacency_implicit_graph({12, 12});
        auto image = make_demo_image();
        const std::vector<double> thresholds{3.0, 5.0};

        for (auto attribute: {ComponentTreeCasfAttribute::area,
                              ComponentTreeCasfAttribute::bounding_box_width,
                              ComponentTreeCasfAttribute::bounding_box_height,
                              ComponentTreeCasfAttribute::bounding_box_diagonal}) {
            INFO("attribute=" << (int) attribute);
            ComponentTreeCasf<uint8_t, decltype(graph)> casf(graph, image, attribute);
            const auto filtered = casf.filter(thresholds);
            const auto exportedMax = casf.exportMaxTree();
            const auto exportedMin = casf.exportMinTree();

            require_export_matches_component_tree<uint8_t>(exportedMax, graph, filtered, true);
            require_export_matches_component_tree<uint8_t>(exportedMin, graph, filtered, false);
        }
    }

    TEST_CASE("component tree CASF is deterministic across independent instances", "[component_tree_casf]") {
        // Two independent instances started from the same input must produce the same filtered state and exports.
        auto graph = get_4_adjacency_implicit_graph({12, 12});
        auto image = make_demo_image();
        const std::vector<double> thresholds{15.0, 100.0};

        ComponentTreeCasf<uint8_t, decltype(graph)> casf1(graph, image);
        ComponentTreeCasf<uint8_t, decltype(graph)> casf2(graph, image);
        const auto first = casf1.filter(thresholds);
        const auto firstMax = casf1.exportMaxTree();
        const auto firstMin = casf1.exportMinTree();

        const auto second = casf2.filter(thresholds);
        const auto secondMax = casf2.exportMaxTree();
        const auto secondMin = casf2.exportMinTree();

        REQUIRE((first == second));
        REQUIRE((firstMax.tree.parents() == secondMax.tree.parents()));
        REQUIRE((firstMax.altitudes == secondMax.altitudes));
        REQUIRE((firstMin.tree.parents() == secondMin.tree.parents()));
        REQUIRE((firstMin.altitudes == secondMin.altitudes));
    }

    TEST_CASE("component tree CASF empty threshold sequence preserves the current state and exports", "[component_tree_casf]") {
        // An empty sequence must leave the current filtered state unchanged.
        auto graph = get_4_adjacency_implicit_graph({12, 12});
        auto image = make_demo_image();
        const std::vector<double> firstThresholds{15.0, 100.0};
        const std::vector<double> emptyThresholds{};

        ComponentTreeCasf<uint8_t, decltype(graph)> casf(graph, image);
        const auto before = casf.filter(firstThresholds);
        const auto beforeMax = casf.exportMaxTree();
        const auto beforeMin = casf.exportMinTree();
        const auto after = casf.filter(emptyThresholds);
        const auto afterMax = casf.exportMaxTree();
        const auto afterMin = casf.exportMinTree();

        REQUIRE((after == before));
        REQUIRE((beforeMax.tree.parents() == afterMax.tree.parents()));
        REQUIRE((beforeMax.altitudes == afterMax.altitudes));
        REQUIRE((beforeMin.tree.parents() == afterMin.tree.parents()));
        REQUIRE((beforeMin.altitudes == afterMin.altitudes));
    }

    TEST_CASE("component tree CASF dense and sparse backends reconstruct the same filtered image", "[component_tree_casf]") {
        // Dense integral and sparse floating-point backends must agree on the filtered image.
        auto graph = get_4_adjacency_implicit_graph({12, 12});
        auto imageUInt8 = make_demo_image();
        auto imageFloat = make_demo_image_as<float>();
        const std::vector<double> thresholds{15.0, 100.0};

        ComponentTreeCasf<uint8_t, decltype(graph)> casfUInt8(graph, imageUInt8);
        ComponentTreeCasf<float, decltype(graph)> casfFloat(graph, imageFloat);
        const auto resultUInt8 = casfUInt8.filter(thresholds);
        const auto resultFloat = casfFloat.filter(thresholds);

        REQUIRE(resultUInt8.size() == resultFloat.size());
        for (index_t i = 0; i < (index_t) resultUInt8.size(); ++i) {
            REQUIRE((float) resultUInt8(i) == resultFloat(i));
        }
    }

    TEST_CASE("component tree CASF bounding-box attributes match a naive reference sequence", "[component_tree_casf]") {
        // Bounding-box-driven pruning must match a full rebuild reference for each supported measure.
        auto graph = get_4_adjacency_implicit_graph({12, 12});
        auto image = make_demo_image();

        struct Case {
            ComponentTreeCasfAttribute casfAttribute;
            detail::hierarchy::DynamicComponentTreeBoundingBoxMeasure measure;
        };

        const std::vector<double> thresholds{2.0, 4.0, 6.0};

        for (const auto &testCase: {
                Case{ComponentTreeCasfAttribute::bounding_box_width, detail::hierarchy::DynamicComponentTreeBoundingBoxMeasure::width},
                Case{ComponentTreeCasfAttribute::bounding_box_height, detail::hierarchy::DynamicComponentTreeBoundingBoxMeasure::height},
                Case{ComponentTreeCasfAttribute::bounding_box_diagonal, detail::hierarchy::DynamicComponentTreeBoundingBoxMeasure::diagonal_length}}) {
            INFO("attribute=" << (int) testCase.casfAttribute);
            const auto expected = run_naive_bounding_box_sequence(graph, image, thresholds, testCase.measure);

            ComponentTreeCasf<uint8_t, decltype(graph)> casf(graph, image, testCase.casfAttribute);
            const auto actual = casf.filter(thresholds);
            const auto exportedMax = casf.exportMaxTree();
            const auto exportedMin = casf.exportMinTree();

            REQUIRE((actual == expected));
            require_export_matches_component_tree<uint8_t>(exportedMax, graph, actual, true);
            require_export_matches_component_tree<uint8_t>(exportedMin, graph, actual, false);
        }
    }

    TEST_CASE("component tree CASF area-based selection matches the reference maxtree thresholds", "[component_tree_casf]") {
        // The BFS threshold selection must follow the same pruning-root semantics used in MorphoTreeAdjust.
        auto graph = get_4_adjacency_implicit_graph({12, 12});
        auto image = make_demo_image();
        auto maxtree = make_dynamic_tree_from_image(graph, image, true);
        detail::hierarchy::DynamicComponentTreeAreaAttributeComputer<> areaComputer;
        std::vector<double> area;
        areaComputer.computeAttribute(maxtree, area);

        using casf_t = ComponentTreeCasf<uint8_t, decltype(graph)>;
        casf_t casf(graph, image);
        REQUIRE(vectorEqual(testing::ComponentTreeCasfTestAccess::selectPruneCandidates(casf, maxtree, area, 1.0), std::vector<index_t>{150, 145, 146, 144}));
        REQUIRE(vectorEqual(testing::ComponentTreeCasfTestAccess::selectPruneCandidates(casf, maxtree, area, 14.0), std::vector<index_t>{152, 154, 149, 145, 146}));
        REQUIRE(vectorEqual(testing::ComponentTreeCasfTestAccess::selectPruneCandidates(casf, maxtree, area, 15.0), std::vector<index_t>{152, 155, 151, 153}));
        REQUIRE(vectorEqual(testing::ComponentTreeCasfTestAccess::selectPruneCandidates(casf, maxtree, area, 100.0), std::vector<index_t>{152, 155, 156}));
        REQUIRE(vectorEqual(testing::ComponentTreeCasfTestAccess::selectPruneCandidates(casf, maxtree, area, 144.0), std::vector<index_t>{157}));

        for (double threshold: {1.0, 14.0, 15.0, 100.0, 144.0}) {
            INFO("threshold=" << threshold);
            require_candidate_set_valid(maxtree, area, testing::ComponentTreeCasfTestAccess::selectPruneCandidates(casf, maxtree, area, threshold), threshold);
        }
    }

    TEST_CASE("component tree CASF area-based selection matches the reference mintree thresholds", "[component_tree_casf]") {
        // The same threshold rule must work symmetrically on the mintree.
        auto graph = get_4_adjacency_implicit_graph({12, 12});
        auto image = make_demo_image();
        auto mintree = make_dynamic_tree_from_image(graph, image, false);
        detail::hierarchy::DynamicComponentTreeAreaAttributeComputer<> areaComputer;
        std::vector<double> area;
        areaComputer.computeAttribute(mintree, area);

        using casf_t = ComponentTreeCasf<uint8_t, decltype(graph)>;
        casf_t casf(graph, image);
        REQUIRE(vectorEqual(testing::ComponentTreeCasfTestAccess::selectPruneCandidates(casf, mintree, area, 1.0), std::vector<index_t>{146, 149}));
        REQUIRE(vectorEqual(testing::ComponentTreeCasfTestAccess::selectPruneCandidates(casf, mintree, area, 40.0), std::vector<index_t>{146, 149, 144}));
        REQUIRE(vectorEqual(testing::ComponentTreeCasfTestAccess::selectPruneCandidates(casf, mintree, area, 100.0), std::vector<index_t>{146, 149, 147}));
        REQUIRE(vectorEqual(testing::ComponentTreeCasfTestAccess::selectPruneCandidates(casf, mintree, area, 144.0), std::vector<index_t>{152}));

        for (double threshold: {1.0, 40.0, 100.0, 144.0}) {
            INFO("threshold=" << threshold);
            require_candidate_set_valid(mintree, area, testing::ComponentTreeCasfTestAccess::selectPruneCandidates(casf, mintree, area, threshold), threshold);
        }
    }

    TEST_CASE("component tree CASF stress test matches the naive area sequence on the structured benchmark image", "[component_tree_casf]") {
        // Stress test the CASF correctness on the structured benchmark pattern across multiple image sizes.
        for (const index_t size: {32, 64, 128, 256, 512, 1014}) {
            INFO("size=" << size);
            auto graph = get_4_adjacency_implicit_graph({size, size});
            auto image = make_structured_benchmark_image(size, size);
            const auto thresholds = make_area_thresholds(size * size, 10);

            const auto expected = run_naive_area_sequence(graph, image, thresholds);

            ComponentTreeCasf<uint8_t, decltype(graph)> casf(graph, image);
            const auto actual = casf.filter(thresholds);
            const auto exportedMax = casf.exportMaxTree();
            const auto exportedMin = casf.exportMinTree();

            REQUIRE((actual == expected));
            require_export_matches_component_tree<uint8_t>(exportedMax, graph, actual, true);
            require_export_matches_component_tree<uint8_t>(exportedMin, graph, actual, false);
        }
    }

} // namespace component_tree_casf
