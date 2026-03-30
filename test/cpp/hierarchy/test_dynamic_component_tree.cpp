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
#include "higra/algo/tree.hpp"
#include "higra/detail/hierarchy/dynamic_component_tree.hpp"
#include "higra/hierarchy/common.hpp"
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

namespace dynamic_component_tree {

    namespace {


        using tree_t = detail::hierarchy::DynamicComponentTree;

        constexpr bool fail_fast_enabled =
#ifndef NDEBUG
                true;
#else
                false;
#endif

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



        template<typename graph_t>
        tree_t make_component_tree_from_image(const graph_t &graph, const array_1d<uint8_t> &image, bool isMaxTree) {
            const auto staticTree = isMaxTree ? component_tree_max_tree(graph, image) : component_tree_min_tree(graph, image);
            tree_t tree;
            tree.reset(staticTree.tree);
            return tree;
        }

        template<typename graph_t>
        std::pair<tree_t, std::vector<uint8_t>> make_component_tree_with_altitude(const graph_t &graph,
                                                                                  const array_1d<uint8_t> &image,
                                                                                  bool isMaxTree) {
            // Builds the static component tree first, then mirrors its topology and altitudes into the dynamic setup.
            const auto staticTree = isMaxTree ? component_tree_max_tree(graph, image) : component_tree_min_tree(graph, image);
            tree_t tree;
            tree.reset(staticTree.tree);
            return {std::move(tree), std::vector<uint8_t>(staticTree.altitudes.begin(), staticTree.altitudes.end())};
        }

        template<typename range_t>
        std::vector<index_t> collect_range(const range_t &range) {
            // Materializes a child range to compare order-sensitive expectations.
            std::vector<index_t> out;
            for (auto v: range) {
                out.push_back(v);
            }
            return out;
        }

        template<typename range_t>
        std::vector<index_t> collect_nodes(const range_t &range) {
            // Materializes generic node ranges used by traversal APIs after mutations.
            std::vector<index_t> out;
            for (auto v: range) {
                out.push_back(v);
            }
            return out;
        }

        template<typename tree_t>
        std::vector<index_t> collect_proper_parts(const tree_t &tree, index_t nodeId) {
            // Materializes proper parts to compare exact node ownership after mutations.
            std::vector<index_t> out;
            for (auto p: tree.getProperParts(nodeId)) {
                out.push_back(p);
            }
            return out;
        }

        template<typename tree_t>
        bool has_unreachable_alive_nodes(const tree_t &tree) {
            // Detects alive internal nodes that are no longer reachable from the current root.
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

        template<typename tree_t, typename altitude_t>
        void require_tree_consistency(const tree_t &tree, const std::vector<altitude_t> &altitude) {
            // Checks global invariants: root reachability, parent/child symmetry, and proper-part partition.
            REQUIRE(tree.getRoot() != invalid_index);
            REQUIRE(tree.isAlive(tree.getRoot()));
            REQUIRE(tree.getNodeParent(tree.getRoot()) == tree.getRoot());
            REQUIRE(!has_unreachable_alive_nodes(tree));

            index_t totalProperParts = 0;
            std::vector<bool> seenProperParts((size_t) tree.getNumTotalProperParts(), false);
            for (auto nodeId: tree.getAliveNodeIds()) {
                if (!tree.isAlive(nodeId)) {
                    continue;
                }

                index_t countedChildren = 0;
                for (auto childId: tree.getChildren(nodeId)) {
                    REQUIRE(tree.isAlive(childId));
                    REQUIRE(tree.getNodeParent(childId) == nodeId);
                    REQUIRE(tree.hasChild(nodeId, childId));
                    countedChildren++;
                }
                REQUIRE(countedChildren == tree.getNumChildren(nodeId));

                index_t countedProperParts = 0;
                for (auto p: tree.getProperParts(nodeId)) {
                    REQUIRE(tree.isProperPart(p));
                    REQUIRE(tree.getSmallestComponent(p) == nodeId);
                    REQUIRE(altitude[(size_t) tree.getSmallestComponent(p)] == altitude[(size_t) nodeId]);
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

    } // namespace

    TEST_CASE("dynamic component tree initial maxtree matches expected structural facts", "[dynamic_component_tree]") {
        // The demo image should build the reference maxtree documented in the file header.
        auto graph = get_4_adjacency_implicit_graph({12, 12});
        auto image = make_demo_image();
        {
            auto [tree, altitude] = make_component_tree_with_altitude(graph, image, true);
    
            REQUIRE(tree.getNumTotalProperParts() == 144);
            REQUIRE(tree.getGlobalIdSpaceSize() == 159);
            REQUIRE(tree.getNumNodes() == 15);
            REQUIRE(tree.getRoot() == 158);
            REQUIRE(altitude[(size_t) tree.getRoot()] == 1);
            REQUIRE(tree.getSmallestComponent(32) == 152);
            REQUIRE(tree.getSmallestComponent(44) == 150);
            REQUIRE(tree.getSmallestComponent(109) == 155);
            REQUIRE(vectorEqual(collect_range(tree.getChildren(157)), std::vector<index_t>{152, 155, 156}));
            REQUIRE(vectorEqual(collect_range(tree.getChildren(156)), std::vector<index_t>{151, 153}));
            REQUIRE(vectorSame(collect_proper_parts(tree, 152), std::vector<index_t>{32, 56}));
            REQUIRE(vectorSame(collect_proper_parts(tree, 150), std::vector<index_t>{44}));
            require_tree_consistency(tree, altitude);
        }
    }

    TEST_CASE("dynamic component tree detach attach move and removeChild preserve topology contracts", "[dynamic_component_tree]") {
        // Primitive topology operations must keep parent links and sibling order coherent.
        auto graph = get_4_adjacency_implicit_graph({12, 12});
        auto image = make_demo_image();
        {
            auto [tree, altitude] = make_component_tree_with_altitude(graph, image, true);
            require_tree_consistency(tree, altitude);
    
            tree.detachNode(151);
            REQUIRE(tree.getNodeParent(151) == 151);
            REQUIRE(!tree.hasChild(156, 151));
            REQUIRE(vectorEqual(collect_range(tree.getChildren(156)), std::vector<index_t>{153}));
    
            tree.attachNode(155, 151);
            REQUIRE(tree.getNodeParent(151) == 155);
            REQUIRE(tree.hasChild(155, 151));
            REQUIRE(vectorEqual(collect_range(tree.getChildren(155)), std::vector<index_t>{154, 151}));
    
            tree.moveNode(153, 155);
            REQUIRE(tree.getNodeParent(153) == 155);
            REQUIRE(vectorEqual(collect_range(tree.getChildren(155)), std::vector<index_t>{154, 151, 153}));
            REQUIRE(vectorEqual(collect_range(tree.getChildren(156)), std::vector<index_t>{}));
    
            tree.removeChild(155, 151, false);
            REQUIRE(tree.getNodeParent(151) == 151);
            REQUIRE(!tree.hasChild(155, 151));
            REQUIRE(vectorEqual(collect_range(tree.getChildren(155)), std::vector<index_t>{154, 153}));
    
            tree.attachNode(156, 151);
            REQUIRE(tree.getNodeParent(151) == 156);
            REQUIRE(vectorEqual(collect_range(tree.getChildren(156)), std::vector<index_t>{151}));
            require_tree_consistency(tree, altitude);
        }
    }

    TEST_CASE("dynamic component tree moveProperPart keeps both mappings consistent", "[dynamic_component_tree]") {
        // Moving one proper part must update both the pixel owner and the node proper-part lists.
        auto graph = get_4_adjacency_implicit_graph({12, 12});
        auto image = make_demo_image();
        {
            auto [tree, altitude] = make_component_tree_with_altitude(graph, image, true);
            require_tree_consistency(tree, altitude);
    
            const index_t sourceNode = tree.getSmallestComponent(32); //nodeId=152
            const index_t targetNode = tree.getSmallestComponent(44); //nodeId=150
    
            REQUIRE(sourceNode != invalid_index);
            REQUIRE(targetNode != invalid_index);
            REQUIRE(sourceNode != targetNode);
            REQUIRE(tree.getNumProperParts(sourceNode) == 2);
            REQUIRE(tree.getNumProperParts(targetNode) == 1);
            REQUIRE(tree.getSmallestComponent(32) == sourceNode);
    
            tree.moveProperPart(targetNode, sourceNode, 32);
    
            REQUIRE(tree.getSmallestComponent(32) == targetNode);
            REQUIRE(altitude[(size_t) tree.getSmallestComponent(32)] == altitude[(size_t) targetNode]);
            REQUIRE(tree.getNumProperParts(sourceNode) == 1);
            REQUIRE(tree.getNumProperParts(targetNode) == 2);
            REQUIRE(vectorSame(collect_proper_parts(tree, sourceNode), std::vector<index_t>{56}));
            REQUIRE(vectorSame(collect_proper_parts(tree, targetNode), std::vector<index_t>{44, 32}));
            require_tree_consistency(tree, altitude);
        }
    }

    TEST_CASE("dynamic component tree moveProperParts transfers all direct proper parts at once", "[dynamic_component_tree]") {
        // Moving all proper parts must preserve the global partition and empty the source node.
        auto graph = get_4_adjacency_implicit_graph({12, 12});
        auto image = make_demo_image();
        {
            auto [tree, altitude] = make_component_tree_with_altitude(graph, image, true);
            require_tree_consistency(tree, altitude);
    
            const index_t sourceNode = 152;
            const index_t targetNode = 150;
    
            tree.moveProperParts(targetNode, sourceNode);
    
            REQUIRE(tree.getNumProperParts(sourceNode) == 0);
            REQUIRE(tree.getNumProperParts(targetNode) == 3);
            REQUIRE(vectorSame(collect_proper_parts(tree, targetNode), std::vector<index_t>{44, 32, 56}));
            REQUIRE(tree.getSmallestComponent(32) == targetNode);
            REQUIRE(tree.getSmallestComponent(56) == targetNode);
            REQUIRE(altitude[(size_t) tree.getSmallestComponent(32)] == altitude[(size_t) targetNode]);
            REQUIRE(altitude[(size_t) tree.getSmallestComponent(56)] == altitude[(size_t) targetNode]);
            require_tree_consistency(tree, altitude);
        }
    }

    TEST_CASE("dynamic component tree moveChildren preserves order and parent mapping", "[dynamic_component_tree]") {
        // Moving a whole child list must preserve sibling order and redirect every parent link.
        auto graph = get_4_adjacency_implicit_graph({12, 12});
        auto image = make_demo_image();
        {
            auto [tree, altitude] = make_component_tree_with_altitude(graph, image, true);
            require_tree_consistency(tree, altitude);
    
            const index_t sourceNode = 156;
            const index_t targetNode = 155;
            REQUIRE(vectorEqual(collect_range(tree.getChildren(sourceNode)), std::vector<index_t>{151, 153}));
            REQUIRE(vectorEqual(collect_range(tree.getChildren(targetNode)), std::vector<index_t>{154}));
    
            tree.moveChildren(targetNode, sourceNode);
    
            REQUIRE(tree.getNumChildren(sourceNode) == 0);
            REQUIRE(tree.getNumChildren(targetNode) == 3);
            REQUIRE(vectorEqual(collect_range(tree.getChildren(targetNode)), std::vector<index_t>{154, 151, 153}));
            REQUIRE(tree.getNodeParent(151) == targetNode);
            REQUIRE(tree.getNodeParent(153) == targetNode);
            require_tree_consistency(tree, altitude);
        }
    }

    TEST_CASE("dynamic component tree keeps surviving node ids stable under pure topology updates", "[dynamic_component_tree]") {
        // Detach/attach/move operations must not renumber or replace surviving internal nodes.
        auto graph = get_4_adjacency_implicit_graph({12, 12});
        auto image = make_demo_image();
        {
            auto [tree, altitude] = make_component_tree_with_altitude(graph, image, true);
            require_tree_consistency(tree, altitude);

            const std::vector<index_t> trackedNodes{148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158};
            std::vector<std::vector<index_t>> trackedProperParts;
            trackedProperParts.reserve(trackedNodes.size());
            for (auto nodeId: trackedNodes) {
                REQUIRE(tree.isAlive(nodeId));
                trackedProperParts.push_back(collect_proper_parts(tree, nodeId));
            }

            tree.detachNode(151);
            tree.attachNode(155, 151);
            tree.moveNode(153, 155);
            tree.moveChildren(155, 156);

            for (size_t i = 0; i < trackedNodes.size(); ++i) {
                const auto nodeId = trackedNodes[i];
                REQUIRE(tree.isAlive(nodeId));
                REQUIRE(vectorSame(collect_proper_parts(tree, nodeId), trackedProperParts[i]));
                for (auto p: trackedProperParts[i]) {
                    REQUIRE(tree.getSmallestComponent(p) == nodeId);
                }
            }

            REQUIRE(tree.getRoot() == 158);
            REQUIRE(tree.getNodeParent(152) == 157);
            REQUIRE(tree.getNodeParent(155) == 157);
            REQUIRE(tree.getNodeParent(156) == 157);
            REQUIRE(tree.getNodeParent(151) == 155);
            REQUIRE(tree.getNodeParent(153) == 155);
            REQUIRE(tree.getNodeParent(149) == 151);
            REQUIRE(tree.getNodeParent(154) == 155);
            REQUIRE(tree.getNodeParent(148) == 154);
            require_tree_consistency(tree, altitude);
        }
    }

    TEST_CASE("dynamic component tree traversal ranges follow the mutated topology", "[dynamic_component_tree]") {
        // Subtree, descendants, path-to-root and post-order traversals must reflect topology updates.
        auto graph = get_4_adjacency_implicit_graph({12, 12});
        auto image = make_demo_image();
        {
            auto [tree, altitude] = make_component_tree_with_altitude(graph, image, true);
            require_tree_consistency(tree, altitude);
    
            tree.moveChildren(155, 156);
    
            REQUIRE(vectorEqual(collect_nodes(tree.getChildren(155)), std::vector<index_t>{154, 151, 153}));
            REQUIRE(vectorEqual(collect_nodes(tree.getNodeSubtree(155)), std::vector<index_t>{155, 154, 148, 147, 144, 151, 149, 153, 145, 146}));
            REQUIRE(vectorEqual(collect_nodes(tree.getDescendants(155)), std::vector<index_t>{154, 148, 147, 144, 151, 149, 153, 145, 146}));
            REQUIRE(vectorEqual(collect_nodes(tree.getPathToRootNodes(149)), std::vector<index_t>{149, 151, 155, 157, 158}));
            REQUIRE(vectorEqual(collect_nodes(tree.getPostOrderNodes(155)), std::vector<index_t>{144, 147, 148, 154, 149, 151, 145, 146, 153, 155}));
            require_tree_consistency(tree, altitude);
        }
    }

    TEST_CASE("dynamic component tree mergeNodeIntoParent promotes children and proper parts", "[dynamic_component_tree]") {
        // Merging an internal node into its parent must preserve child order and release the merged node.
        auto graph = get_4_adjacency_implicit_graph({12, 12});
        auto image = make_demo_image();
        {
            auto [tree, altitude] = make_component_tree_with_altitude(graph, image, true);
            require_tree_consistency(tree, altitude);
    
            const index_t mergedNode = 151;
            const index_t parentNode = tree.getNodeParent(mergedNode);
            const index_t oldNumNodes = tree.getNumNodes();
            const index_t oldParentProperParts = tree.getNumProperParts(parentNode);
    
            tree.mergeNodeIntoParent(mergedNode);
    
            REQUIRE(!tree.isAlive(mergedNode));
            REQUIRE(tree.getNumNodes() == oldNumNodes - 1);
            REQUIRE(tree.getNumProperParts(parentNode) == oldParentProperParts + 12);
            REQUIRE(vectorEqual(collect_range(tree.getChildren(parentNode)), std::vector<index_t>{153, 149}));
            REQUIRE(tree.getNodeParent(149) == parentNode);
            REQUIRE(tree.getSmallestComponent(13) == parentNode);
            REQUIRE(tree.getSmallestComponent(63) == parentNode);
            require_tree_consistency(tree, altitude);
        }
    }

    TEST_CASE("dynamic component tree pruneNode removes subtree and moves proper parts to parent", "[dynamic_component_tree]") {
        // Pruning removes all internal nodes in the subtree and transfers their proper parts to the parent.
        auto graph = get_4_adjacency_implicit_graph({12, 12});
        auto image = make_demo_image();
        {
            auto [tree, altitude] = make_component_tree_with_altitude(graph, image, true);
            require_tree_consistency(tree, altitude);
    
            const index_t prunedNode = 151;
            const index_t parentNode = tree.getNodeParent(prunedNode);
            const index_t oldNumNodes = tree.getNumNodes();
            const index_t oldParentProperParts = tree.getNumProperParts(parentNode);
    
            tree.pruneNode(prunedNode);
    
            REQUIRE(!tree.isAlive(prunedNode));
            REQUIRE(tree.getNumNodes() == oldNumNodes - 2);
            REQUIRE(tree.getNumProperParts(parentNode) == oldParentProperParts + 15);
            REQUIRE(vectorEqual(collect_range(tree.getChildren(parentNode)), std::vector<index_t>{153}));
            REQUIRE(tree.getSmallestComponent(13) == parentNode);
            REQUIRE(tree.getSmallestComponent(44) == 150);
            REQUIRE(tree.getSmallestComponent(26) == parentNode);
            require_tree_consistency(tree, altitude);
        }
    }

    TEST_CASE("dynamic component tree setRoot only changes the designated root node", "[dynamic_component_tree]") {
        // setRoot promotes one node and leaves the previous root detached from the rooted hierarchy.
        auto graph = get_4_adjacency_implicit_graph({12, 12});
        auto image = make_demo_image();
        {
            tree_t tree = make_component_tree_from_image(graph, image, true);
    
            const index_t oldRoot = tree.getRoot();
            const index_t newRoot = 157;
    
            tree.setRoot(newRoot);
    
            REQUIRE(tree.getRoot() == newRoot);
            REQUIRE(tree.getNodeParent(newRoot) == newRoot);
            REQUIRE(tree.getNodeParent(oldRoot) == oldRoot);
            REQUIRE(!tree.hasChild(oldRoot, newRoot));
            REQUIRE(has_unreachable_alive_nodes(tree));
            REQUIRE(collect_nodes(tree.getNodeSubtree(newRoot)).front() == newRoot);
            REQUIRE(tree.getNumNodes() == 15);
        }
    }

    TEST_CASE("dynamic component tree releaseNode and allocateNode reuse detached leaf slots", "[dynamic_component_tree]") {
        // Releasing a detached empty node must make its slot immediately reusable by allocateNode.
        auto graph = get_4_adjacency_implicit_graph({12, 12});
        auto image = make_demo_image();
        {
            auto [tree, altitude] = make_component_tree_with_altitude(graph, image, true);
            require_tree_consistency(tree, altitude);
    
            const index_t reusableNode = 150;
            const index_t parentNode = tree.getNodeParent(reusableNode);
            const index_t siblingOwner = 152;
            const index_t oldFreeIds = tree.getNumFreeNodeSlots();
    
            tree.moveProperPart(siblingOwner, reusableNode, 44);
            tree.removeChild(parentNode, reusableNode, false);
            tree.releaseNode(reusableNode);
    
            REQUIRE(!tree.isAlive(reusableNode));
            REQUIRE(tree.getNumFreeNodeSlots() == oldFreeIds + 1);
            REQUIRE(vectorEqual(collect_range(tree.getChildren(parentNode)), std::vector<index_t>{}));
            REQUIRE(vectorSame(collect_proper_parts(tree, siblingOwner), std::vector<index_t>{32, 56, 44}));
    
            const index_t newNode = tree.allocateNode();
            REQUIRE(newNode == reusableNode);
            REQUIRE(tree.isAlive(newNode));
            REQUIRE(tree.getNodeParent(newNode) == newNode);
            REQUIRE(tree.getNumChildren(newNode) == 0);
            REQUIRE(tree.getNumProperParts(newNode) == 0);
            tree.attachNode(parentNode, newNode);
            REQUIRE(tree.getNodeParent(newNode) == parentNode);
            REQUIRE(tree.hasChild(parentNode, newNode));
            require_tree_consistency(tree, altitude);
        }
    }

    TEST_CASE("dynamic component tree removeChild with releaseNode removes empty leaf child", "[dynamic_component_tree]") {
        // removeChild(..., true) must detach and release a leaf internal node with no proper parts.
        auto graph = get_4_adjacency_implicit_graph({12, 12});
        auto image = make_demo_image();
        {
            auto [tree, altitude] = make_component_tree_with_altitude(graph, image, true);
            require_tree_consistency(tree, altitude);
    
            const index_t parentNode = 152;
            const index_t releasedNode = 150;
            const index_t oldNumNodes = tree.getNumNodes();
            const index_t oldFreeIds = tree.getNumFreeNodeSlots();
    
            tree.moveProperPart(parentNode, releasedNode, 44);
            tree.removeChild(parentNode, releasedNode, true);
    
            REQUIRE(!tree.isAlive(releasedNode));
            REQUIRE(tree.getNumNodes() == oldNumNodes - 1);
            REQUIRE(tree.getNumFreeNodeSlots() == oldFreeIds + 1);
            REQUIRE(vectorEqual(collect_range(tree.getChildren(parentNode)), std::vector<index_t>{}));
            REQUIRE(vectorSame(collect_proper_parts(tree, parentNode), std::vector<index_t>{32, 56, 44}));
            require_tree_consistency(tree, altitude);
        }
    }

    TEST_CASE("dynamic component tree fail-fast invalidates alive-node range on node-set changes", "[dynamic_component_tree]") {
        // Alive-node iteration must fail fast if the alive node set changes after the range is created.
        auto graph = get_4_adjacency_implicit_graph({12, 12});
        auto image = make_demo_image();
        {
            tree_t tree = make_component_tree_from_image(graph, image, true);
    
            auto nodes = tree.getAliveNodeIds();
            auto it = nodes.begin();
            REQUIRE(*it == 144);

            tree.moveProperPart(152, 150, 44);
            tree.detachNode(150);
            tree.releaseNode(150);

            if constexpr (fail_fast_enabled) {
                REQUIRE_THROWS_AS(*it, std::runtime_error);
            } else {
                (void) *it;
            }
        }
    }

    TEST_CASE("dynamic component tree fail-fast invalidates topology ranges on structural changes", "[dynamic_component_tree]") {
        // Topology-based traversals must fail fast after a structural mutation.
        auto graph = get_4_adjacency_implicit_graph({12, 12});
        auto image = make_demo_image();
        {
            tree_t tree = make_component_tree_from_image(graph, image, true);
    
            auto children = tree.getChildren(156);
            auto childIt = children.begin();
            REQUIRE(*childIt == 151);
            tree.moveNode(153, 155);
            if constexpr (fail_fast_enabled) {
                REQUIRE_THROWS_AS(*childIt, std::runtime_error);
            } else {
                (void) *childIt;
            }
    
            auto subtree = tree.getNodeSubtree(157);
            auto subtreeIt = subtree.begin();
            REQUIRE(*subtreeIt == 157);
            tree.detachNode(152);
            if constexpr (fail_fast_enabled) {
                REQUIRE_THROWS_AS(++subtreeIt, std::runtime_error);
            } else {
                ++subtreeIt;
            }
    
            auto postOrder = tree.getPostOrderNodes(157);
            auto postIt = postOrder.begin();
            REQUIRE(*postIt == 144);
            tree.attachNode(157, 152);
            if constexpr (fail_fast_enabled) {
                REQUIRE_THROWS_AS(*postIt, std::runtime_error);
            } else {
                (void) *postIt;
            }
        }
    }

    TEST_CASE("dynamic component tree fail-fast invalidates proper-part ranges on ownership changes", "[dynamic_component_tree]") {
        // Proper-part iteration must fail fast after a proper-part ownership mutation.
        auto graph = get_4_adjacency_implicit_graph({12, 12});
        auto image = make_demo_image();
        {
            tree_t tree = make_component_tree_from_image(graph, image, true);

            auto parts = tree.getProperParts(152);
            auto partIt = parts.begin();
            REQUIRE(tree.getSmallestComponent(*partIt) == 152);

            tree.moveProperPart(150, 152, 32);
    
            if constexpr (fail_fast_enabled) {
                REQUIRE_THROWS_AS(*partIt, std::runtime_error);
            } else {
                (void) *partIt;
            }
        }
    }
} // namespace dynamic_component_tree
