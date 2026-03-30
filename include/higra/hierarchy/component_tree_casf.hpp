#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "higra/algo/tree.hpp"
#include "higra/attribute/tree_attribute.hpp"
#include "higra/hierarchy/component_tree.hpp"
#include "higra/detail/hierarchy/dual_min_max_tree_incremental_filter.hpp"
#include "higra/detail/hierarchy/dynamic_component_tree.hpp"
#include "higra/detail/hierarchy/dynamic_component_tree_attribute_computers.hpp"

namespace hg {

    namespace testing {
        struct ComponentTreeCasfTestAccess;
    }

    /**
     * @brief Increasing attributes currently supported by the component-tree CASF.
     */
    enum class ComponentTreeCasfAttribute {
        area,
        bounding_box_width,
        bounding_box_height,
        bounding_box_diagonal
    };

    /**
     * @brief Connected alternating sequential filter driven by paired dynamic min-tree / max-tree updates.
     *
     * This class implements a connected alternating sequential filter
     * (CASF) on a fixed image-domain graph. Instead of rebuilding both trees
     * after each threshold step, it maintains a mutable min-tree / max-tree pair
     * and updates one tree incrementally when rooted subtrees are pruned from the
     * dual one.
     *
     * Supported workflow:
     *  - build the initial dynamic min-tree and max-tree from an input image;
     *  - choose one increasing attribute used to select pruning candidates
     *    (`area`, `bounding_box_width`, `bounding_box_height`, or
     *    `bounding_box_diagonal`);
     *  - run `filter(thresholds)` on one or several threshold sequences applied
     *    successively to the current state;
     *  - export the current dynamic state back to Higra's static
     *    `tree + altitudes` representation with `exportMaxTree()` and
     *    `exportMinTree()`.
     *
     * Internal state owned by this class:
     *  - the mutable dynamic min-tree and max-tree;
     *  - the node-altitude buffers associated with both trees;
     *  - one incremental attribute computer per dynamic tree and their output buffers;
     *  - the dual-tree adjustment helper used to propagate subtree pruning from
     *    one tree to the other;
     *  - temporary buffers reused to select maximal pruning candidates.
     *
     * Update semantics:
     *  - calling `filter(...)` with a non-empty threshold sequence mutates the
     *    internal dynamic trees;
     *  - later `filter(...)` calls continue from the current filtered state;
     *  - an empty threshold sequence leaves the current state unchanged.
     *
     * Graph requirements:
     *  - the graph must remain valid for the whole lifetime of the object;
     *  - `area` is supported on any graph type accepted by the component-tree
     *    builders;
     *  - bounding-box attributes additionally require a graph exposing a 2D
     *    `embedding()`.
     */
    template<typename altitude_t, typename graph_t>
    class ComponentTreeCasf {
    public:
        using tree_t = detail::hierarchy::DynamicComponentTree;
        using attribute_computer_t = detail::hierarchy::DynamicComponentTreeAttributeComputer<double>;

        /**
         * @brief Static export of one current component-tree state.
         * @details `tree` and `altitudes` follow Higra's usual static component-tree conventions.
         */
        struct ExportedTree {
            tree tree;
            array_1d<altitude_t> altitudes;
        };

    private:
        friend struct testing::ComponentTreeCasfTestAccess;

        /**
         * @brief Detects whether the graph type exposes an `embedding()` method.
         */
        template<typename g_t, typename = void>
        struct has_embedding_impl : std::false_type {
        };

        /**
         * @brief Specialization enabled when the graph type exposes an `embedding()` method.
         */
        template<typename g_t>
        struct has_embedding_impl<g_t, std::void_t<decltype(std::declval<const g_t &>().embedding())>> : std::true_type {
        };

        const graph_t *graph_ = nullptr;
        tree_t mintree_;
        tree_t maxtree_;
        std::unique_ptr<detail::hierarchy::DualMinMaxTreeIncrementalFilter<altitude_t, graph_t>> adjust_;
        std::unique_ptr<attribute_computer_t> attributeComputerMin_;
        std::unique_ptr<attribute_computer_t> attributeComputerMax_;
        std::vector<double> attributeBufferMin_;
        std::vector<double> attributeBufferMax_;
        array_1d<altitude_t> altitudeBufferMin_;
        array_1d<altitude_t> altitudeBufferMax_;
        std::vector<index_t> pruneCandidateQueue_;
        std::vector<index_t> selectedPruneCandidates_;
        ComponentTreeCasfAttribute attribute_;
        static constexpr bool graph_has_embedding = has_embedding_impl<graph_t>::value;

        /**
         * @brief Returns `true` iff the selected CASF attribute is bounding-box-based.
         */
        static bool usesBoundingBoxAttribute(ComponentTreeCasfAttribute attribute) {
            return attribute == ComponentTreeCasfAttribute::bounding_box_width ||
                   attribute == ComponentTreeCasfAttribute::bounding_box_height ||
                   attribute == ComponentTreeCasfAttribute::bounding_box_diagonal;
        }

        /**
         * @brief Creates the incremental attribute computer matching the selected CASF attribute.
         * @details Area uses a scalar area computer. Bounding-box attributes require a graph with a 2D embedding.
         */
        static std::unique_ptr<attribute_computer_t> makeAttributeComputer(const graph_t &graph, ComponentTreeCasfAttribute attribute) {
            if (!usesBoundingBoxAttribute(attribute)) {
                return std::make_unique<detail::hierarchy::DynamicComponentTreeAreaAttributeComputer<>>();
            }

            if constexpr (graph_has_embedding) {
                using embedding_t = std::remove_cv_t<std::remove_reference_t<decltype(std::declval<const graph_t &>().embedding())>>;
                static_assert(embedding_t::_dim == 2, "Bounding-box CASF attributes require a graph with a 2D embedding.");

                auto pixelToPoint = [&embedding = graph.embedding()](index_t pixelId) {
                    const auto point = embedding.lin2grid(pixelId);
                    return std::pair<index_t, index_t>{point[0], point[1]};
                };

                using pixel_to_point_t = decltype(pixelToPoint);
                auto measure = detail::hierarchy::DynamicComponentTreeBoundingBoxMeasure::diagonal_length;
                switch (attribute) {
                    case ComponentTreeCasfAttribute::bounding_box_width:
                        measure = detail::hierarchy::DynamicComponentTreeBoundingBoxMeasure::width;
                        break;
                    case ComponentTreeCasfAttribute::bounding_box_height:
                        measure = detail::hierarchy::DynamicComponentTreeBoundingBoxMeasure::height;
                        break;
                    case ComponentTreeCasfAttribute::bounding_box_diagonal:
                        measure = detail::hierarchy::DynamicComponentTreeBoundingBoxMeasure::diagonal_length;
                        break;
                    case ComponentTreeCasfAttribute::area:
                        hg_assert(false, "Bounding-box attribute computer requested from area attribute.");
                        break;
                }
                return std::make_unique<detail::hierarchy::DynamicComponentTreeBoundingBoxAttributeComputer<pixel_to_point_t>>(pixelToPoint, measure);
            } else {
                hg_assert(false, "Bounding-box CASF attributes require a graph exposing a 2D embedding.");
                return nullptr;
            }
        }

        /**
         * @brief Reconstructs the current filtered image from the dynamic tree and its node altitudes.
         * @details Each pixel is assigned the altitude of its current smallest surviving component.
         */
        template<typename altitude_buffer_t>
        static array_1d<altitude_t> reconstructImage(const tree_t &tree, const altitude_buffer_t &altitude) {
            const index_t numPixels = tree.getNumTotalProperParts();
            auto image = array_1d<altitude_t>::from_shape({(size_t) numPixels});
            for (index_t p = 0; p < numPixels; ++p) {
                const auto nodeId = tree.getSmallestComponent(p);
                hg_assert(nodeId != invalid_index, "Each pixel must map to a valid surviving component.");
                image(p) = altitude[(size_t) nodeId];
            }
            return image;
        }

        /**
         * @brief Selects maximal non-root pruning candidates whose increasing attribute is below or equal to a threshold.
         * @details Traverses the rooted hierarchy in breadth-first order, following the same selection semantics used
         * in MorphoTreeAdjust: descend while `attribute(node) > threshold`, otherwise select the current node and stop
         * descending below it. The root is never returned because `DynamicComponentTree::pruneNode` requires a non-root
         * node. Internal queue/output buffers are reused across calls and sized from the largest dynamic tree.
         */
        template<typename attribute_buffer_t, typename value_t> std::vector<index_t> selectPruneCandidates(const tree_t &tree, const attribute_buffer_t &attribute, value_t threshold) {
            pruneCandidateQueue_.clear();
            selectedPruneCandidates_.clear();
            const auto root = tree.getRoot();
            if (root == invalid_index || !tree.isAlive(root)) {
                return selectedPruneCandidates_;
            }

            pruneCandidateQueue_.push_back(root);
            std::size_t head = 0;

            while (head < pruneCandidateQueue_.size()) {
                const auto nodeId = pruneCandidateQueue_[head++];

                if (!tree.isAlive(nodeId)) {
                    continue;
                }

                const auto nodeAttribute = attribute[(size_t) nodeId];
                if (!tree.isRoot(nodeId) && nodeAttribute <= threshold) {
                    selectedPruneCandidates_.push_back(nodeId);
                    continue;
                }

                for (auto childId: tree.getChildren(nodeId)) {
                    if (tree.isAlive(childId)) {
                        pruneCandidateQueue_.push_back(childId);
                    }
                }
            }

            return selectedPruneCandidates_;
        }

        /**
         * @brief Applies one threshold step of the CASF on the current dynamic state.
         * @details The step first applies the extensive half-step (max-tree pruning, min-tree update), then the
         * anti-extensive half-step (min-tree pruning, max-tree update).
         */
        void applyFilterStep(double threshold) {
            const auto candidates1 = selectPruneCandidates(maxtree_, attributeBufferMax_, threshold);
            adjust_->pruneMaxTreeAndUpdateMinTree(candidates1);

            const auto candidates2 = selectPruneCandidates(mintree_, attributeBufferMin_, threshold);
            adjust_->pruneMinTreeAndUpdateMaxTree(candidates2);
        }

        /**
         * @brief Exports one dynamic tree state to Higra's static `tree + altitudes` representation.
         * @details Alive dynamic nodes are reindexed after the leaves while preserving the monotone topological
         * ordering expected by Higra's static `tree` structure.
         */
        template<typename altitude_buffer_t>
        ExportedTree exportTreeState(const tree_t &tree, const altitude_buffer_t &altitude) const {
            hg_assert(altitude.size() >= (size_t) tree.getGlobalIdSpaceSize(), "Altitude buffer must cover the full dynamic-tree global id space.");

            const index_t numLeaves = tree.getNumTotalProperParts();
            const index_t numAliveNodes = tree.getNumNodes();
            const index_t numVertices = numLeaves + numAliveNodes;

            auto parents = array_1d<index_t>::from_shape({(size_t) numVertices});
            auto exportedAltitude = array_1d<altitude_t>::from_shape({(size_t) numVertices});
            std::vector<index_t> oldToNew((size_t) tree.getGlobalIdSpaceSize(), invalid_index);
            std::vector<index_t> exportedNodes;
            exportedNodes.reserve((size_t) numAliveNodes);

            for (index_t p = 0; p < numLeaves; ++p) {
                oldToNew[(size_t) p] = p;
                const auto ownerId = tree.getSmallestComponent(p);
                hg_assert(ownerId != invalid_index, "Each proper part must map to a valid alive component.");
                exportedAltitude(p) = altitude[(size_t) ownerId];
            }

            bool sortAscendingAltitude = true;
            for (auto nodeId: tree.getAliveNodeIds()) {
                if (!tree.isAlive(nodeId)) {
                    continue;
                }
                exportedNodes.push_back(nodeId);
                if (!tree.isRoot(nodeId)) {
                    const auto parentId = tree.getNodeParent(nodeId);
                    hg_assert(parentId != invalid_index, "Alive exported node must have a valid parent.");
                    if (altitude[(size_t) nodeId] > altitude[(size_t) parentId]) {
                        sortAscendingAltitude = false;
                    }
                }
            }

            hg_assert((index_t) exportedNodes.size() == numAliveNodes, "Export expects the rooted dynamic tree to contain all alive nodes.");

            std::stable_sort(exportedNodes.begin(), exportedNodes.end(),
                             [&](index_t lhs, index_t rhs) {
                                 const auto altL = altitude[(size_t) lhs];
                                 const auto altR = altitude[(size_t) rhs];
                                 if (altL != altR) {
                                     return sortAscendingAltitude ? altL < altR : altL > altR;
                                 }
                                 return lhs < rhs;
                             });

            for (index_t i = 0; i < numAliveNodes; ++i) {
                const auto oldNodeId = exportedNodes[(size_t) i];
                const auto newNodeId = numLeaves + i;
                oldToNew[(size_t) oldNodeId] = newNodeId;
                exportedAltitude(newNodeId) = altitude[(size_t) oldNodeId];
            }

            for (index_t p = 0; p < numLeaves; ++p) {
                const auto ownerId = tree.getSmallestComponent(p);
                hg_assert(ownerId != invalid_index, "Each proper part must map to a valid alive component.");
                parents(p) = oldToNew[(size_t) ownerId];
            }

            for (auto oldNodeId: exportedNodes) {
                const auto newNodeId = oldToNew[(size_t) oldNodeId];
                const auto oldParentId = tree.getNodeParent(oldNodeId);
                parents(newNodeId) = oldParentId == oldNodeId ? newNodeId : oldToNew[(size_t) oldParentId];
            }

            auto exportedTree = hg::tree(parents);
            return ExportedTree{std::move(exportedTree), std::move(exportedAltitude)};
        }

    public:
        /**
         * @brief Initializes the CASF state from the input image and the chosen increasing attribute.
         * @param graph Adjacency graph of the image domain.
         * @param image Input image values on the graph vertices.
         * @param attribute Increasing attribute used to select pruning candidates.
         */
        ComponentTreeCasf(const graph_t &graph, const array_1d<altitude_t> &image, ComponentTreeCasfAttribute attribute = ComponentTreeCasfAttribute::area)
                : graph_(&graph), attributeComputerMin_(nullptr), attributeComputerMax_(nullptr), attribute_(attribute) {
            const auto minStatic = component_tree_min_tree(*graph_, image);
            const auto maxStatic = component_tree_max_tree(*graph_, image);
            altitudeBufferMin_ = std::move(minStatic.altitudes);
            altitudeBufferMax_ = std::move(maxStatic.altitudes);
            mintree_.reset(minStatic.tree);
            maxtree_.reset(maxStatic.tree);
            attributeComputerMin_ = makeAttributeComputer(*graph_, attribute_);
            attributeComputerMax_ = makeAttributeComputer(*graph_, attribute_);
            adjust_ = std::make_unique<detail::hierarchy::DualMinMaxTreeIncrementalFilter<altitude_t, graph_t>>(&mintree_, &maxtree_, *graph_);
            adjust_->setAttributeComputer(*attributeComputerMin_, *attributeComputerMax_, attributeBufferMin_, attributeBufferMax_);
            adjust_->setAltitudeBuffers(altitudeBufferMin_, altitudeBufferMax_);
            attributeComputerMin_->computeAttribute(mintree_, attributeBufferMin_);
            attributeComputerMax_->computeAttribute(maxtree_, attributeBufferMax_);
            const auto maxNodes = (std::size_t) std::max(mintree_.getNumNodes(), maxtree_.getNumNodes());
            pruneCandidateQueue_.reserve(maxNodes);
            selectedPruneCandidates_.reserve(maxNodes);
        }

        /**
         * @brief Runs the CASF on the threshold sequence and returns the filtered image.
         * @details Thresholds are applied to the current internal state. This object is therefore stateful:
         * successive non-empty calls continue filtering the result of the previous ones.
         */
        array_1d<altitude_t> filter(const std::vector<double> &thresholds) {
            for (auto threshold: thresholds) {
                applyFilterStep(threshold);
            }
            return reconstructImage(mintree_, altitudeBufferMin_);
        }

        /**
         * @brief Exports the current max-tree state to Higra's static representation.
         */
        ExportedTree exportMaxTree() const {
            return exportTreeState(maxtree_, altitudeBufferMax_);
        }

        /**
         * @brief Exports the current min-tree state to Higra's static representation.
         */
        ExportedTree exportMinTree() const {
            return exportTreeState(mintree_, altitudeBufferMin_);
        }
    };

    namespace testing {

        struct ComponentTreeCasfTestAccess {
            template<typename altitude_t, typename graph_t, typename attribute_buffer_t, typename value_t> static std::vector<index_t>
            selectPruneCandidates(ComponentTreeCasf<altitude_t, graph_t> &casf, const typename ComponentTreeCasf<altitude_t, graph_t>::tree_t &tree, const attribute_buffer_t &attribute, value_t threshold) {
                return casf.selectPruneCandidates(tree, attribute, threshold);
            }
        };

    } // namespace testing

} // namespace hg
