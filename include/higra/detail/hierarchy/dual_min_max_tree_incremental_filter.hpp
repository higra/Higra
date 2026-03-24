/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Wonder Alexandre Luz Alves                              *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <type_traits>
#include <vector>

#include "higra/detail/hierarchy/dynamic_component_tree_attribute_computers.hpp"
#include "higra/graph.hpp"
#include "higra/detail/hierarchy/dynamic_component_tree.hpp"
#include "higra/utils.hpp"

#ifndef HG_COMPONENT_TREE_ADJUSTMENT_DENSE_MAX_BITS
#define HG_COMPONENT_TREE_ADJUSTMENT_DENSE_MAX_BITS 8
#endif

namespace hg::detail::hierarchy {

    namespace testing {
        // Test-only gateway used to keep the primitive update step private in production code
        // while preserving direct coverage of the core adjustment routine in unit tests.
        struct DualMinMaxTreeIncrementalFilterTestAccess;
    }

    /**
     * @brief Incremental updater for paired dynamic min-tree / max-tree hierarchies.
     * 
     * This class implements the update-rather-than-rebuild strategy introduced
     * in [1,2] and used for efficient connected alternating sequential filters.
     * It handles the structural update of one component tree when the dual tree
     * undergoes a subtree-removal operation induced by an extensive or
     * anti-extensive connected operator.
     *
     * More precisely, when a rooted subtree is removed from one hierarchy, this
     * helper updates the dual hierarchy in place so that both trees remain
     * consistent with the same filtered image. The update is performed by
     * collecting valid adjacent seeds around the induced proper-part set `C`,
     * climbing each relevant ancestor chain once, and merging nodes level by
     * level instead of rebuilding the whole dual tree from scratch.
     *
     * Public usage model:
     *  - construct the helper from a dynamic min-tree / max-tree pair and the
     *    shared adjacency graph;
     *  - optionally register an incremental attribute computer and the
     *    attribute buffers to be refreshed after local edits;
     *  - register the altitude buffers used to query node levels during the
     *    adjustment;
     *  - call either `pruneMaxTreeAndUpdateMinTree(...)` or
     *    `pruneMinTreeAndUpdateMaxTree(...)`.
     *
     * State managed by this class:
     *  - two mutable dynamic component trees (`mintree`, `maxtree`);
     *  - one immutable adjacency graph shared by both trees;
     *  - one incremental attribute computer and two external attribute
     *    buffers, used to keep tree attributes synchronized after local
     *    topology edits.
     *
     * Ownership and lifetime:
     *  - this class does not own the trees, graph, altitude buffers, or
     *    attribute buffers;
     *  - callers must keep these objects alive for the whole lifetime of the
     *    filter object;
     *  - the external buffers must be indexed on the global id space of the
     *    associated `DynamicComponentTree`.
     *
     * Internal notation used by the adjustment routine follows [2]:
     *  - `nodeCa`: implementation-side representative of the node `C_a^-`
     *    (equivalently `C_a^+` in Remark 7), i.e. the extremal node of the
     *    updated tree that contains the set `C` at level `a`;
     *  - `b`: altitude value of the parent of the removed subtree in the
     *    primal hierarchy.
     *
     * Backend selection for `mergeNodesByLevel`:
     *  - dense array-based buckets for small integral altitude domains
     *    (up to `HG_COMPONENT_TREE_ADJUSTMENT_DENSE_MAX_BITS` bits),
     *    using an order-preserving dense index for signed types;
     *  - sparse ordered-map buckets for larger integral domains and floating-point altitudes.
     *
     * The class does not own the altitude or attribute buffers. Callers are
     * responsible for keeping these buffers alive and indexed on the global id
     * space of the associated `DynamicComponentTree`.
     *
     * 
     * References:
     * 
     * [1] Wonder Alves, Nicolas Passat, Dênnis José da Silva, Alexandre
     * Morimitsu, Ronaldo F. Hashimoto. "Efficient connected alternating
     * sequential filters based on component trees." International Conference on
     * Discrete Geometry and Mathematical Morphology (DGMM), Nov. 2025,
     * Groningen, Netherlands.
     *
     * [2] Wonder Alves, Nicolas Passat, Dênnis José da Silva, Alexandre
     * Morimitsu, Ronaldo F. Hashimoto. "Component tree: Update rather than
     * rebuild." Journal of Mathematical Imaging and Vision, 2026.
     */
    template<typename altitude_t, typename graph_t>
    class DualMinMaxTreeIncrementalFilter {
    private:
        using tree_t = DynamicComponentTree;
        // Friend access is restricted to the test helper so wrappers remain the only
        // production entry points while unit tests can still exercise updateTree directly.
        friend struct testing::DualMinMaxTreeIncrementalFilterTestAccess;

        /**
         * @brief O(1)-reset mark set based on generation stamps.
         * @details Entries are considered marked when their stored generation
         * matches the current one.
         */
        class GenerationStampSet {
        public:
            using gen_t = uint32_t;

        private:
            std::vector<gen_t> stamp;
            gen_t cur = 1;

        public:
            /**
             * @brief Creates an empty generation-stamped mark set.
             */
            GenerationStampSet() = default;

            /**
             * @brief Creates a mark set pre-sized for `n` possible indices.
             */
            explicit GenerationStampSet(size_t n): stamp(n, 0) {}

            /**
             * @brief Resizes the mark set and clears all logical marks.
             */
            void resize(size_t n) {
                stamp.assign(n, 0);
                cur = 1;
            }

            /**
             * @brief Marks one index in the current generation.
             */
            void mark(size_t idx) noexcept {
                stamp[idx] = cur;
            }

            /**
             * @brief Tests whether one index is marked in the current generation.
             */
            bool isMarked(size_t idx) const noexcept {
                return stamp[idx] == cur;
            }

            /**
             * @brief Clears the set in O(1) amortized time.
             * @details The method advances the active generation. If the
             * generation counter overflows, the underlying storage is physically
             * zeroed and the generation is restarted at 1.
             */
            void resetAll() {
                if (++cur == 0) {
                    std::fill(stamp.begin(), stamp.end(), 0);
                    cur = 1;
                }
            }
        };

        /**
         * @brief Returns `true` iff the altitude type should use dense buckets.
         * @details Dense buckets are enabled only for small integral altitude
         * domains. Larger integral domains and floating-point types use the
         * sparse backend.
         */
        static constexpr bool usesDenseLevels() {
            if constexpr (std::is_integral_v<altitude_t>) {
                // Only integral altitude types have a finite bit-width usable for dense buckets.
                using unsigned_altitude_t = std::make_unsigned_t<altitude_t>;
                return std::numeric_limits<unsigned_altitude_t>::digits <= HG_COMPONENT_TREE_ADJUSTMENT_DENSE_MAX_BITS;
            } else {
                return false;
            }
        }

        /**
         * @brief True iff `mergeNodesByLevel` uses the dense bucket backend.
         */
        static constexpr bool use_dense_levels = usesDenseLevels();

        /**
         * @brief Collection of merged and frontier nodes grouped by altitude.
         * @details The storage backend of `mergeNodesByLevelStorage_` is chosen
         * at compile time. Small integral altitude domains use a dense array of
         * buckets, while larger integral and floating-point domains use a sparse
         * ordered map.
         */
        class MergedNodesCollection {
        private:
            template<bool dense, typename altitude_type>
            struct merge_nodes_storage_selector;

            template<typename altitude_type>
            struct merge_nodes_storage_selector<true, altitude_type> {
                static constexpr std::size_t domain_size = static_cast<std::size_t>(static_cast<long long>(std::numeric_limits<altitude_type>::max()) - static_cast<long long>(std::numeric_limits<altitude_type>::lowest()) + 1);
                using type = std::array<std::vector<index_t>, domain_size>;
            };

            template<typename altitude_type>
            struct merge_nodes_storage_selector<false, altitude_type> {
                using type = std::map<altitude_type, std::vector<index_t>>;
            };

            using merge_nodes_storage_t = typename merge_nodes_storage_selector<use_dense_levels, altitude_t>::type;

            merge_nodes_storage_t mergeNodesByLevelStorage_;
            std::vector<altitude_t> mergeLevels_;
            std::vector<index_t> frontierNodesAboveB_;
            GenerationStampSet collectedNodeMarks_;
            GenerationStampSet mergeBucketNodeMarks_;
            GenerationStampSet adjacentSeedMarks_;
            std::size_t maxBucketSize_ = 0;
            int currentMergeLevelIndex_ = 0;
            bool isMaxtree_ = false;

        public:
            static std::size_t denseBucketIndex(altitude_t level) {
                if constexpr (std::is_signed_v<altitude_t>) {
                    return static_cast<std::size_t>(
                            static_cast<long long>(level) -
                            static_cast<long long>(std::numeric_limits<altitude_t>::lowest()));
                } else {
                    return static_cast<std::size_t>(level);
                }
            }

            /**
             * @brief Creates a merged-node collection sized for `maxNodes`.
             */
            explicit MergedNodesCollection(index_t maxNodes = 0): collectedNodeMarks_(std::max<index_t>(maxNodes, 0)),
                                                                   mergeBucketNodeMarks_(std::max<index_t>(maxNodes, 0)),
                                                                   adjacentSeedMarks_(std::max<index_t>(maxNodes, 0)) {
            }

            /**
             * @brief Resets the transient state of one adjustment step.
             * @param isMaxtree `true` for max-tree updates, `false` for min-tree updates.
             */
            void resetCollection(bool isMaxtree) {
                isMaxtree_ = isMaxtree;
                for (auto level: mergeLevels_) {
                    if constexpr (use_dense_levels) {
                        // Dense backend reuses one bucket per altitude value and clears
                        // only the levels that were active in the previous step.
                        mergeNodesByLevelStorage_[denseBucketIndex(level)].clear();
                    } else {
                        // Sparse backend stores only touched levels, so only those
                        // buckets need to be cleared here.
                        auto it = mergeNodesByLevelStorage_.find(level);
                        if (it != mergeNodesByLevelStorage_.end()) {
                            it->second.clear();
                        }
                    }
                }
                mergeLevels_.clear();
                frontierNodesAboveB_.clear();
                collectedNodeMarks_.resetAll();
                mergeBucketNodeMarks_.resetAll();
                adjacentSeedMarks_.resetAll();
                maxBucketSize_ = 0;
                currentMergeLevelIndex_ = 0;
            }

            /**
             * @brief Returns the bucket associated with one altitude value.
             */
            std::vector<index_t> &getMergedNodes(const altitude_t &level) {
                if constexpr (use_dense_levels) {
                    return mergeNodesByLevelStorage_[denseBucketIndex(level)];
                } else {
                    return mergeNodesByLevelStorage_[level];
                }
            }

            /**
             * @brief Returns the frontier roots collected above `b`.
             * @details Each stored node is the root of the first branch that
             * leaves the merge interval while climbing from a valid adjacent seed.
             */
            std::vector<index_t> &getFrontierNodesAboveB() {
                return frontierNodesAboveB_;
            }

            /**
             * @brief Returns the largest bucket observed in the current build.
             * @details This value is used to reserve the sweep worklist.
             */
            std::size_t getMaxBucketSize() const {
                return maxBucketSize_;
            }

            /**
             * @brief Marks one adjacent seed at most once in the current step.
             * @return `true` if the seed was accepted now, `false` if it had already been seen.
             */
            bool markAdjacentSeed(index_t nodeId) {
                if (adjacentSeedMarks_.isMarked(nodeId)) {
                    return false;
                }
                adjacentSeedMarks_.mark(nodeId);
                return true;
            }

            /**
             * @brief Registers one frontier root above `b` only once.
             */
            void addFrontierNodeAboveB(index_t nodeId) {
                if (!collectedNodeMarks_.isMarked(nodeId)) {
                    frontierNodesAboveB_.push_back(nodeId);
                    collectedNodeMarks_.mark(nodeId);
                }
            }

            /**
             * @brief Inserts one node into the bucket of its own altitude.
             * @details Each visited node is registered only in the bucket of its
             * own level, without materializing the full ancestor path in advance.
             */
            void addMergeNode(const altitude_t *altitude, index_t nodeId) {
                if (collectedNodeMarks_.isMarked(nodeId)) {
                    return;
                }
                auto &bucket = getMergedNodes(altitude[nodeId]);
                bucket.push_back(nodeId);
                maxBucketSize_ = std::max(maxBucketSize_, bucket.size());
                collectedNodeMarks_.mark(nodeId);
                mergeBucketNodeMarks_.mark(nodeId);
            }

            /**
             * @brief Tests whether a node was inserted in a merge bucket.
             */
            bool isMergeNode(index_t nodeId) const {
                return mergeBucketNodeMarks_.isMarked(nodeId);
            }

            /**
             * @brief Builds the ordered list of active levels and returns the first one.
             * @details The order matches the sweep direction of the current tree:
             * descending levels for max-tree updates and ascending levels for min-tree updates.
             */
            altitude_t firstMergeLevel() {
                mergeLevels_.clear();
                if constexpr (use_dense_levels) {
                    // Dense backend scans the discrete altitude domain and keeps only
                    // the non-empty levels of the current step.
                    for (std::size_t i = 0; i < mergeNodesByLevelStorage_.size(); ++i) {
                        if (!mergeNodesByLevelStorage_[i].empty()) {
                            mergeLevels_.push_back((altitude_t) i);
                        }
                    }
                } else {
                    // Sparse backend iterates only over levels that were instantiated.
                    mergeLevels_.reserve(mergeNodesByLevelStorage_.size());
                    for (const auto &entry: mergeNodesByLevelStorage_) {
                        if (!entry.second.empty()) {
                            mergeLevels_.push_back(entry.first);
                        }
                    }
                }

                if (mergeLevels_.empty()) {
                    return altitude_t{};
                }

                currentMergeLevelIndex_ = isMaxtree_ ? (int) mergeLevels_.size() - 1 : 0;
                return mergeLevels_[currentMergeLevelIndex_];
            }

            /**
             * @brief Returns `true` while the current level iterator is valid.
             */
            bool hasMergeLevel() const {
                return !mergeLevels_.empty() && currentMergeLevelIndex_ >= 0 && currentMergeLevelIndex_ < (int) mergeLevels_.size();
            }

            /**
             * @brief Advances the ordered level iterator and returns the next level.
             */
            altitude_t nextMergeLevel() {
                currentMergeLevelIndex_ = isMaxtree_ ? currentMergeLevelIndex_ - 1 : currentMergeLevelIndex_ + 1;
                if (!hasMergeLevel()) {
                    return altitude_t{};
                }
                return mergeLevels_[currentMergeLevelIndex_];
            }
        };

    private:
        // Dynamic primal/dual trees and shared domain adjacency.
        tree_t *mintree_ = nullptr;
        tree_t *maxtree_ = nullptr;
        const graph_t *graph_ = nullptr;

        // Transient state of the current adjustment step.
        MergedNodesCollection mergeNodesByLevel_;
        GenerationStampSet removedMarks_;
        GenerationStampSet pixelsInCMarks_;
        GenerationStampSet climbedNodeMarks_;
        std::vector<index_t> properPartSetC_;
        std::vector<index_t> nodesPendingRemoval_;
        std::vector<index_t> removedNodesPendingAbsorption_;

        // Incremental attribute computation and external altitude buffers.
        const DynamicComponentTreeAttributeComputer<double> *attributeComputer_ = nullptr;
        std::vector<double> *attributeBufferMin_ = nullptr;
        std::vector<double> *attributeBufferMax_ = nullptr;
        const altitude_t *altitudeBufferMinData_ = nullptr;
        const altitude_t *altitudeBufferMaxData_ = nullptr;
        std::size_t altitudeBufferMinSize_ = 0;
        std::size_t altitudeBufferMaxSize_ = 0;

        /**
         * @brief Detaches a node from its parent, optionally releasing it.
         * @details Delegates to `DynamicComponentTree::removeChild`, preserving
         * the root case and ignoring nodes with invalid parents.
         */
        void disconnect(tree_t *tree, index_t nodeId, bool releaseNode) {
            hg_assert(tree != nullptr, "disconnect: tree must not be null.");
            if (tree->isRoot(nodeId)) {
                return;
            }
            const index_t parentId = tree->getNodeParent(nodeId);
            if (parentId == invalid_index || parentId == nodeId) {
                return;
            }
            tree->removeChild(parentId, nodeId, releaseNode);
        }

        /**
         * @brief Returns the attribute buffer associated with one tree.
         * @return `nullptr` when no incremental attribute computer is configured.
         */
        std::vector<double> *getAttributeBuffer(tree_t *tree) const {
            if (attributeComputer_ == nullptr) {
                return nullptr;
            }
            return tree == maxtree_ ? attributeBufferMax_ : attributeBufferMin_;
        }

        /**
         * @brief Returns the altitude buffer associated with one tree.
         * @return `nullptr` when `tree` is null.
         */
        const altitude_t *getAltitudeBufferData(const tree_t *tree) const {
            if (tree == nullptr) {
                return nullptr;
            }
            return tree == maxtree_ ? altitudeBufferMaxData_ : altitudeBufferMinData_;
        }

        /**
         * @brief Returns the size of the altitude buffer associated with one tree.
         */
        std::size_t getAltitudeBufferSize(const tree_t *tree) const {
            if (tree == nullptr) {
                return 0;
            }
            return tree == maxtree_ ? altitudeBufferMaxSize_ : altitudeBufferMinSize_;
        }

        /**
         * @brief Returns the altitude of one node from the configured buffer.
         * @details The access is validated against the stored buffer size.
         */
        inline altitude_t nodeAltitude(const tree_t *tree, index_t nodeId) const {
            const auto *buffer = getAltitudeBufferData(tree);
            const auto bufferSize = getAltitudeBufferSize(tree);
            hg_assert(buffer != nullptr, "Altitude buffer must be configured before querying node levels.");
            hg_assert(nodeId >= 0 && (std::size_t) nodeId < bufferSize, "Node id must be inside the configured altitude buffer.");
            return buffer[nodeId];
        }

        /**
         * @brief Recomputes the incremental attribute of one node after a local edit.
         * @details The recomputation uses the registered attribute computer and
         * updates the buffer associated with the edited tree in place.
         */
        void computeAttributeOnTreeNode(tree_t *tree, index_t nodeId) {
            if (attributeComputer_ == nullptr || tree == nullptr || nodeId == invalid_index || !tree->isAlive(nodeId)) {
                return;
            }
            auto *buffer = getAttributeBuffer(tree);
            hg_assert(buffer != nullptr, "Attribute computer configured without a valid buffer.");
            attributeComputer_->computeAttributeOnNode(*tree, nodeId, *buffer);
        }

        /**
         * @brief Moves the removed proper-part set onto the current union node.
         * @details Any node left without direct proper parts is marked in
         * `removedMarks_` so it can be contracted later in the step.
         */
        void moveSelectedProperPartsToNode(tree_t *dualTree, index_t unionNode, const std::vector<index_t> &properPartSetC) {
            for (auto pixelId: properPartSetC) {
                const index_t ownerId = dualTree->getSmallestComponent(pixelId);
                if (ownerId == invalid_index || ownerId == unionNode) {
                    continue;
                }

                dualTree->moveProperPart(unionNode, ownerId, pixelId);

                if (dualTree->isAlive(ownerId) && dualTree->getNumProperParts(ownerId) == 0 && ownerId != unionNode && !removedMarks_.isMarked(ownerId)) {
                    removedMarks_.mark(ownerId);
                    removedNodesPendingAbsorption_.push_back(ownerId);
                }
            }
        }

        /**
         * @brief Tests whether a marked node can still be absorbed.
         * @details A removable node must still exist, remain alive, stay marked
         * in `removedMarks_`, and have no direct proper parts.
         */
        bool canAbsorbRemovedNode(tree_t *dualTree, index_t nodeId) const {
            return dualTree != nullptr && nodeId != invalid_index && dualTree->isNode(nodeId) &&
                   dualTree->isAlive(nodeId) && removedMarks_.isMarked(nodeId) && dualTree->getNumProperParts(nodeId) == 0;
        }

        /**
         * @brief Absorbs one removed non-root node into its parent.
         * @details Children and proper parts are moved to `parentId`, then the
         * parent attribute is recomputed. If the parent also becomes empty, it is
         * returned so the contraction can propagate upward.
         * @return The parent re-enqueued for absorption, or `invalid_index` when
         * no further propagation is required.
         */
        index_t absorbRemovedNonRootNode(tree_t *dualTree, index_t removedNodeId) {
            const index_t parentId = dualTree->getNodeParent(removedNodeId);
            if (parentId == invalid_index || parentId == removedNodeId || !dualTree->isAlive(parentId)) {
                return invalid_index;
            }

            dualTree->moveChildren(parentId, removedNodeId);
            dualTree->moveProperParts(parentId, removedNodeId);
            disconnect(dualTree, removedNodeId, true);
            computeAttributeOnTreeNode(dualTree, parentId);

            if (dualTree->isAlive(parentId) && dualTree->getNumProperParts(parentId) == 0) {
                removedMarks_.mark(parentId);
                return parentId;
            }

            return invalid_index;
        }

        /**
         * @brief Absorbs one removed root node by promoting a surviving child.
         * @details Among the remaining direct children, the method selects the
         * representative compatible with the altitude order of the current tree,
         * reattaches the others below it, and releases the removed root.
         */
        void absorbRemovedRootNode(tree_t *dualTree, index_t removedNodeId) {
            const bool isMaxtree = dualTree == maxtree_;
            const index_t firstChild = dualTree->getFirstChild(removedNodeId);
            if (firstChild == invalid_index) {
                return;
            }

            index_t newRoot = firstChild;
            for (auto childId = firstChild; childId != invalid_index; childId = dualTree->getNextSibling(childId)) {
                if ((isMaxtree && nodeAltitude(dualTree, childId) < nodeAltitude(dualTree, newRoot)) || (!isMaxtree && nodeAltitude(dualTree, childId) > nodeAltitude(dualTree, newRoot))) {
                    newRoot = childId;
                }
            }

            for (auto childId = firstChild; childId != invalid_index;) {
                const index_t next = dualTree->getNextSibling(childId);
                if (childId != newRoot && !dualTree->hasChild(newRoot, childId)) {
                    dualTree->detachNode(childId);
                    dualTree->attachNode(newRoot, childId);
                }
                childId = next;
            }

            dualTree->setRoot(newRoot);
            dualTree->releaseNode(removedNodeId);
            computeAttributeOnTreeNode(dualTree, newRoot);
        }

        /**
         * @brief Contracts, in post-order, nodes that remained alive but empty.
         * @details This pass runs after the level sweep, when the local topology
         * is already stable. The explicit stack visits only nodes that are still
         * marked and still empty, skipping obsolete seeds.
         */
        void absorbRemovedNodes(tree_t *dualTree, const std::vector<index_t> &removedNodeIds) {
            struct Frame {
                index_t nodeId = invalid_index;
                index_t nextChildId = invalid_index;
            };

            if (dualTree == nullptr || removedNodeIds.empty()) {
                return;
            }

            std::vector<Frame> stack;
            stack.reserve(std::max<std::size_t>(64, removedNodeIds.size()));
            const auto makeFrame = [dualTree](index_t nodeId) {
                return Frame{nodeId, dualTree->getFirstChild(nodeId)};
            };
            for (auto it = removedNodeIds.rbegin(); it != removedNodeIds.rend(); ++it) {
                if (*it != invalid_index) {
                    stack.push_back(makeFrame(*it));
                }
            }

            while (!stack.empty()) {
                Frame &frame = stack.back();
                index_t currentNodeId = invalid_index;

                if (!canAbsorbRemovedNode(dualTree, frame.nodeId)) {
                    stack.pop_back();
                } else if (frame.nextChildId != invalid_index) {
                    const index_t childId = frame.nextChildId;
                    frame.nextChildId = dualTree->getNextSibling(childId);
                    stack.push_back(makeFrame(childId));
                } else {
                    currentNodeId = frame.nodeId;
                    stack.pop_back();
                }

                if (canAbsorbRemovedNode(dualTree, currentNodeId)) {
                    if (dualTree->isRoot(currentNodeId)) {
                        absorbRemovedRootNode(dualTree, currentNodeId);
                    } else {
                        const index_t parentId = absorbRemovedNonRootNode(dualTree, currentNodeId);
                        if (parentId != invalid_index) {
                            stack.push_back(makeFrame(parentId));
                        }
                    }
                }
            }
        }

        /**
         * @brief Normalizes one child branch of a removed root to its surviving representative.
         * @details While the top of the branch remains marked in `removedMarks_`
         * and has no proper parts, the best child is promoted directly under `rootId`.
         * This exposes the representative that will actually survive the closure step.
         */
        index_t collapseRemovedRootBranch(tree_t *dualTree, index_t rootId, index_t childId) {
            hg_assert(dualTree != nullptr, "dualTree must not be null.");
            hg_assert(rootId != invalid_index, "rootId must be valid.");
            const bool isMaxtree = dualTree == maxtree_;

            index_t current = childId;
            while (current != invalid_index && dualTree->isNode(current) && dualTree->isAlive(current) && dualTree->getNodeParent(current) == rootId && removedMarks_.isMarked(current) && dualTree->getNumProperParts(current) == 0) {
                const index_t firstGrandchild = dualTree->getFirstChild(current);
                if (firstGrandchild == invalid_index) {
                    break;
                }

                index_t promoted = firstGrandchild;
                for (auto grandchildId = firstGrandchild; grandchildId != invalid_index; grandchildId = dualTree->getNextSibling(grandchildId)) {
                    if ((isMaxtree && nodeAltitude(dualTree, grandchildId) < nodeAltitude(dualTree, promoted)) || (!isMaxtree && nodeAltitude(dualTree, grandchildId) > nodeAltitude(dualTree, promoted))) {
                        promoted = grandchildId;
                    }
                }

                if (!dualTree->isRoot(promoted)) {
                    disconnect(dualTree, promoted, false);
                }
                dualTree->attachNode(rootId, promoted);

                for (auto grandchildId = dualTree->getFirstChild(current); grandchildId != invalid_index;) {
                    const index_t next = dualTree->getNextSibling(grandchildId);
                    if (grandchildId != promoted && !dualTree->hasChild(promoted, grandchildId)) {
                        if (!dualTree->isRoot(grandchildId)) {
                            disconnect(dualTree, grandchildId, false);
                        }
                        dualTree->attachNode(promoted, grandchildId);
                    }
                    grandchildId = next;
                }

                disconnect(dualTree, current, true);
                computeAttributeOnTreeNode(dualTree, promoted);
                current = promoted;
            }

            return current;
        }

        /**
         * @brief Finalizes the reconnection between `nodeCa`, `finalUnionNode`, and removed nodes.
         * @details First resolves the final topological position of the union node
         * produced by the level sweep. Then contracts, in post-order, the nodes
         * that remained alive but empty until the end of the step.
         */
        void finalizeUpdateTreeAndContractRemovedNodes(tree_t *dualTree, index_t nodeCa, index_t finalUnionNode) {
            if (dualTree == nullptr) {
                return;
            }

            const bool isMaxtree = dualTree == maxtree_;

            if (finalUnionNode != invalid_index && dualTree->isAlive(finalUnionNode)) {
                if (removedMarks_.isMarked(nodeCa)) {
                    // If `nodeCa` was emptied, the final union node must occupy its
                    // topological position in the updated hierarchy.
                    if (!dualTree->isRoot(nodeCa)) {
                        const index_t nodeCaParentId = dualTree->getNodeParent(nodeCa);

                        if (!dualTree->isRoot(finalUnionNode)) {
                            this->disconnect(dualTree, finalUnionNode, false);
                        }
                        dualTree->attachNode(nodeCaParentId, finalUnionNode);

                        for (auto childId = dualTree->getFirstChild(nodeCa); childId != invalid_index;) {
                            const index_t next = dualTree->getNextSibling(childId);
                            if (childId != finalUnionNode && !dualTree->hasChild(finalUnionNode, childId)) {
                                if (!dualTree->isRoot(childId)) {
                                    this->disconnect(dualTree, childId, false);
                                }
                                dualTree->attachNode(finalUnionNode, childId);
                            }
                            childId = next;
                        }

                        this->disconnect(dualTree, nodeCa, true);
                        computeAttributeOnTreeNode(dualTree, finalUnionNode);
                        computeAttributeOnTreeNode(dualTree, nodeCaParentId);
                    } else {
                        // When `nodeCa` is the root, each child branch is reduced to
                        // its immediate surviving representative before choosing the new root.
                        index_t survivingFinalUnionNode = finalUnionNode;

                        for (auto childId = dualTree->getFirstChild(nodeCa); childId != invalid_index;) {
                            const index_t next = dualTree->getNextSibling(childId);
                            const index_t normalizedChild = collapseRemovedRootBranch(dualTree, nodeCa, childId);
                            if (childId == finalUnionNode && normalizedChild != invalid_index) {
                                survivingFinalUnionNode = normalizedChild;
                            }
                            childId = next;
                        }

                        index_t candidateRootId = survivingFinalUnionNode;
                        // The new root must respect the altitude order of the current tree.
                        for (auto childId = dualTree->getFirstChild(nodeCa); childId != invalid_index; childId = dualTree->getNextSibling(childId)) {
                            if ((isMaxtree && nodeAltitude(dualTree, childId) < nodeAltitude(dualTree, candidateRootId)) ||
                                (!isMaxtree && nodeAltitude(dualTree, childId) > nodeAltitude(dualTree, candidateRootId))) {
                                candidateRootId = childId;
                            }
                        }

                        if (candidateRootId != survivingFinalUnionNode) {
                            if (!dualTree->isRoot(survivingFinalUnionNode)) {
                                this->disconnect(dualTree, survivingFinalUnionNode, false);
                            }
                            dualTree->attachNode(candidateRootId, survivingFinalUnionNode);
                        }

                        for (auto childId = dualTree->getFirstChild(nodeCa); childId != invalid_index;) {
                            const index_t next = dualTree->getNextSibling(childId);
                            if (childId != candidateRootId && !dualTree->hasChild(candidateRootId, childId)) {
                                if (!dualTree->isRoot(childId)) {
                                    this->disconnect(dualTree, childId, false);
                                }
                                dualTree->attachNode(candidateRootId, childId);
                            }
                            childId = next;
                        }

                        dualTree->setRoot(candidateRootId);
                        dualTree->releaseNode(nodeCa);
                        computeAttributeOnTreeNode(dualTree, candidateRootId);
                    }
                } else {
                    // If `nodeCa` survived, the final union node becomes its child.
                    if (!dualTree->isRoot(finalUnionNode)) {
                        this->disconnect(dualTree, finalUnionNode, false);
                    }
                    dualTree->attachNode(nodeCa, finalUnionNode);
                    computeAttributeOnTreeNode(dualTree, nodeCa);
                }
            }

            absorbRemovedNodes(dualTree, removedNodesPendingAbsorption_);
        }

        /**
         * @brief Returns the primal hierarchy for one update direction.
         */
        tree_t *getPrimalTree(bool isMaxtree) {
            return isMaxtree ? mintree_ : maxtree_;
        }

        /**
         * @brief Reattaches under `targetNodeId` only the direct children of `sourceNodeId` outside `Γ[a,b]`.
         * @details Children still marked as belonging to the merge interval are
         * detached and left isolated until their own sweep level is processed.
         */
        void reattachOutsideIntervalChildren(tree_t *tree, index_t targetNodeId, index_t sourceNodeId) {
            hg_assert(tree != nullptr, "tree must not be null.");
            if (sourceNodeId == targetNodeId) {
                for (auto childId = tree->getFirstChild(sourceNodeId); childId != invalid_index;) {
                    const index_t next = tree->getNextSibling(childId);
                    if (mergeNodesByLevel_.isMergeNode(childId)) {
                        tree->detachNode(childId);
                    }
                    childId = next;
                }
                return;
            }

            for (auto childId = tree->getFirstChild(sourceNodeId); childId != invalid_index;) {
                const index_t next = tree->getNextSibling(childId);
                if (mergeNodesByLevel_.isMergeNode(childId)) {
                    tree->detachNode(childId);
                }
                childId = next;
            }

            tree->moveChildren(targetNodeId, sourceNodeId);
        }

        /**
         * @brief Builds `mergeNodesByLevel_` and `frontierNodesAboveB_` for one adjustment step.
         * @details The method scans graph neighbors of the proper-part set `C`,
         * filters valid adjacent seeds, and climbs each relevant ancestor chain
         * at most once. Each visited node is inserted either in the bucket of its
         * own level or, when above `b`, in `frontierNodesAboveB_`.
         */
        void buildMergedAndNestedCollections(const tree_t &tree, const altitude_t *altitude, const std::vector<index_t> &properPartSetC, index_t nodeCa, altitude_t b, bool isMaxtree) {
            mergeNodesByLevel_.resetCollection(isMaxtree);
            hg_assert(nodeCa != invalid_index, "nodeCa must map to a valid smallest component.");
            climbedNodeMarks_.resetAll();
            const altitude_t altitudeCa = altitude[nodeCa];

            // For each proper part of `C`, collect valid adjacent seeds and climb
            // only once through each relevant ancestor chain.
            for (auto p: properPartSetC) {
                for (auto q: adjacent_vertex_iterator(p, *graph_)) {
                    if (pixelsInCMarks_.isMarked(q)) {
                        continue; // Neighbors internal to `C` do not generate adjacent seeds.
                    }

                    const index_t nodeQ = tree.getSmallestComponent(q);
                    if (nodeQ == invalid_index) {
                        continue; // Pixels without a live component do not enter the collection.
                    }

                    const altitude_t altitudeQ = altitude[nodeQ];
                    const bool validSeed = (isMaxtree && altitudeQ >= altitudeCa) || (!isMaxtree && altitudeQ <= altitudeCa);
                    if (!validSeed) {
                        continue; // Seeds outside the valid interval do not participate in the merge.
                    }

                    if (mergeNodesByLevel_.markAdjacentSeed(nodeQ)) {
                        index_t nodeSubtree = nodeQ;
                        index_t n = nodeQ;
                        while (n != invalid_index && tree.isAlive(n) && !climbedNodeMarks_.isMarked(n)) {
                            const altitude_t levelCurrent = altitude[n];
                            if (!((isMaxtree && levelCurrent >= altitudeCa) || (!isMaxtree && levelCurrent <= altitudeCa))) {
                                break; // The climb left the level interval induced by `C`.
                            }

                            climbedNodeMarks_.mark(n);
                            nodeSubtree = n;

                            if ((isMaxtree && levelCurrent <= b) || (!isMaxtree && levelCurrent >= b)) {
                                mergeNodesByLevel_.addMergeNode(altitude, nodeSubtree);
                            } else {
                                auto parentId = tree.getNodeParent(nodeSubtree);
                                if (parentId == nodeSubtree) {
                                    parentId = invalid_index;
                                }
                                if (!(parentId != invalid_index && ((isMaxtree && altitude[parentId] > b) || (!isMaxtree && altitude[parentId] < b)))) {
                                    mergeNodesByLevel_.addFrontierNodeAboveB(nodeSubtree);
                                }
                            }

                            const auto parentId = tree.getNodeParent(n);
                            if (parentId == n) {
                                break; // Reached the structural top of this path.
                            }
                            n = parentId;
                        }
                    }
                }
            }
        }


        /**
         * @brief Updates one hierarchy after removing a rooted subtree in the primal hierarchy.
         * @details The method:
         * 1. collects the proper parts of the removed subtree in the primal tree;
         * 2. finds the representative of `C_a^-` and the altitude interval to sweep;
         * 3. builds the merge collections;
         * 4. merges nodes level by level;
         * 5. reconnects the final union node in the updated hierarchy.
         */
        void updateTree(tree_t *dualTree, index_t subtreeRoot) {
            hg_assert(dualTree != nullptr, "updateTree: dualTree must not be null.");
            hg_assert(subtreeRoot != invalid_index, "updateTree: subtreeRoot is invalid.");

            const bool isMaxtree = dualTree == maxtree_;
            tree_t *primalTree = getPrimalTree(isMaxtree);
            
            hg_assert(primalTree != nullptr, "updateTree: primal tree is null.");
            hg_assert(subtreeRoot >= 0 && subtreeRoot < primalTree->getGlobalIdSpaceSize(), "updateTree: subtreeRoot must be in primal-tree bounds.");
            hg_assert(primalTree->isNode(subtreeRoot) && primalTree->isAlive(subtreeRoot), "updateTree: subtreeRoot must be valid/alive in primal tree.");

            const index_t subtreeParentId = primalTree->getNodeParent(subtreeRoot);
            hg_assert(subtreeParentId != invalid_index && subtreeParentId != subtreeRoot, "updateTree: subtreeRoot must be attached in primal tree.");
            const altitude_t b = nodeAltitude(primalTree, subtreeParentId);

            index_t nodeCa = invalid_index;
            altitude_t altitudeCa = altitude_t{};

            // Phase 1: collect the set `C` in the primal tree and locate `nodeCa`
            // as the extreme representative of `C` in the updated tree.
            properPartSetC_.clear();
            properPartSetC_.reserve(64);
            pixelsInCMarks_.resetAll();
            for (auto subtreeNodeId: primalTree->getNodeSubtree(subtreeRoot)) {
                for (auto p: primalTree->getProperParts(subtreeNodeId)) {
                    properPartSetC_.push_back(p);
                    pixelsInCMarks_.mark(p);
                
                    const index_t nodeP = dualTree->getSmallestComponent(p);
                    if (nodeP == invalid_index) {
                        continue; // Proper parts without a live dual component do not contribute to `nodeCa`.
                    }
                    const altitude_t altitudeP = nodeAltitude(dualTree, nodeP);
                    if (nodeCa == invalid_index || ((isMaxtree && altitudeP < altitudeCa) || (!isMaxtree && altitudeP > altitudeCa))) {
                        altitudeCa = altitudeP;
                        nodeCa = nodeP;
                    }
                }
            }
            if (properPartSetC_.empty()) {
                return;
            }
            hg_assert(nodeCa != invalid_index, "updateTree: invalid C_a representative.");
            hg_assert(!properPartSetC_.empty(), "updateTree: expected non-empty proper parts in removed subtree.");

            
            const auto *targetAltitude = getAltitudeBufferData(dualTree);
            hg_assert(targetAltitude != nullptr, "updateTree: altitude buffer must be configured for target tree.");

            // Phase 2: build the merge buckets by level and the frontier roots above `b`.
            buildMergedAndNestedCollections(*dualTree, targetAltitude, properPartSetC_, nodeCa, b, isMaxtree);

            // Phase 3: sweep the active levels between `b` and `a`, merging the
            // buckets and propagating the union node created at each level.
            altitude_t currentMergeLevel = mergeNodesByLevel_.firstMergeLevel();
            index_t currentUnionNode;
            index_t previousLevelUnionNode = invalid_index;
            removedMarks_.resetAll();
            removedNodesPendingAbsorption_.clear();
            nodesPendingRemoval_.reserve(mergeNodesByLevel_.getMaxBucketSize());
            while (mergeNodesByLevel_.hasMergeLevel() && ((isMaxtree && currentMergeLevel > altitudeCa) || (!isMaxtree && currentMergeLevel < altitudeCa))) {
                auto &nodesAtCurrentLevel = mergeNodesByLevel_.getMergedNodes(currentMergeLevel);
                currentUnionNode = invalid_index;
                nodesPendingRemoval_.clear();

                for (auto nodeId: nodesAtCurrentLevel) {
                    if (!dualTree->isAlive(nodeId)) {
                        continue; // Nodes already removed before this level are ignored.
                    }

                    if (currentUnionNode == invalid_index) {
                        if (removedMarks_.isMarked(nodeId)) {
                            // If the first candidate at this level was already emptied
                            // when moving proper parts, keep it pending until an actual
                            // union node appears at this level.
                            nodesPendingRemoval_.push_back(nodeId);
                            continue;
                        }
                        // The first surviving node of the level becomes the current union node.
                        currentUnionNode = nodeId;
                        this->disconnect(dualTree, currentUnionNode, false);
                        this->reattachOutsideIntervalChildren(dualTree, currentUnionNode, currentUnionNode);

                        // Nodes that were already empty at this level are absorbed into
                        // the first effective union node that survives the sweep.
                        for (auto pendingNodeId: nodesPendingRemoval_) {
                            this->reattachOutsideIntervalChildren(dualTree, currentUnionNode, pendingNodeId);
                            dualTree->moveProperParts(currentUnionNode, pendingNodeId);
                            this->disconnect(dualTree, pendingNodeId, true);
                        }
                        nodesPendingRemoval_.clear();
                        continue;
                    }

                    this->reattachOutsideIntervalChildren(dualTree, currentUnionNode, nodeId);
                    dualTree->moveProperParts(currentUnionNode, nodeId);
                    this->disconnect(dualTree, nodeId, true);
                }

                if (currentUnionNode == invalid_index) {
                    // If no union node survived at this level, finish collapsing the
                    // pending empty nodes into their current parents and move on.
                    for (auto nodeId: nodesPendingRemoval_) {
                        if (!dualTree->isAlive(nodeId) || dualTree->isRoot(nodeId)) {
                            continue;
                        }
                        const auto parentId = dualTree->getNodeParent(nodeId);
                        dualTree->moveChildren(parentId, nodeId);
                        dualTree->moveProperParts(parentId, nodeId);
                        this->disconnect(dualTree, nodeId, true);
                    }
                    currentMergeLevel = mergeNodesByLevel_.nextMergeLevel();
                    currentUnionNode = previousLevelUnionNode;
                    continue;
                }

                if (currentMergeLevel == b) {
                    // When the sweep reaches `b`, move the set `C` onto the current
                    // union node and reconnect the frontier branches above `b`.
                    this->moveSelectedProperPartsToNode(dualTree, currentUnionNode, properPartSetC_);
                    for (auto nodeId: mergeNodesByLevel_.getFrontierNodesAboveB()) {
                        this->disconnect(dualTree, nodeId, false);
                        dualTree->attachNode(currentUnionNode, nodeId);
                    }
                }

                if (previousLevelUnionNode != invalid_index) {
                    // The union node produced at the previous level becomes a child of
                    // the current level union node when both still survive.
                    if (dualTree->isAlive(previousLevelUnionNode) && !dualTree->hasChild(currentUnionNode, previousLevelUnionNode)) {
                        if (!dualTree->isRoot(previousLevelUnionNode)) {
                            this->disconnect(dualTree, previousLevelUnionNode, false);
                        }
                        dualTree->attachNode(currentUnionNode, previousLevelUnionNode);
                    }
                }

                computeAttributeOnTreeNode(dualTree, currentUnionNode);

                previousLevelUnionNode = currentUnionNode;
                currentMergeLevel = mergeNodesByLevel_.nextMergeLevel();
            }

            finalizeUpdateTreeAndContractRemovedNodes(dualTree, nodeCa, previousLevelUnionNode);
        }


    public:
        /**
         * @brief Returns `true` iff this instantiation uses the dense level backend.
         * @details Exposed mainly so tests and local benchmarks can report which backend is active.
         * @return `true` when altitude buckets are stored in a compile-time dense
         * array, `false` when an ordered sparse map is used instead.
         */
        static constexpr bool usesDenseLevelBackend() {
            return use_dense_levels;
        }

        /**
         * @brief Returns the compile-time bit threshold used to enable the dense backend.
         * @details Integral altitude domains whose effective bit width does not
         * exceed this threshold use the dense bucket backend.
         */
        static constexpr int denseLevelBackendMaxBits() {
            return HG_COMPONENT_TREE_ADJUSTMENT_DENSE_MAX_BITS;
        }

        /**
         * @brief Creates the adjustment helper from a dynamic min-tree / max-tree pair.
         * @param mintree Dynamic min-tree.
         * @param maxtree Dynamic max-tree.
         * @param graph Graph shared by both hierarchies.
         * @pre `mintree`, `maxtree`, and `graph` must remain valid for the whole
         * lifetime of this object.
         * @pre Both trees must represent the same image domain and be consistent
         * with the same adjacency graph.
         */
        DualMinMaxTreeIncrementalFilter(tree_t *mintree, tree_t *maxtree, const graph_t &graph): mintree_(mintree),
                                                                                                  maxtree_(maxtree),
                                                                                                  graph_(&graph),
                                                                                                  mergeNodesByLevel_(std::max(mintree ? mintree->getGlobalIdSpaceSize() : 0, maxtree ? maxtree->getGlobalIdSpaceSize() : 0)),
                                                                                                  removedMarks_(std::max(mintree ? mintree->getGlobalIdSpaceSize() : 0, maxtree ? maxtree->getGlobalIdSpaceSize() : 0)),
                                                                                                  pixelsInCMarks_(std::max(mintree ? mintree->getNumTotalProperParts() : 0, maxtree ? maxtree->getNumTotalProperParts() : 0)),
                                                                                                  climbedNodeMarks_(std::max(mintree ? mintree->getGlobalIdSpaceSize() : 0, maxtree ? maxtree->getGlobalIdSpaceSize() : 0)) {
            hg_assert(mintree_ != nullptr, "mintree must not be null.");
            hg_assert(maxtree_ != nullptr, "maxtree must not be null.");
            hg_assert(graph_ != nullptr, "graph must not be null.");
        }

        /**
         * @brief Defaulted destructor.
         */
        virtual ~DualMinMaxTreeIncrementalFilter() = default;

        /**
         * @brief Registers the incremental attribute computer and its output buffers.
         * @details Buffer sizing and lifetime remain the caller responsibility.
         * Each buffer must cover the global id space of the corresponding tree.
         * Once configured, local structural edits performed by the incremental
         * update refresh the touched node attributes through this computer.
         * @param computer Incremental attribute computer shared by both trees.
         * @param bufferMin Output buffer associated with the min-tree.
         * @param bufferMax Output buffer associated with the max-tree.
         */
        void setAttributeComputer(const DynamicComponentTreeAttributeComputer<double> &computer, std::vector<double> &bufferMin, std::vector<double> &bufferMax) {
            attributeComputer_ = &computer;
            attributeBufferMin_ = &bufferMin;
            attributeBufferMax_ = &bufferMax;
        }

        /**
         * @brief Registers external altitude buffers for the min-tree and max-tree.
         * @details The adjustment logic reads node levels exclusively from these buffers.
         * The buffers are not copied: the filter stores only raw pointers to
         * their data and the corresponding sizes.
         * @param bufferMin Altitude buffer associated with the min-tree.
         * @param bufferMax Altitude buffer associated with the max-tree.
         * @pre Both buffers must keep stable storage for the whole period in
         * which the filter may query node altitudes.
         */
        template<typename altitude_buffer_min_t, typename altitude_buffer_max_t>
        void setAltitudeBuffers(const altitude_buffer_min_t &bufferMin, const altitude_buffer_max_t &bufferMax) {
            altitudeBufferMinData_ = bufferMin.data();
            altitudeBufferMaxData_ = bufferMax.data();
            altitudeBufferMinSize_ = bufferMin.size();
            altitudeBufferMaxSize_ = bufferMax.size();
        }
    
        /**
         * @brief Prunes rooted subtrees from the max-tree and updates the min-tree.
         * @details Each valid entry of `nodesToPrune` is processed independently.
         * Invalid ids, dead nodes, and the current root of the max-tree are
         * ignored. For each remaining node, the min-tree is updated first and
         * the corresponding subtree is then pruned from the max-tree.
         * @param nodesToPrune Root ids of max-tree subtrees to be pruned.
         */
        void pruneMaxTreeAndUpdateMinTree(const std::vector<index_t> &nodesToPrune) {
            hg_assert(mintree_ != nullptr, "pruneMaxTreeAndUpdateMinTree: mintree must not be null.");
            hg_assert(maxtree_ != nullptr, "pruneMaxTreeAndUpdateMinTree: maxtree must not be null.");
            for (auto rootSubtree: nodesToPrune) {
                if (rootSubtree == invalid_index || rootSubtree == maxtree_->getRoot() || !maxtree_->isNode(rootSubtree) || !maxtree_->isAlive(rootSubtree)) {
                    continue;
                }
                updateTree(mintree_, rootSubtree);
                maxtree_->pruneNode(rootSubtree);
            }
        }

        /**
         * @brief Prunes rooted subtrees from the min-tree and updates the max-tree.
         * @details Each valid entry of `nodesToPrune` is processed independently.
         * Invalid ids, dead nodes, and the current root of the min-tree are
         * ignored. For each remaining node, the max-tree is updated first and
         * the corresponding subtree is then pruned from the min-tree.
         * @param nodesToPrune Root ids of min-tree subtrees to be pruned.
         */
        void pruneMinTreeAndUpdateMaxTree(const std::vector<index_t> &nodesToPrune) {
            hg_assert(mintree_ != nullptr, "pruneMinTreeAndUpdateMaxTree: mintree must not be null.");
            hg_assert(maxtree_ != nullptr, "pruneMinTreeAndUpdateMaxTree: maxtree must not be null.");
            for (auto rootSubtree: nodesToPrune) {
                if (rootSubtree == invalid_index || rootSubtree == mintree_->getRoot() || !mintree_->isNode(rootSubtree) || !mintree_->isAlive(rootSubtree)) {
                    continue;
                }
                updateTree(maxtree_, rootSubtree);
                mintree_->pruneNode(rootSubtree);
            }
        }
    };

    namespace testing {
        struct DualMinMaxTreeIncrementalFilterTestAccess {
            template<typename altitude_t, typename graph_t>
            static void updateTree(DualMinMaxTreeIncrementalFilter<altitude_t, graph_t> &adjust, DynamicComponentTree *tree, index_t subtreeRoot) {
                adjust.updateTree(tree, subtreeRoot);
            }
        };
    }

} // namespace hg::detail::hierarchy
