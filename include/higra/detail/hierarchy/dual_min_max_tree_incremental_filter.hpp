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
     * This class implements the update-rather-than-rebuild strategy introduced in [1,2]
     * and used for efficient connected alternating sequential filters. It handles the
     * structural updates of one component tree when the dual tree undergoes a subtree removal
     * operation induced by an extensive or anti-extensive connected operator.
     *
     * More precisely, when a rooted subtree is removed from one hierarchy, this
     * helper updates the dual hierarchy in place so that both trees remain
     * consistent with the same filtered image. The update is performed by
     * collecting the affected nodes over an altitude interval and merging them
     * level by level instead of rebuilding the whole dual tree from scratch.
     *
     * State managed by this class:
     *  - two mutable dynamic component trees (`mintree`, `maxtree`);
     *  - one immutable adjacency graph shared by both trees;
     *  - one incremental attribute computer and two external attribute
     *    buffers, used to keep tree attributes synchronized after local
     *    topology edits.
     *
     * Internal notation used by the adjustment routine follows [2]:
     *  - `nodeCa`: implementation-side representative of the node `C_a^-`
     *    (equivalently `C_a^+` in Remark 7), i.e. the extremal node of the
     *    updated tree that contains the set `C` at level `a`;
     *  - `b`: altitude value of the parent of the removed subtree in the
     *    complementary hierarchy.
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
         * @brief Collection of merged and nested nodes grouped by altitude.
         * @details The storage backend of `mergeNodesByLevelStorage_` is selected at compile time:
         *  - dense array of buckets for small integral altitude domains, with
         *    an order-preserving dense index for signed types;
         *  - sparse ordered map for large integral or floating-point domains.
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
            std::vector<index_t> adjacentNodes_;
            std::vector<index_t> frontierNodesAboveB_;
            GenerationStampSet visited_;
            GenerationStampSet visitedAdj_;
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
            explicit MergedNodesCollection(index_t maxNodes = 0): visited_(std::max<index_t>(maxNodes, 0)), visitedAdj_(std::max<index_t>(maxNodes, 0)) {
            }

            /**
             * @brief Resets all transient state for a new adjustment step.
             * @param isMaxtree True for max-tree updates, false for min-tree updates.
             */
            void resetCollection(bool isMaxtree) {
                isMaxtree_ = isMaxtree;
                for (auto level: mergeLevels_) {
                    if constexpr (use_dense_levels) {
                        // Dense backend stores one preallocated bucket per altitude value, using
                        // an order-preserving dense index for signed altitude domains.
                        mergeNodesByLevelStorage_[denseBucketIndex(level)].clear();
                    } else {
                        // Sparse backend stores only encountered levels in an ordered map.
                        auto it = mergeNodesByLevelStorage_.find(level);
                        if (it != mergeNodesByLevelStorage_.end()) {
                            it->second.clear();
                        }
                    }
                }
                mergeLevels_.clear();
                adjacentNodes_.clear();
                frontierNodesAboveB_.clear();
                visited_.resetAll();
                visitedAdj_.resetAll();
                maxBucketSize_ = 0;
                currentMergeLevelIndex_ = 0;
            }

            /**
             * @brief Returns the bucket associated with one altitude value.
             */
            std::vector<index_t> &getMergedNodes(const altitude_t &level) {
                if constexpr (use_dense_levels) {
                    // Dense backend resolves the bucket by an order-preserving dense index.
                    return mergeNodesByLevelStorage_[denseBucketIndex(level)];
                } else {
                    // Sparse backend lazily creates/accesses the bucket for this altitude.
                    return mergeNodesByLevelStorage_[level];
                }
            }

            /**
             * @brief Returns the current list of graph-adjacent nodes.
             * @details These nodes are adjacent, in the image-domain graph, to
             * the proper parts of the removed dual subtree. They are the
             * implementation-side representatives of the nodes connected to `C`
             * in the theoretical construction.
             */
            std::vector<index_t> &getAdjacentNodes() {
                return adjacentNodes_;
            }

            /**
             * @brief Returns the current frontier-node roots stored for `frontierNodesAboveB`.
             * @details Each stored node is the root of a subtree associated
             * with adjacent nodes lying beyond `b`.
             */
            std::vector<index_t> &getFrontierNodesAboveB() {
                return frontierNodesAboveB_;
            }

            /**
             * @brief Returns the largest bucket size seen during the current build.
             */
            std::size_t getMaxBucketSize() const {
                return maxBucketSize_;
            }

            /**
             * @brief Inserts one subtree root into `frontierNodesAboveB` if it has not been seen yet.
             */
            void addFrontierNodeAboveB(index_t nodeId) {
                if (!visited_.isMarked(nodeId)) {
                    frontierNodesAboveB_.push_back(nodeId);
                    visited_.mark(nodeId);
                }
            }

            /**
             * @brief Inserts the path from one adjacent node to `nodeCa` into `mergeNodesByLevel`.
             * @details Nodes are distributed by their altitude and inserted at
             * most once across the whole collection build.
             */
            void addNodesOfPath(const tree_t &tree, const std::vector<altitude_t> &altitude, index_t adjacentNode, index_t nodeCa) {
                if (visited_.isMarked(adjacentNode)) {
                    return;
                }
                index_t nodeId = adjacentNode;
                while (nodeId != invalid_index) {
                    if (!visited_.isMarked(nodeId)) {
                        auto &bucket = getMergedNodes(altitude[nodeId]);
                        bucket.push_back(nodeId);
                        maxBucketSize_ = std::max(maxBucketSize_, bucket.size());
                        visited_.mark(nodeId);
                    } else {
                        break;
                    }
                    if (nodeId == nodeCa) {
                        break;
                    }
                    const auto parentId = tree.getNodeParent(nodeId);
                    nodeId = (parentId == nodeId) ? invalid_index : parentId;
                }
            }

            /**
             * @brief Computes the graph-adjacent nodes of the removed dual subtree.
             * @details For each proper part of the subtree removed in the dual
             * tree, the method scans its graph neighbors and keeps the
             * smallest-component node in the updated tree when that neighbor
             * lies on the merge side of the altitude sweep. These nodes may
             * then contribute to `mergeNodesByLevel` or `frontierNodesAboveB`.
             *
             * @param tree Tree currently being updated.
             * @param altitude Altitude buffer associated with `tree`.
             * @param properPartSetC Proper parts of the removed subtree in the
             *        dual tree. This vector corresponds to the set `C` in [2].
             * @param graph Adjacency graph of the image domain shared by both trees.
             */
            template<typename graph_type>
            void computeAdjacentNodes(const tree_t &tree, const std::vector<altitude_t> &altitude, const std::vector<index_t> &properPartSetC, const graph_type &graph) {
                for (auto p: properPartSetC) {
                    const auto nodeP = tree.getSmallestComponent(p);
                    if (nodeP == invalid_index) {
                        continue;
                    }
                    const auto altitudeP = altitude[nodeP];
                    for (auto q: adjacent_vertex_iterator(p, graph)) {
                        const auto nodeQ = tree.getSmallestComponent(q);
                        if (nodeQ == invalid_index) {
                            continue;
                        }
                        const auto altitudeQ = altitude[nodeQ];
                        if (((isMaxtree_ && altitudeQ > altitudeP) || (!isMaxtree_ && altitudeQ < altitudeP)) && !visitedAdj_.isMarked(nodeQ)) {
                            adjacentNodes_.push_back(nodeQ);
                            visitedAdj_.mark(nodeQ);
                        }
                    }
                }
            }

            /**
             * @brief Builds `mergeNodesByLevel` and `frontierNodesAboveB` for one subtree-adjustment step.
             * @details Starting from the set `C` of proper parts removed in the
             * dual tree, this method first computes the graph-adjacent nodes in
             * the tree being updated. Each adjacent node is then classified with
             * respect to the sweep ending at level `b`: if it already lies in
             * the interval toward `nodeCa`, its path to `nodeCa` is inserted
             * into `mergeNodesByLevel`; otherwise, the method climbs toward the root to
             * find the last node that still belongs to the sweep and inserts
             * either that path into `mergeNodesByLevel` or the resulting frontier node
             * into `frontierNodesAboveB`.
             */
            template<typename graph_type>
            void build(const tree_t &tree, const std::vector<altitude_t> &altitude, const std::vector<index_t> &properPartSetC, index_t nodeCa, altitude_t b, bool isMaxtree, const graph_type &graph) {
                resetCollection(isMaxtree);
                computeAdjacentNodes(tree, altitude, properPartSetC, graph);
                hg_assert(nodeCa != invalid_index, "nodeCa must map to a valid smallest component.");

                for (auto adjacentNode: adjacentNodes_) {
                    if ((isMaxtree && altitude[adjacentNode] <= b) || (!isMaxtree && altitude[adjacentNode] >= b)) {
                        addNodesOfPath(tree, altitude, adjacentNode, nodeCa);
                    } else {
                        index_t nodeSubtree = adjacentNode;
                        index_t n = adjacentNode;
                        while (n != invalid_index) { // Path from the adjacent node to the root, looking for the last node above b.
                            if ((isMaxtree && b > altitude[n]) || (!isMaxtree && b < altitude[n])) {
                                break;
                            }
                            nodeSubtree = n;
                            const auto parentId = tree.getNodeParent(n);
                            n = (parentId == n) ? invalid_index : parentId;
                        }

                        if (altitude[nodeSubtree] == b) {
                            addNodesOfPath(tree, altitude, nodeSubtree, nodeCa);
                        } else if (tree.getNodeParent(nodeSubtree) != invalid_index && tree.getNodeParent(nodeSubtree) != nodeCa) {
                            addNodesOfPath(tree, altitude, nodeSubtree, nodeCa);
                        } else {
                            addFrontierNodeAboveB(nodeSubtree);
                        }
                    }
                }
            }

            /**
             * @brief Builds the ordered list of active levels and returns the first one.
             */
            altitude_t firstMergeLevel() {
                mergeLevels_.clear();
                if constexpr (use_dense_levels) {
                    // Dense backend scans the full discrete altitude domain in numeric order.
                    for (std::size_t i = 0; i < mergeNodesByLevelStorage_.size(); ++i) {
                        if (!mergeNodesByLevelStorage_[i].empty()) {
                            mergeLevels_.push_back((altitude_t) i);
                        }
                    }
                } else {
                    // Sparse backend iterates only over altitude levels that were instantiated.
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
        tree_t *mintree_ = nullptr;
        tree_t *maxtree_ = nullptr;
        const graph_t *graph_ = nullptr;
        MergedNodesCollection mergeNodesByLevel_;
        GenerationStampSet removedMarks_;
        
        std::vector<index_t> properPartSetC_;
        std::vector<index_t> nodesPendingRemoval_;
        const DynamicComponentTreeAttributeComputer<double> *attributeComputer_ = nullptr;
        std::vector<double> *attributeBufferMin_ = nullptr;
        std::vector<double> *attributeBufferMax_ = nullptr;
        const std::vector<altitude_t> *altitudeBufferMin_ = nullptr;
        const std::vector<altitude_t> *altitudeBufferMax_ = nullptr;

        /**
         * @brief Detaches a node from its parent, optionally releasing it.
         * @details Delegates to `DynamicComponentTree::removeChild(parent, node, releaseNode)`.
         * @param tree Tree containing `nodeId`.
         * @param nodeId Node to detach.
         * @param releaseNode True to release the node slot after detaching it.
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
         * @brief Returns the attribute buffer associated with one of the two trees.
         * @param tree Either the min-tree or the max-tree owned by this helper.
         * @return `nullptr` when no incremental attribute computer is configured.
         */
        std::vector<double> *getAttributeBuffer(tree_t *tree) const {
            if (attributeComputer_ == nullptr) {
                return nullptr;
            }
            return tree == maxtree_ ? attributeBufferMax_ : attributeBufferMin_;
        }

        /**
         * @brief Returns the altitude buffer associated with one of the two trees.
         * @param tree Either the min-tree or the max-tree owned by this helper.
         * @return `nullptr` when the tree pointer is null.
         */
        const std::vector<altitude_t> *getAltitudeBuffer(const tree_t *tree) const {
            if (tree == nullptr) {
                return nullptr;
            }
            return tree == maxtree_ ? altitudeBufferMax_ : altitudeBufferMin_;
        }

        /**
         * @brief Reads the altitude of a node from the configured altitude buffer.
         * @return The altitude value associated with `nodeId`.
         */
        inline altitude_t nodeAltitude(const tree_t *tree, index_t nodeId) const {
            const auto *buffer = getAltitudeBuffer(tree);
            hg_assert(buffer != nullptr, "Altitude buffer must be configured before querying node levels.");
            return (*buffer)[nodeId];
        }

        /**
         * @brief Recomputes the incremental attribute of a single node after a local edit.
         * @details The recomputation uses the registered attribute computer and updates the tree-specific buffer in place.
         * @param tree Tree containing the edited node.
         * @param nodeId Edited node whose attribute must be refreshed.
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
         * @details Any node left without direct proper parts is marked as removable so it can be merged away later in the sweep.
         * @param tree Tree being updated.
         * @param unionNode Current union node receiving the removed proper parts.
         * @param properPartSetC Proper parts collected from the removed subtree. This vector corresponds to the set `C` in [2].
         */
        void moveSelectedProperPartsToNode(tree_t *tree, index_t unionNode, const std::vector<index_t> &properPartSetC) {
            for (auto pixelId: properPartSetC) {
                const index_t ownerId = tree->getSmallestComponent(pixelId);
                if (ownerId == invalid_index || ownerId == unionNode) {
                    continue;
                }

                tree->moveProperPart(unionNode, ownerId, pixelId);

                if (tree->isAlive(ownerId) && tree->getNumProperParts(ownerId) == 0 && ownerId != unionNode && !removedMarks_.isMarked(ownerId)) {
                    removedMarks_.mark(ownerId);
                }
            }
        }

        /**
         * @brief Returns the complementary hierarchy for the current update direction.
         * @param isMaxtree True if the target tree is the max-tree.
         */
        tree_t *getOtherTree(bool isMaxtree) {
            return isMaxtree ? mintree_ : maxtree_;
        }

        /**
         * @brief Reattaches all direct children of `sourceNodeId` under `parentId`.
         * @details Children order is preserved and proper parts are absorbed as well.
         */
        void absorbNodeIntoParent(tree_t *targetTree, index_t parentId, index_t sourceNodeId) {
            hg_assert(targetTree != nullptr, "targetTree must not be null.");
            targetTree->moveChildren(parentId, sourceNodeId);
            targetTree->moveProperParts(parentId, sourceNodeId);
        }

        /**
         * @brief Builds the internal merge buckets for one subtree-adjustment step.
         */
        void buildMergedCollections(const tree_t &tree, const std::vector<altitude_t> &altitude, const std::vector<index_t> &properPartSetC, index_t nodeCa, altitude_t b, bool isMaxtree) {
            mergeNodesByLevel_.build(tree, altitude, properPartSetC, nodeCa, b, isMaxtree, *graph_);
        }


        /**
         * @brief Updates one hierarchy after removing a rooted subtree in the dual hierarchy.
         * @details The method:
         * 1. gathers the proper parts of the removed subtree;
         * 2. finds the representative of `C_a^-` and the altitude interval to sweep;
         * 3. builds the internal merge collections;
         * 4. merges nodes level by level;
         * 5. reconnects the final union node in the updated hierarchy.
         *
         * Test-only access is provided through `testing::DualMinMaxTreeIncrementalFilterTestAccess`.
         *
         * @param tree Target hierarchy updated in place.
         * @param rootIdSubtree Root node of the subtree already removed in the complementary hierarchy.
         */
        void updateTree(tree_t *tree, index_t rootIdSubtree) {
            hg_assert(tree != nullptr, "updateTree: tree must not be null.");
            hg_assert(rootIdSubtree != invalid_index, "updateTree: rootIdSubtree is invalid.");

            const bool isMaxtree = tree == maxtree_;
            tree_t *otherTree = getOtherTree(isMaxtree);
            
            hg_assert(otherTree != nullptr, "updateTree: complementary tree is null.");
            hg_assert(rootIdSubtree >= 0 && rootIdSubtree < otherTree->getGlobalIdSpaceSize(), "updateTree: rootIdSubtree must be in complementary-tree bounds.");
            hg_assert(otherTree->isNode(rootIdSubtree) && otherTree->isAlive(rootIdSubtree), "updateTree: rootIdSubtree must be valid/alive in complementary tree.");

            const index_t parentSubtree = otherTree->getNodeParent(rootIdSubtree);
            hg_assert(parentSubtree != invalid_index && parentSubtree != rootIdSubtree, "updateTree: rootIdSubtree must be attached in complementary tree.");
            const altitude_t b = nodeAltitude(otherTree, parentSubtree);

            index_t nodeCa = invalid_index;
            altitude_t altitudeCa = altitude_t{};

            properPartSetC_.clear();
            properPartSetC_.reserve(64);
            for (auto nSubtree: otherTree->getNodeSubtree(rootIdSubtree)) {
                for (auto p: otherTree->getProperParts(nSubtree)) {
                    properPartSetC_.push_back(p);
                
                    const index_t nodeP = tree->getSmallestComponent(p);
                    if (nodeP == invalid_index) {
                        continue;
                    }
                    const altitude_t altitudeP = nodeAltitude(tree, nodeP);
                    if (nodeCa == invalid_index || ((isMaxtree && altitudeP < altitudeCa) || (!isMaxtree && altitudeP > altitudeCa))) {
                        altitudeCa = altitudeP;
                        nodeCa = nodeP;
                    }
                }
            }
            hg_assert(nodeCa != invalid_index, "updateTree: invalid C_a representative.");
            hg_assert(!properPartSetC_.empty(), "updateTree: expected non-empty proper parts in removed subtree.");

            
            const auto *targetAltitude = getAltitudeBuffer(tree);
            hg_assert(targetAltitude != nullptr, "updateTree: altitude buffer must be configured for target tree.");
            buildMergedCollections(*tree, *targetAltitude, properPartSetC_, nodeCa, b, isMaxtree);

            altitude_t mergeLevel = mergeNodesByLevel_.firstMergeLevel();
            index_t unionNode;
            index_t previousUnionNode = invalid_index;
            removedMarks_.resetAll();
            nodesPendingRemoval_.reserve(mergeNodesByLevel_.getMaxBucketSize());
            while (mergeNodesByLevel_.hasMergeLevel() && ((isMaxtree && mergeLevel > altitudeCa) || (!isMaxtree && mergeLevel < altitudeCa))) {
                auto &mergeNodesAtLevel = mergeNodesByLevel_.getMergedNodes(mergeLevel);
                unionNode = invalid_index;
                nodesPendingRemoval_.clear();

                for (auto nodeId: mergeNodesAtLevel) {
                    if (!tree->isAlive(nodeId)) {
                        continue;
                    }

                    if (unionNode == invalid_index) {
                        if (removedMarks_.isMarked(nodeId)) {
                            nodesPendingRemoval_.push_back(nodeId);
                            continue;
                        }
                        unionNode = nodeId;
                        this->disconnect(tree, unionNode, false);

                        for (auto pendingNodeId: nodesPendingRemoval_) {
                            this->absorbNodeIntoParent(tree, unionNode, pendingNodeId);
                            this->disconnect(tree, pendingNodeId, true);
                        }
                        nodesPendingRemoval_.clear();
                        continue;
                    }

                    this->absorbNodeIntoParent(tree, unionNode, nodeId);
                    this->disconnect(tree, nodeId, true);
                }

                if (unionNode == invalid_index) {
                    for (auto nodeId: nodesPendingRemoval_) {
                        this->absorbNodeIntoParent(tree, tree->getNodeParent(nodeId), nodeId);
                        this->disconnect(tree, nodeId, true);
                    }
                    mergeLevel = mergeNodesByLevel_.nextMergeLevel();
                    unionNode = previousUnionNode;
                    continue;
                }

                if (mergeLevel == b) {
                    this->moveSelectedProperPartsToNode(tree, unionNode, properPartSetC_);
                    for (auto nodeId: mergeNodesByLevel_.getFrontierNodesAboveB()) {
                        this->disconnect(tree, nodeId, false);
                        tree->attachNode(unionNode, nodeId);
                    }
                }

                if (previousUnionNode != invalid_index) {
                    if (tree->isAlive(previousUnionNode) && !tree->hasChild(unionNode, previousUnionNode)) {
                        if (!tree->isRoot(previousUnionNode)) {
                            this->disconnect(tree, previousUnionNode, false);
                        }
                        tree->attachNode(unionNode, previousUnionNode);
                    }
                }

                computeAttributeOnTreeNode(tree, unionNode);

                previousUnionNode = unionNode;
                mergeLevel = mergeNodesByLevel_.nextMergeLevel();
            }

            const index_t finalUnionNode = previousUnionNode;
            if (finalUnionNode != invalid_index && tree->isAlive(finalUnionNode)) {
                if (removedMarks_.isMarked(nodeCa)) {
                    if (!tree->isRoot(nodeCa)) {
                        const index_t parentIdNodeCa = tree->getNodeParent(nodeCa);

                        if (!tree->isRoot(finalUnionNode)) {
                            this->disconnect(tree, finalUnionNode, false);
                        }
                        tree->attachNode(parentIdNodeCa, finalUnionNode);

                        std::vector<index_t> childrenOfNodeCa;
                        childrenOfNodeCa.reserve((size_t) tree->getNumChildren(nodeCa));
                        for (auto n: tree->getChildren(nodeCa)) {
                            childrenOfNodeCa.push_back(n);
                        }

                        for (auto n: childrenOfNodeCa) {
                            if (n != finalUnionNode && !tree->hasChild(finalUnionNode, n)) {
                                if (!tree->isRoot(n)) {
                                    this->disconnect(tree, n, false);
                                }
                                tree->attachNode(finalUnionNode, n);
                            }
                        }

                        this->disconnect(tree, nodeCa, true);
                        computeAttributeOnTreeNode(tree, finalUnionNode);
                        computeAttributeOnTreeNode(tree, parentIdNodeCa);
                    } else {
                        index_t newRoot = finalUnionNode;

                        std::vector<index_t> childrenOfNodeCa;
                        childrenOfNodeCa.reserve((size_t) tree->getNumChildren(nodeCa));
                        for (auto n: tree->getChildren(nodeCa)) {
                            childrenOfNodeCa.push_back(n);
                        }

                        for (auto n: childrenOfNodeCa) {
                            if ((isMaxtree && nodeAltitude(tree, n) < nodeAltitude(tree, newRoot)) || (!isMaxtree && nodeAltitude(tree, n) > nodeAltitude(tree, newRoot))) {
                                newRoot = n;
                            }
                        }

                        if (newRoot != finalUnionNode) {
                            if (!tree->isRoot(finalUnionNode)) {
                                this->disconnect(tree, finalUnionNode, false);
                            }
                            tree->attachNode(newRoot, finalUnionNode);
                        }

                        for (auto n: childrenOfNodeCa) {
                            if (n != newRoot && !tree->hasChild(newRoot, n)) {
                                if (!tree->isRoot(n)) {
                                    this->disconnect(tree, n, false);
                                }
                                tree->attachNode(newRoot, n);
                            }
                        }

                        tree->setRoot(newRoot);

                        tree->releaseNode(nodeCa);
                        computeAttributeOnTreeNode(tree, newRoot);
                    }
                } else if (finalUnionNode != nodeCa) {
                    if (!tree->isRoot(finalUnionNode)) {
                        this->disconnect(tree, finalUnionNode, false);
                    }
                    tree->attachNode(nodeCa, finalUnionNode);
                    computeAttributeOnTreeNode(tree, nodeCa);
                }
            }

        }


    public:
        /**
         * @brief Returns `true` iff this instantiation uses the dense altitude-bucket backend.
         * @details Exposed mainly so tests and local benchmarks can report which backend is active.
         */
        static constexpr bool usesDenseLevelBackend() {
            return use_dense_levels;
        }

        /**
         * @brief Returns the compile-time bit-width threshold used to enable the dense backend.
         */
        static constexpr int denseLevelBackendMaxBits() {
            return HG_COMPONENT_TREE_ADJUSTMENT_DENSE_MAX_BITS;
        }

        /**
         * @brief Creates an adjustment helper from paired dynamic min/max trees.
         * @param mintree Dynamic min-tree.
         * @param maxtree Dynamic max-tree.
         * @param graph Underlying graph shared by both hierarchies.
         */
        DualMinMaxTreeIncrementalFilter(tree_t *mintree, tree_t *maxtree, const graph_t &graph): mintree_(mintree), maxtree_(maxtree), graph_(&graph), mergeNodesByLevel_(std::max(mintree ? mintree->getGlobalIdSpaceSize() : 0, maxtree ? maxtree->getGlobalIdSpaceSize() : 0)), removedMarks_(std::max(mintree ? mintree->getGlobalIdSpaceSize() : 0, maxtree ? maxtree->getGlobalIdSpaceSize() : 0)) {
            hg_assert(mintree_ != nullptr, "mintree must not be null.");
            hg_assert(maxtree_ != nullptr, "maxtree must not be null.");
            hg_assert(graph_ != nullptr, "graph must not be null.");
        }

        /**
         * @brief Defaulted destructor.
         */
        virtual ~DualMinMaxTreeIncrementalFilter() = default;

        /**
         * @brief Registers the incremental attribute computer and its two output buffers.
         * @details Buffer sizing is the caller's responsibility. Each buffer must be
         * indexable on the full global id space of its associated dynamic tree.
         * @param computer Incremental attribute computer used after local edits.
         * @param bufferMin Output buffer for min-tree attributes.
         * @param bufferMax Output buffer for max-tree attributes.
         */
        void setAttributeComputer(const DynamicComponentTreeAttributeComputer<double> &computer, std::vector<double> &bufferMin, std::vector<double> &bufferMax) {
            attributeComputer_ = &computer;
            attributeBufferMin_ = &bufferMin;
            attributeBufferMax_ = &bufferMax;
        }

        /**
         * @brief Registers external altitude arrays for the min-tree and max-tree.
         * @details The adjustment logic reads node levels exclusively from these buffers.
         * @param bufferMin Altitude buffer for the min-tree.
         * @param bufferMax Altitude buffer for the max-tree.
         */
        void setAltitudeBuffers(const std::vector<altitude_t> &bufferMin, const std::vector<altitude_t> &bufferMax) {
            altitudeBufferMin_ = &bufferMin;
            altitudeBufferMax_ = &bufferMax;
        }
    
        /**
         * @brief Prunes rooted subtrees from the max-tree and updates the min-tree after each valid prune.
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
         * @brief Prunes rooted subtrees from the min-tree and updates the max-tree after each valid prune.
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
            static void updateTree(DualMinMaxTreeIncrementalFilter<altitude_t, graph_t> &adjust, DynamicComponentTree *tree, index_t rootIdSubtree) {
                adjust.updateTree(tree, rootIdSubtree);
            }
        };
    }

} // namespace hg::detail::hierarchy
