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

#include <cstddef>
#include <iterator>
#include <vector>

#include "higra/structure/array.hpp"
#include "higra/structure/tree_graph.hpp"
#include "higra/utils.hpp"

namespace hg::detail::hierarchy {

    /**
     * @brief Dynamic representation of a component tree with separated internal-node and proper-part storage.
     *
     * The structure stores:
     *  - a mutable internal hierarchy restricted to internal nodes, and
     *  - a distinct proper-part ownership structure for pixels/leaves.
     *
     * Notation used in this class:
     *  - `N`: size of the global id space (`N = getGlobalIdSpaceSize()`).
     *  - `L`: number of proper parts / pixels (`L = getNumTotalProperParts()`).
     *  - `I`: number of node slots (`I = getNumInternalNodeSlots()`).
     *  - `C_component_tree`: cost of building a component tree with Higra
     *    (`component_tree_max_tree` or `component_tree_min_tree`), including sorting
     *    and union-find/canonization steps.
     *
     * Node id conventions:
     *  - Global id: id in `[0, N)`, addressing both proper parts and internal nodes.
     *  - Local node-slot id: id in `[0, I)`, addressing only internal nodes.
     *
     * Mapping between both spaces:
     *  - `globalNodeId = localNodeSlotId + L`
     *  - `localNodeSlotId = globalNodeId - L`
     *
     * Internal nodes occupy the interval `[L, N)`, following Higra topological convention.
     *
     * Local-slot policy:
     *  - mutable internal hierarchy arrays (`firstChild`, `lastChild`, `nextSibling`,
     *    `prevSibling`, `numChildrenByNode`, `nodeParent`) are indexed by
     *    and store local node-slot ids when applicable;
     *  - proper-part ownership arrays are indexed by global proper-part ids `[0, L)`;
     *  - public APIs still expose global ids to stay consistent with Higra conventions.
     *
     * Iterators/ranges provided by this class:
     *  - `getAliveNodeIds()`: linear range over all alive internal global ids `[L, N)`.
     *  - `getChildren(nodeId)`: direct internal children of `nodeId`.
     *  - `getProperParts(nodeId)`: proper-part pixels directly owned by `nodeId`.
     *  - `getPostOrderNodes()`, `getPostOrderNodes(rootNodeId)`:
     *    internal hierarchy in post-order.
     *  - `getBreadthFirstNodes()`, `getBreadthFirstNodes(rootNodeId)`:
     *    internal hierarchy in breadth-first order.
     *  - `getPathToRootNodes(nodeId)`: path from `nodeId` to root (both endpoints included).
     *  - `getNodeSubtree(nodeId)`: internal nodes of the rooted subtree (pre-order, includes `nodeId`).
     *  - `getDescendants(nodeId)`: descendants of `nodeId` (pre-order, excludes `nodeId`).
     *
     * Traversal ranges are lazy: creation is O(1); full iteration cost is linear in the
     * number of returned elements.
     */
    class DynamicComponentTree {
    private:
        // ---------------------------------------------------------------------
        // Private attributes
        // ---------------------------------------------------------------------

        // Global counters.
        index_t numTotalProperParts_ = 0;
        index_t numInternalNodeSlots_ = 0;
        index_t rootNodeId = invalid_index;

        // Free internal-node slots, stored as local node-slot ids [0, I).
        std::vector<index_t> freeNodeSlots;

        // Internal-node arrays, indexed by local node-slot id [0, I).
        // Values are global internal ids [L, N) or invalid_index when appropriate.
        std::vector<index_t> nodeParent;

        // Proper-part ownership, indexed by proper-part global id [0, L).
        // Values are global internal ids [L, N).
        std::vector<index_t> properPartOwner;

        // Proper-parts linked lists, indexed by local node-slot id [0, I).
        // Stored values are global proper-part ids [0, L).
        std::vector<index_t> properHead;
        std::vector<index_t> properTail;
        std::vector<index_t> numProperPartsByNode;
        std::vector<index_t> nextProperPart;
        std::vector<index_t> prevProperPart;

        // Internal hierarchy linked structure, indexed by local node-slot id [0, I).
        // Stored values are local internal ids [0, I) or invalid_index.
        std::vector<index_t> firstChild;
        std::vector<index_t> lastChild;
        std::vector<index_t> nextSibling;
        std::vector<index_t> prevSibling;
        std::vector<index_t> numChildrenByNode;

        // Fail-fast iterator versions.
        std::size_t nodeStructureVersion = 0;
        std::size_t topologyVersion = 0;
        std::size_t properPartVersion = 0;

        // ---------------------------------------------------------------------
        // Private methods
        // ---------------------------------------------------------------------

        /**
         * @brief Appends a detached child at the back of a parent's child list.
         * @warning Caller must ensure detached child and valid relation.
         * @complexity Time O(1), Space O(1).
         */
        void linkChildBack(index_t parentId, index_t childId) {
            const auto parentLocalId = localOf(parentId);
            const auto childLocalId = localOf(childId);
            const auto tail = lastChild[(size_t) parentLocalId];
            if (tail == invalid_index) {
                firstChild[(size_t) parentLocalId] = childLocalId;
                lastChild[(size_t) parentLocalId] = childLocalId;
            } else {
                nextSibling[(size_t) tail] = childLocalId;
                prevSibling[(size_t) childLocalId] = tail;
                lastChild[(size_t) parentLocalId] = childLocalId;
            }
            numChildrenByNode[(size_t) parentLocalId]++;
        }

        /**
         * @brief Unlinks an attached child from its parent child list.
         * @warning Caller must ensure attached state and non-root use.
         * @complexity Time O(1), Space O(1).
         */
        void unlinkChild(index_t childId) {
            const auto childLocalId = localOf(childId);
            const auto parentId = nodeParent[(size_t) childLocalId];
            const auto parentLocalId = localOf(parentId);
            const auto prev = prevSibling[(size_t) childLocalId];
            const auto next = nextSibling[(size_t) childLocalId];

            if (prev != invalid_index) {
                nextSibling[(size_t) prev] = next;
            } else {
                firstChild[(size_t) parentLocalId] = next;
            }

            if (next != invalid_index) {
                prevSibling[(size_t) next] = prev;
            } else {
                lastChild[(size_t) parentLocalId] = prev;
            }

            prevSibling[(size_t) childLocalId] = invalid_index;
            nextSibling[(size_t) childLocalId] = invalid_index;
            numChildrenByNode[(size_t) parentLocalId]--;
        }

        /**
         * @brief Moves all proper parts owned by nodeB to nodeA.
         * @warning This is a low-level helper that assumes both nodes are valid and alive.
         * @complexity Time O(k), Space O(1), where k is the number of moved proper parts.
         */
        void moveProperPartsInBackend(index_t nodeA, index_t nodeB) {
            const auto localA = localOf(nodeA);
            const auto localB = localOf(nodeB);
            if (numProperPartsByNode[(size_t) localB] == 0) {
                return;
            }

            for (auto pixelId = properHead[(size_t) localB]; pixelId != invalid_index; pixelId = nextProperPart[(size_t) pixelId]) {
                properPartOwner[(size_t) pixelId] = nodeA;
            }

            if (numProperPartsByNode[(size_t) localA] == 0) {
                properHead[(size_t) localA] = properHead[(size_t) localB];
                properTail[(size_t) localA] = properTail[(size_t) localB];
                numProperPartsByNode[(size_t) localA] = numProperPartsByNode[(size_t) localB];
            } else {
                prevProperPart[(size_t) properHead[(size_t) localB]] = properTail[(size_t) localA];
                nextProperPart[(size_t) properTail[(size_t) localA]] = properHead[(size_t) localB];
                properTail[(size_t) localA] = properTail[(size_t) localB];
                numProperPartsByNode[(size_t) localA] += numProperPartsByNode[(size_t) localB];
            }

            properHead[(size_t) localB] = invalid_index;
            properTail[(size_t) localB] = invalid_index;
            numProperPartsByNode[(size_t) localB] = 0;
        }

        /**
         * @brief Detaches an internal node from its parent in the backend state.
         * @complexity Time O(1), Space O(1).
         */
        void detachNodeInBackend(index_t nodeId) {
            unlinkChild(nodeId);
            const auto localId = localOf(nodeId);
            nodeParent[(size_t) localId] = nodeId;
        }

        /**
         * @brief Releases an internal node slot in the backend state.
         * @warning Node must already be detached and contain no children/proper parts.
         * @complexity Time O(1), Space O(1).
         */
        void releaseNodeSlot(index_t nodeId) {
            const auto localId = localOf(nodeId);
            firstChild[(size_t) localId] = invalid_index;
            lastChild[(size_t) localId] = invalid_index;
            nextSibling[(size_t) localId] = invalid_index;
            prevSibling[(size_t) localId] = invalid_index;
            numChildrenByNode[(size_t) localId] = 0;
            properHead[(size_t) localId] = invalid_index;
            properTail[(size_t) localId] = invalid_index;
            numProperPartsByNode[(size_t) localId] = 0;
            nodeParent[(size_t) localId] = invalid_index;
            freeNodeSlots.push_back(localId);
        }

        void checkNodeIteratorVersion(std::size_t expectedVersion) const {
            #ifndef NDEBUG
            hg_assert(expectedVersion == nodeStructureVersion,"Alive-node iterator invalidated by node-structure mutation.");
            #else
            (void) expectedVersion;
            #endif
        }

        void checkTopologyIteratorVersion(std::size_t expectedVersion) const {
            #ifndef NDEBUG
            hg_assert(expectedVersion == topologyVersion,"Topology iterator invalidated by tree-structure mutation.");
            #else
            (void) expectedVersion;
            #endif
        }

        void checkProperPartIteratorVersion(std::size_t expectedVersion) const {
            #ifndef NDEBUG
            hg_assert(expectedVersion == properPartVersion,"Proper-parts iterator invalidated by proper-part mutation.");
            #else
            (void) expectedVersion;
            #endif
        }

        /**
         * @brief Initializes storage for a dynamic tree with the given number of leaves and node slots.
         * @complexity Time O(L + I), Space O(L + I).
         */
        void initializeStorage(index_t numProperParts, index_t numInternalNodeSlots) {
            hg_assert(numProperParts > 0, "DynamicComponentTree must contain at least one proper part.");

            numTotalProperParts_ = numProperParts;
            numInternalNodeSlots_ = numInternalNodeSlots;
            rootNodeId = invalid_index;

            freeNodeSlots.clear();
            nodeParent.resize((size_t) numInternalNodeSlots_);
            properPartOwner.resize((size_t) numTotalProperParts_);

            properHead.assign((size_t) numInternalNodeSlots_, invalid_index);
            properTail.assign((size_t) numInternalNodeSlots_, invalid_index);
            numProperPartsByNode.assign((size_t) numInternalNodeSlots_, 0);
            nextProperPart.assign((size_t) numTotalProperParts_, invalid_index);
            prevProperPart.assign((size_t) numTotalProperParts_, invalid_index);
            firstChild.assign((size_t) numInternalNodeSlots_, invalid_index);
            lastChild.assign((size_t) numInternalNodeSlots_, invalid_index);
            nextSibling.assign((size_t) numInternalNodeSlots_, invalid_index);
            prevSibling.assign((size_t) numInternalNodeSlots_, invalid_index);
            numChildrenByNode.assign((size_t) numInternalNodeSlots_, 0);
            nodeStructureVersion = 0;
            topologyVersion = 0;
            properPartVersion = 0;
        }

        /**
         * @brief Builds the dynamic tree from Higra parent arrays.
         * @complexity Time O(N), Space O(N).
         */
        template<typename T1>
        void buildFromParent(const xt::xexpression<T1> &xparent, index_t numProperParts) {
            auto &sourceParent = xparent.derived_cast();

            hg_assert_1d_array(sourceParent);
            hg_assert_integral_value_type(sourceParent);

            const auto parent = xt::cast<index_t>(sourceParent);
            const index_t n = (index_t) parent.size();

            hg_assert(n > 0, "DynamicComponentTree cannot be empty.");
            hg_assert(numProperParts > 0 && numProperParts < n, "Invalid number of proper parts.");

            initializeStorage(numProperParts, n - numProperParts);
            rootNodeId = n - 1;

            for (index_t nodeId = numTotalProperParts_; nodeId < n; ++nodeId) {
                const auto localId = localOf(nodeId);
                nodeParent[(size_t) localId] = parent(nodeId);
                if (parent(nodeId) == invalid_index) {
                    freeNodeSlots.push_back(localId);
                }
            }

            for (index_t pixelId = 0; pixelId < numTotalProperParts_; ++pixelId) {
                const auto ownerId = parent(pixelId);
                properPartOwner[(size_t) pixelId] = ownerId;
                const auto ownerLocalId = localOf(ownerId);
                if (properHead[(size_t) ownerLocalId] == invalid_index) {
                    properHead[(size_t) ownerLocalId] = pixelId;
                } else {
                    prevProperPart[(size_t) pixelId] = properTail[(size_t) ownerLocalId];
                    nextProperPart[(size_t) properTail[(size_t) ownerLocalId]] = pixelId;
                }
                properTail[(size_t) ownerLocalId] = pixelId;
                numProperPartsByNode[(size_t) ownerLocalId]++;
            }

            for (index_t nodeId = numTotalProperParts_; nodeId < n; ++nodeId) {
                if (nodeId == rootNodeId || !isAlive(nodeId)) {
                    continue;
                }
                linkChildBack(parent(nodeId), nodeId);
            }
        }

        template<typename T>
        index_t inferNumProperParts(const xt::xexpression<T> &xparent) const {
            const tree inferredTree(xparent);
            return (index_t) inferredTree.num_leaves();
        }

        /**
         * @brief Maps a global internal id to a local internal id.
         * @complexity Time O(1), Space O(1).
         */
        index_t localOf(index_t nodeId) const {
            hg_assert(nodeId >= numTotalProperParts_ && nodeId < getGlobalIdSpaceSize(), "Node is not an internal node.");
            return nodeId - numTotalProperParts_;
        }

        /**
         * @brief Maps a local internal id to a global internal id.
         * @complexity Time O(1), Space O(1).
         */
        index_t globalOf(index_t localId) const { 
            return localId + numTotalProperParts_; 
        }

    public:

        // Iterators/ranges are declared here and implemented after the class body.
        class ChildrenIterator;             // Iterator over the direct node children of a node.
        class ChildrenRange;                // Range over the direct node children of a node.
        class AliveNodeIterator;            // Iterator over alive node global ids.
        class AliveNodeRange;               // Range over alive node global ids.
        class PostOrderNodeIterator;   // Iterator over a rooted node subtree in post-order.
        class PostOrderNodeRange;      // Range over a rooted node subtree in post-order.
        class PathToRootIterator;           // Iterator over the path from a node to the root.
        class PathToRootRange;              // Range over the path from a node to the root.
        class SubtreeNodeIterator;     // Iterator over a rooted node subtree in pre-order.
        class SubtreeNodeRange;        // Range over a rooted node subtree in pre-order.
        class DescendantNodeRange;    // Range over node descendants, excluding the node itself.
        class ProperPartsIterator;          // Iterator over the proper parts directly owned by a node.
        class ProperPartsRange;             // Range over the proper parts directly owned by a node.
        class BreadthFirstNodeIterator;// Iterator over a rooted node subtree in breadth-first order.
        class BreadthFirstNodeRange;   // Range over a rooted node subtree in breadth-first order.

        DynamicComponentTree() = default;

        /**
         * @brief Constructs a dynamic tree from parent arrays using Higra tree conventions.
         * @warning The parent array is expected to follow the Higra tree convention, with proper parts
         *          occupying the prefix `[0, L)`.
         * @complexity Time O(N), Space O(N).
         */
        template<typename T1>
        explicit DynamicComponentTree(const xt::xexpression<T1> &xparent) {
            buildFromParent(xparent, inferNumProperParts(xparent));
        }

        /**
         * @brief Constructs a dynamic tree from a Higra tree.
         * @complexity Time O(N), Space O(N).
         */
        explicit DynamicComponentTree(const tree &inputTree) {
            buildFromParent(inputTree.parents(), (index_t) inputTree.num_leaves());
        }

        /**
         * @brief Resets this dynamic tree from parent arrays using Higra tree conventions.
         * @warning The parent array is expected to follow the Higra tree convention, with proper parts
         *          occupying the prefix `[0, L)`.
         * @complexity Time O(N), Space O(N).
         */
        template<typename T1>
        void reset(const xt::xexpression<T1> &xparent) {
            buildFromParent(xparent, inferNumProperParts(xparent));
        }

        /**
         * @brief Resets this dynamic tree from a Higra tree.
         * @complexity Time O(N), Space O(N).
         */
        void reset(const tree &inputTree) {
            buildFromParent(inputTree.parents(), (index_t) inputTree.num_leaves());
        }

        /**
         * @brief Returns the number of internal-node slots, alive or free.
         * @complexity Time O(1), Space O(1).
         */
        index_t getNumInternalNodeSlots() const { 
            return numInternalNodeSlots_; 
        }

        /**
         * @brief Returns the size of the global id space `[0, L + I)`.
         * @complexity Time O(1), Space O(1).
         */
        index_t getGlobalIdSpaceSize() const { 
            return numTotalProperParts_ + numInternalNodeSlots_; 
        }

        /**
         * @brief Counts alive leaf nodes in the internal hierarchy.
         * @complexity Time O(I), Space O(1).
         */
        index_t getNumLeafNodes() const {
            index_t count = 0;
            for (index_t nodeId = numTotalProperParts_; nodeId < getGlobalIdSpaceSize(); ++nodeId) {
                if (!isAlive(nodeId)) {
                    continue;
                }
                if (isLeaf(nodeId)) {
                    count++;
                }
            }
            return count;
        }

        /**
         * @brief Returns the total number of proper parts (pixels) in the image domain.
         * @complexity Time O(1), Space O(1).
         */
        index_t getNumTotalProperParts() const { 
            return numTotalProperParts_; 
        }

        /**
         * @brief Returns the number of alive internal nodes.
         * @complexity Time O(1), Space O(1).
         */
        index_t getNumNodes() const { 
            return numInternalNodeSlots_ - (index_t) freeNodeSlots.size(); 
        }

        /**
         * @brief Tests whether a global id refers to an internal node slot.
         * @complexity Time O(1), Space O(1).
         */
        bool isNode(index_t nodeId) const { 
            return nodeId >= numTotalProperParts_ && nodeId < getGlobalIdSpaceSize(); 
        }

        /**
         * @brief Tests whether a global id refers to a proper part / pixel.
         * @complexity Time O(1), Space O(1).
         */
        bool isProperPart(index_t nodeId) const { 
            return nodeId >= 0 && nodeId < numTotalProperParts_; 
        }

        /**
         * @brief Returns the current root id.
         * @complexity Time O(1), Space O(1).
         */
        index_t getRoot() const { 
            return rootNodeId; 
        }

        /**
         * @brief Tests whether a node is the current root.
         * @complexity Time O(1), Space O(1).
         */
        bool isRoot(index_t nodeId) const { 
            return nodeId == rootNodeId; 
        }

        /**
         * @brief Returns the number of free internal-node slots currently available for reuse.
         * @complexity Time O(1), Space O(1).
         */
        index_t getNumFreeNodeSlots() const { 
            return (index_t) freeNodeSlots.size(); 
        }

        /**
         * @brief Returns the number of direct children of an internal node.
         * @complexity Time O(1), Space O(1).
         */
        index_t getNumChildren(index_t nodeId) const { 
            return numChildrenByNode[(size_t) localOf(nodeId)];
        }

        /**
         * @brief Tests whether an internal node has no internal children.
         * @complexity Time O(1), Space O(1).
         */
        bool isLeaf(index_t nodeId) const { 
            return firstChild[(size_t) localOf(nodeId)] == invalid_index; 
        }

        /**
         * @brief Returns the number of direct proper parts currently owned by a given node.
         * @complexity Time O(1), Space O(1).
         */
        index_t getNumProperParts(index_t nodeId) const { 
            return numProperPartsByNode[(size_t) localOf(nodeId)];
        }

        /**
         * @brief Tests whether childId is currently a direct child of parentId.
         * @complexity Time O(1), Space O(1).
         */
        bool hasChild(index_t parentId, index_t childId) const { 
            return getNodeParent(childId) == parentId;
        }

        /**
         * @brief Returns all alive internal nodes.
         * @complexity Range creation O(1); full iteration O(I).
         */
        AliveNodeRange getAliveNodeIds() const;

        /**
         * @brief Returns direct internal children of a node.
         * @complexity Range creation O(1); full iteration O(#children).
         */
        ChildrenRange getChildren(index_t nodeId) const;

        /**
         * @brief Returns direct proper parts owned by a node.
         * @complexity Range creation O(1); full iteration O(#proper parts).
         */
        ProperPartsRange getProperParts(index_t nodeId) const;

        /**
         * @brief Returns a post-order traversal from the current root.
         * @complexity Range creation O(1); full iteration O(#returned nodes).
         */
        PostOrderNodeRange getPostOrderNodes() const;

        /**
         * @brief Returns a post-order traversal rooted at rootNodeId.
         * @complexity Range creation O(1); full iteration O(#returned nodes).
         */
        PostOrderNodeRange getPostOrderNodes(index_t rootNodeId) const;

        /**
         * @brief Returns a breadth-first traversal from the current root.
         * @complexity Range creation O(1); full iteration O(#returned nodes).
         */
        BreadthFirstNodeRange getBreadthFirstNodes() const;

        /**
         * @brief Returns a breadth-first traversal rooted at rootNodeId.
         * @complexity Range creation O(1); full iteration O(#returned nodes).
         */
        BreadthFirstNodeRange getBreadthFirstNodes(index_t rootNodeId) const;

        /**
         * @brief Returns the path from nodeId to the root, both endpoints included.
         * @complexity Range creation O(1); full iteration O(path length).
         */
        PathToRootRange getPathToRootNodes(index_t nodeId) const;

        /**
         * @brief Returns the rooted subtree of nodeId in pre-order.
         * @complexity Range creation O(1); full iteration O(#returned nodes).
         */
        SubtreeNodeRange getNodeSubtree(index_t nodeId) const;

        /**
         * @brief Returns descendants of nodeId in pre-order, excluding nodeId.
         * @complexity Range creation O(1); full iteration O(#returned nodes).
         */
        DescendantNodeRange getDescendants(index_t nodeId) const;

        /**
         * @brief Returns the parent of an internal node.
         * @warning nodeId must be an internal node.
         * @complexity Time O(1), Space O(1).
         */
        index_t getNodeParent(index_t nodeId) const {
            return nodeParent[(size_t) localOf(nodeId)];
        }

        /**
         * @brief Returns the current owner / smallest component of a proper part.
         * @complexity Time O(1), Space O(1).
         */
        index_t getSmallestComponent(index_t pixelId) const {
            if (pixelId >= 0 && pixelId < this->getNumTotalProperParts()) {
                return properPartOwner[(size_t) pixelId];
            }
            return invalid_index;
        }

        /**
         * @brief Tests whether an internal node slot is alive.
         * @complexity Time O(1), Space O(1).
         */
        bool isAlive(index_t nodeId) const {
            if (!isNode(nodeId)) {
                return false;
            }
            return getNodeParent(nodeId) != invalid_index;
        }

        /**
         * @brief Moves one direct proper part from sourceNodeId to targetNodeId.
         * @complexity Time O(1), Space O(1).
         */
        void moveProperPart(index_t targetNodeId, index_t sourceNodeId, index_t pixelId) {
            const auto targetLocalId = localOf(targetNodeId);
            const auto sourceLocalId = localOf(sourceNodeId);
            const auto prevPixel = prevProperPart[(size_t) pixelId];
            const auto nextPixel = nextProperPart[(size_t) pixelId];

            if (prevPixel == invalid_index) {
                properHead[(size_t) sourceLocalId] = nextPixel;
            } else {
                nextProperPart[(size_t) prevPixel] = nextPixel;
            }

            if (nextPixel == invalid_index) {
                properTail[(size_t) sourceLocalId] = prevPixel;
            } else {
                prevProperPart[(size_t) nextPixel] = prevPixel;
            }

            numProperPartsByNode[(size_t) sourceLocalId]--;

            prevProperPart[(size_t) pixelId] = properTail[(size_t) targetLocalId];
            nextProperPart[(size_t) pixelId] = invalid_index;

            if (numProperPartsByNode[(size_t) targetLocalId] == 0) {
                properHead[(size_t) targetLocalId] = pixelId;
                properTail[(size_t) targetLocalId] = pixelId;
                numProperPartsByNode[(size_t) targetLocalId] = 1;
            } else {
                nextProperPart[(size_t) properTail[(size_t) targetLocalId]] = pixelId;
                properTail[(size_t) targetLocalId] = pixelId;
                numProperPartsByNode[(size_t) targetLocalId] += 1;
            }
            properPartOwner[(size_t) pixelId] = targetNodeId;
            properPartVersion++;
        }

        /**
         * @brief Moves all direct proper parts of nodeB to nodeA.
         * @complexity Time O(k), Space O(1), where k is the number of moved proper parts.
         */
        void moveProperParts(index_t nodeA, index_t nodeB) {
            const auto localA = localOf(nodeA);
            const auto localB = localOf(nodeB);
            auto movedHead = properHead[(size_t) localB];
            if (movedHead == invalid_index) {
                return;
            }

            for (auto pixelId = movedHead; pixelId != invalid_index; pixelId = nextProperPart[(size_t) pixelId]) {
                properPartOwner[(size_t) pixelId] = nodeA;
            }

            if (numProperPartsByNode[(size_t) localA] == 0) {
                properHead[(size_t) localA] = movedHead;
                properTail[(size_t) localA] = properTail[(size_t) localB];
                numProperPartsByNode[(size_t) localA] = numProperPartsByNode[(size_t) localB];
            } else {
                prevProperPart[(size_t) movedHead] = properTail[(size_t) localA];
                nextProperPart[(size_t) properTail[(size_t) localA]] = movedHead;
                properTail[(size_t) localA] = properTail[(size_t) localB];
                numProperPartsByNode[(size_t) localA] += numProperPartsByNode[(size_t) localB];
            }

            properHead[(size_t) localB] = invalid_index;
            properTail[(size_t) localB] = invalid_index;
            numProperPartsByNode[(size_t) localB] = 0;
            properPartVersion++;
        }

        /**
         * @brief Sets nodeId as the tree root.
         * @warning If nodeId is attached, it is detached first.
         * @complexity Time O(1), Space O(1).
         */
        void setRoot(index_t nodeId) {
            const auto oldRoot = getRoot();
            const auto oldParentId = getNodeParent(nodeId);
            if (oldParentId != invalid_index && oldParentId != nodeId) {
                unlinkChild(nodeId);
            }
            if (oldRoot != invalid_index && oldRoot != nodeId && isNode(oldRoot) && isAlive(oldRoot)) {
                const auto oldRootLocalId = localOf(oldRoot);
                nodeParent[(size_t) oldRootLocalId] = getNodeParent(oldRoot);
            }
            rootNodeId = nodeId;
            const auto localId = localOf(nodeId);
            nodeParent[(size_t) localId] = nodeId;
            topologyVersion++;
        }

        /**
         * @brief Allocates a released internal-node slot (LIFO free list).
         * @return Global internal id, or invalid_index if there is no free slot.
         * @complexity Time O(1), Space O(1).
         */
        index_t allocateNode() {
            if (freeNodeSlots.empty()) {
                return invalid_index;
            }
            const auto localId = freeNodeSlots.back();
            freeNodeSlots.pop_back();
            const auto nodeId = globalOf(localId);
            if (nodeId != invalid_index) {
                nodeParent[(size_t) localId] = nodeId;
                firstChild[(size_t) localId] = invalid_index;
                lastChild[(size_t) localId] = invalid_index;
                nextSibling[(size_t) localId] = invalid_index;
                prevSibling[(size_t) localId] = invalid_index;
                numChildrenByNode[(size_t) localId] = 0;
                properHead[(size_t) localId] = invalid_index;
                properTail[(size_t) localId] = invalid_index;
                numProperPartsByNode[(size_t) localId] = 0;
            }
            nodeStructureVersion++;
            return nodeId;
        }

        /**
         * @brief Releases an internal-node slot back to the free list.
         * @warning Node must be detached and empty.
         * @complexity Time O(1), Space O(1).
         */
        void releaseNode(index_t nodeId) {
            hg_assert(isAlive(nodeId), "releaseNode expects an alive internal node.");
            hg_assert(getNodeParent(nodeId) == nodeId, "releaseNode expects a detached node.");
            hg_assert(firstChild[(size_t) localOf(nodeId)] == invalid_index, "releaseNode expects node with no children.");
            hg_assert(numProperPartsByNode[(size_t) localOf(nodeId)] == 0, "releaseNode expects node with no proper parts.");
            const auto localId = localOf(nodeId);
            firstChild[(size_t) localId] = invalid_index;
            lastChild[(size_t) localId] = invalid_index;
            nextSibling[(size_t) localId] = invalid_index;
            prevSibling[(size_t) localId] = invalid_index;
            numChildrenByNode[(size_t) localId] = 0;
            properHead[(size_t) localId] = invalid_index;
            properTail[(size_t) localId] = invalid_index;
            numProperPartsByNode[(size_t) localId] = 0;
            nodeParent[(size_t) localId] = invalid_index;
            freeNodeSlots.push_back(localId);
            nodeStructureVersion++;
        }

        /**
         * @brief Removes a direct child from a parent, optionally releasing the child slot.
         * @complexity Time O(1), Space O(1).
         */
        void removeChild(index_t parentId, index_t childId, bool releaseNodeFlag) {
            const auto childWasAlive = isNode(childId) && isAlive(childId);
            const auto wasDirectChild = childWasAlive && getNodeParent(childId) == parentId;
            if (!wasDirectChild) {
                return;
            }
            unlinkChild(childId);
            const auto childLocalId = localOf(childId);
            if (releaseNodeFlag) {
                nodeParent[(size_t) childLocalId] = invalid_index;
                freeNodeSlots.push_back(childLocalId);
                nodeStructureVersion++;
            } else {
                nodeParent[(size_t) childLocalId] = childId;
            }
            topologyVersion++;
        }

        /**
         * @brief Attaches a detached node as the last child of parentId.
         * @complexity Time O(1), Space O(1).
         */
        void attachNode(index_t parentId, index_t nodeId) {
            linkChildBack(parentId, nodeId);
            const auto localId = localOf(nodeId);
            nodeParent[(size_t) localId] = parentId;
            topologyVersion++;
        }

        /**
         * @brief Detaches an attached node from its parent.
         * @complexity Time O(1), Space O(1).
         */
        void detachNode(index_t nodeId) {
            unlinkChild(nodeId);
            const auto localId = localOf(nodeId);
            nodeParent[(size_t) localId] = nodeId;
            topologyVersion++;
        }

        /**
         * @brief Moves an attached node under a new parent.
         * @complexity Time O(1), Space O(1).
         */
        void moveNode(index_t nodeId, index_t newParentId) {
            const auto oldParentId = getNodeParent(nodeId);
            if (oldParentId != newParentId) {
                unlinkChild(nodeId);
                linkChildBack(newParentId, nodeId);
            }
            const auto localId = localOf(nodeId);
            nodeParent[(size_t) localId] = newParentId;
            topologyVersion++;
        }

        /**
         * @brief Moves all direct children of sourceId under parentId.
         * @complexity Time O(k), Space O(1), where k is the number of moved children.
         */
        void moveChildren(index_t parentId, index_t sourceId) {
            const auto parentLocalId = localOf(parentId);
            const auto sourceLocalId = localOf(sourceId);
            auto firstLocalId = firstChild[(size_t) sourceLocalId];
            if (firstLocalId == invalid_index) {
                return;
            }
            auto lastLocalId = lastChild[(size_t) sourceLocalId];
            auto movedCount = numChildrenByNode[(size_t) sourceLocalId];
            auto tail = lastChild[(size_t) parentLocalId];

            firstChild[(size_t) sourceLocalId] = invalid_index;
            lastChild[(size_t) sourceLocalId] = invalid_index;
            numChildrenByNode[(size_t) sourceLocalId] = 0;
            prevSibling[(size_t) firstLocalId] = invalid_index;
            nextSibling[(size_t) lastLocalId] = invalid_index;

            if (tail == invalid_index) {
                firstChild[(size_t) parentLocalId] = firstLocalId;
                lastChild[(size_t) parentLocalId] = lastLocalId;
            } else {
                nextSibling[(size_t) tail] = firstLocalId;
                prevSibling[(size_t) firstLocalId] = tail;
                lastChild[(size_t) parentLocalId] = lastLocalId;
            }
            numChildrenByNode[(size_t) parentLocalId] += movedCount;

            for (auto childLocalId = firstLocalId; childLocalId != invalid_index; childLocalId = nextSibling[(size_t) childLocalId]) {
                nodeParent[(size_t) childLocalId] = parentId;
            }
            topologyVersion++;
        }

        /**
         * @brief Removes a subtree and absorbs its direct proper parts into the parent.
         * @warning nodeId must be a non-root alive internal node.
         * @complexity Time O(s + p), Space O(1), where s is the number of nodes in the
         *             subtree and p the number of proper parts owned by those nodes.
         */
        void pruneNode(index_t nodeId) {
            const auto parentId = this->getNodeParent(nodeId);
            auto descendToDeepestLastChild = [this](index_t startId) {
                auto currentId = startId;
                while (!isLeaf(currentId)) {
                    currentId = globalOf(lastChild[(size_t) localOf(currentId)]);
                }
                return currentId;
            };
            auto currentId = descendToDeepestLastChild(nodeId);

            while (true) {
                if (currentId == nodeId) {
                    detachNodeInBackend(nodeId);
                    moveProperPartsInBackend(parentId, nodeId);
                    releaseNodeSlot(nodeId);
                    topologyVersion++;
                    nodeStructureVersion++;
                    properPartVersion++;
                    break;
                }

                const auto currentParentId = getNodeParent(currentId);
                detachNodeInBackend(currentId);
                moveProperPartsInBackend(parentId, currentId);
                releaseNodeSlot(currentId);
                topologyVersion++;
                nodeStructureVersion++;
                properPartVersion++;

                if (isLeaf(currentParentId)) {
                    currentId = currentParentId;
                } else {
                    currentId = descendToDeepestLastChild(currentParentId);
                }
            }
        }

        /**
         * @brief Merges a node into its parent, reattaching direct children and proper parts.
         * @warning nodeId must be a non-root alive internal node.
         * @complexity Time O(c + p), Space O(1), where c is the number of direct children and
         *             p the number of direct proper parts of nodeId.
         */
        void mergeNodeIntoParent(index_t nodeId) {
            const auto parentId = getNodeParent(nodeId);
            detachNodeInBackend(nodeId);
            moveChildren(parentId, nodeId);
            moveProperPartsInBackend(parentId, nodeId);
            releaseNodeSlot(nodeId);
            topologyVersion++;
            nodeStructureVersion++;
            properPartVersion++;
        }

    };





    // -------------------------------------------------------------------------
    // Iterators and ranges implementation
    // -------------------------------------------------------------------------


    /**
     * @brief Iterator over the direct internal children of a node.
     */
    class DynamicComponentTree::ChildrenIterator {
    private:
        const DynamicComponentTree *tree_ = nullptr;
        index_t currentLocal_ = invalid_index;
        std::size_t expectedVersion_ = 0;

    public:
        using value_type = index_t;
        using difference_type = std::ptrdiff_t;
        using pointer = const index_t *;
        using reference = const index_t &;
        using iterator_category = std::forward_iterator_tag;

        ChildrenIterator() = default;
        ChildrenIterator(const DynamicComponentTree *tree, index_t currentLocal, std::size_t expectedVersion)
                : tree_(tree), currentLocal_(currentLocal), expectedVersion_(expectedVersion) {
        }

        index_t operator*() const {
            tree_->checkTopologyIteratorVersion(expectedVersion_);
            return tree_->globalOf(currentLocal_);
        }

        ChildrenIterator &operator++() {
            tree_->checkTopologyIteratorVersion(expectedVersion_);
            if (currentLocal_ != invalid_index) {
                currentLocal_ = tree_->nextSibling[(size_t) currentLocal_];
            }
            return *this;
        }

        bool operator==(const ChildrenIterator &other) const {
            return tree_ == other.tree_ && currentLocal_ == other.currentLocal_;
        }

        bool operator!=(const ChildrenIterator &other) const { return !(*this == other); }
    };

    /**
     * @brief Range over the direct internal children of a node.
     * @usage `for (auto n : tree.getChildren(nodeId)) { ... }`
     * @complexity Time O(1) to create, O(k) to iterate, Space O(1), where k is the number of children.
     * @warning Lazy range. Invalidated if the tree topology changes during iteration.
     */
    class DynamicComponentTree::ChildrenRange {
    private:
        const DynamicComponentTree *tree_ = nullptr;
        index_t firstLocal_ = invalid_index;
        std::size_t expectedVersion_ = 0;

    public:
        ChildrenRange() = default;
        ChildrenRange(const DynamicComponentTree *tree, index_t firstLocal)
                : tree_(tree), firstLocal_(firstLocal), expectedVersion_(tree ? tree->topologyVersion : 0) {
        }

        ChildrenIterator begin() const { return ChildrenIterator(tree_, firstLocal_, expectedVersion_); }
        ChildrenIterator end() const { return ChildrenIterator(tree_, invalid_index, expectedVersion_); }
    };

    /**
     * @brief Iterator over alive internal-node global ids.
     */
    class DynamicComponentTree::AliveNodeIterator {
    private:
        const DynamicComponentTree *tree_ = nullptr;
        index_t current_ = invalid_index;
        index_t end_ = invalid_index;
        std::size_t expectedVersion_ = 0;

        void settle() {
            while (tree_ && current_ < end_) {
                tree_->checkNodeIteratorVersion(expectedVersion_);
                if (tree_->getNodeParent(current_) != invalid_index) {
                    return;
                }
                current_++;
            }
            current_ = end_;
        }

    public:
        using value_type = index_t;
        using difference_type = std::ptrdiff_t;
        using pointer = const index_t *;
        using reference = const index_t &;
        using iterator_category = std::forward_iterator_tag;

        AliveNodeIterator() = default;
        AliveNodeIterator(const DynamicComponentTree *tree, index_t current, index_t end, std::size_t expectedVersion)
                : tree_(tree), current_(current), end_(end), expectedVersion_(expectedVersion) {
            settle();
        }

        index_t operator*() const {
            tree_->checkNodeIteratorVersion(expectedVersion_);
            return current_;
        }

        AliveNodeIterator &operator++() {
            tree_->checkNodeIteratorVersion(expectedVersion_);
            current_++;
            settle();
            return *this;
        }

        bool operator==(const AliveNodeIterator &other) const {
            return tree_ == other.tree_ && current_ == other.current_ && end_ == other.end_;
        }

        bool operator!=(const AliveNodeIterator &other) const { return !(*this == other); }
    };

    /**
     * @brief Range over alive internal-node global ids.
     * @usage `for (auto n : tree.getAliveNodeIds()) { ... }`
     * @complexity Time O(1) to create, O(I) to iterate, Space O(1), where I is the number of node slots.
     * @warning Lazy range. Invalidated if the node set changes during iteration.
     */
    class DynamicComponentTree::AliveNodeRange {
    private:
        const DynamicComponentTree *tree_ = nullptr;
        index_t begin_ = 0;
        index_t end_ = 0;
        std::size_t expectedVersion_ = 0;

    public:
        AliveNodeRange() = default;
        AliveNodeRange(const DynamicComponentTree *tree, index_t begin, index_t end)
                : tree_(tree), begin_(begin), end_(end), expectedVersion_(tree ? tree->nodeStructureVersion : 0) {
        }

        AliveNodeIterator begin() const { return AliveNodeIterator(tree_, begin_, end_, expectedVersion_); }
        AliveNodeIterator end() const { return AliveNodeIterator(tree_, end_, end_, expectedVersion_); }
    };

    /**
     * @brief Iterator over an internal rooted subtree in post-order.
     */
    class DynamicComponentTree::PostOrderNodeIterator {
    private:
        struct Item {
            index_t nodeId = invalid_index;
            bool expanded = false;
        };

        const DynamicComponentTree *tree_ = nullptr;
        std::vector<Item> stack_;
        index_t current_ = invalid_index;
        std::size_t expectedVersion_ = 0;

        void settle() {
            tree_->checkTopologyIteratorVersion(expectedVersion_);
            current_ = invalid_index;
            while (!stack_.empty()) {
                auto &top = stack_.back();
                if (!top.expanded) {
                    top.expanded = true;
                    auto nodeLocal = tree_->localOf(top.nodeId);
                    auto childLocal = tree_->lastChild[(size_t) nodeLocal];
                    while (childLocal != invalid_index) {
                        stack_.push_back(Item{tree_->globalOf(childLocal), false});
                        childLocal = tree_->prevSibling[(size_t) childLocal];
                    }
                } else {
                    current_ = top.nodeId;
                    return;
                }
            }
        }

    public:
        using value_type = index_t;
        using difference_type = std::ptrdiff_t;
        using pointer = const index_t *;
        using reference = const index_t &;
        using iterator_category = std::input_iterator_tag;

        PostOrderNodeIterator() = default;
        PostOrderNodeIterator(const DynamicComponentTree *tree, index_t rootNodeId, std::size_t expectedVersion)
                : tree_(tree), expectedVersion_(expectedVersion) {
            if (!tree_ || rootNodeId < tree_->getNumTotalProperParts() || rootNodeId >= tree_->getGlobalIdSpaceSize()) {
                current_ = invalid_index;
                return;
            }
            stack_.push_back(Item{rootNodeId, false});
            settle();
        }

        index_t operator*() const {
            tree_->checkTopologyIteratorVersion(expectedVersion_);
            return current_;
        }

        PostOrderNodeIterator &operator++() {
            tree_->checkTopologyIteratorVersion(expectedVersion_);
            if (!stack_.empty()) {
                stack_.pop_back();
            }
            settle();
            return *this;
        }

        bool operator==(const PostOrderNodeIterator &other) const { return current_ == other.current_; }
        bool operator!=(const PostOrderNodeIterator &other) const { return !(*this == other); }
    };

    /**
     * @brief Range over an internal rooted subtree in post-order.
     * @usage `for (auto n : tree.getPostOrderNodes()) { ... }`
     * @complexity Time O(1) to create, O(S) to iterate, Space O(h), where S is subtree size and h its height.
     * @warning Lazy range. Invalidated if the tree topology changes during iteration.
     */
    class DynamicComponentTree::PostOrderNodeRange {
    private:
        const DynamicComponentTree *tree_ = nullptr;
        index_t rootNodeId_ = invalid_index;
        std::size_t expectedVersion_ = 0;

    public:
        PostOrderNodeRange() = default;
        PostOrderNodeRange(const DynamicComponentTree *tree, index_t rootNodeId)
                : tree_(tree), rootNodeId_(rootNodeId), expectedVersion_(tree ? tree->topologyVersion : 0) {
        }

        PostOrderNodeIterator begin() const { return PostOrderNodeIterator(tree_, rootNodeId_, expectedVersion_); }
        PostOrderNodeIterator end() const { return PostOrderNodeIterator(); }
    };

    /**
     * @brief Iterator over the path from a node to the root.
     */
    class DynamicComponentTree::PathToRootIterator {
    private:
        const DynamicComponentTree *tree_ = nullptr;
        index_t currentId_ = invalid_index;
        std::size_t expectedVersion_ = 0;

    public:
        using value_type = index_t;
        using difference_type = std::ptrdiff_t;
        using pointer = const index_t *;
        using reference = const index_t &;
        using iterator_category = std::input_iterator_tag;

        PathToRootIterator() = default;
        PathToRootIterator(const DynamicComponentTree *tree, index_t startId, std::size_t expectedVersion)
                : tree_(tree), currentId_(startId), expectedVersion_(expectedVersion) {
        }

        index_t operator*() const {
            tree_->checkTopologyIteratorVersion(expectedVersion_);
            return currentId_;
        }

        PathToRootIterator &operator++() {
            tree_->checkTopologyIteratorVersion(expectedVersion_);
            if (!tree_ || currentId_ == invalid_index) {
                currentId_ = invalid_index;
                return *this;
            }
            auto parentId = tree_->getNodeParent(currentId_);
            currentId_ = (parentId == currentId_) ? invalid_index : parentId;
            return *this;
        }

        bool operator==(const PathToRootIterator &other) const { return currentId_ == other.currentId_; }
        bool operator!=(const PathToRootIterator &other) const { return !(*this == other); }
    };

    /**
     * @brief Range over the path from a node to the root.
     * @usage `for (auto n : tree.getPathToRootNodes(nodeId)) { ... }`
     * @complexity Time O(1) to create, O(h) to iterate, Space O(1), where h is path length.
     * @warning Lazy range. Invalidated if the tree topology changes during iteration.
     */
    class DynamicComponentTree::PathToRootRange {
    private:
        const DynamicComponentTree *tree_ = nullptr;
        index_t startId_ = invalid_index;
        std::size_t expectedVersion_ = 0;

    public:
        PathToRootRange() = default;
        PathToRootRange(const DynamicComponentTree *tree, index_t nodeId)
                : tree_(tree), startId_(nodeId), expectedVersion_(tree ? tree->topologyVersion : 0) {
        }

        PathToRootIterator begin() const {
            if (!tree_ || startId_ < 0 || startId_ >= tree_->getGlobalIdSpaceSize()) {
                return PathToRootIterator();
            }
            return PathToRootIterator(tree_, startId_, expectedVersion_);
        }

        PathToRootIterator end() const { return PathToRootIterator(); }
    };

    /**
     * @brief Iterator over an internal rooted subtree in pre-order.
     */
    class DynamicComponentTree::SubtreeNodeIterator {
    private:
        const DynamicComponentTree *tree_ = nullptr;
        std::vector<index_t> stack_;
        index_t current_ = invalid_index;
        std::size_t expectedVersion_ = 0;

        void settle() { current_ = stack_.empty() ? invalid_index : stack_.back(); }

    public:
        using value_type = index_t;
        using difference_type = std::ptrdiff_t;
        using pointer = const index_t *;
        using reference = const index_t &;
        using iterator_category = std::input_iterator_tag;

        SubtreeNodeIterator() = default;
        SubtreeNodeIterator(const DynamicComponentTree *tree, index_t rootNodeId, std::size_t expectedVersion)
                : tree_(tree), expectedVersion_(expectedVersion) {
            if (!tree_ || rootNodeId < tree_->getNumTotalProperParts() || rootNodeId >= tree_->getGlobalIdSpaceSize()) {
                current_ = invalid_index;
                return;
            }
            stack_.push_back(rootNodeId);
            settle();
        }

        index_t operator*() const {
            tree_->checkTopologyIteratorVersion(expectedVersion_);
            return current_;
        }

        SubtreeNodeIterator &operator++() {
            tree_->checkTopologyIteratorVersion(expectedVersion_);
            if (stack_.empty()) {
                current_ = invalid_index;
                return *this;
            }
            auto nodeId = stack_.back();
            stack_.pop_back();
            auto nodeLocal = tree_->localOf(nodeId);
            auto childLocal = tree_->lastChild[(size_t) nodeLocal];
            while (childLocal != invalid_index) {
                stack_.push_back(tree_->globalOf(childLocal));
                childLocal = tree_->prevSibling[(size_t) childLocal];
            }
            settle();
            return *this;
        }

        bool operator==(const SubtreeNodeIterator &other) const { return current_ == other.current_; }
        bool operator!=(const SubtreeNodeIterator &other) const { return !(*this == other); }
    };

    /**
     * @brief Range over an internal rooted subtree in pre-order.
     * @usage `for (auto n : tree.getNodeSubtree(nodeId)) { ... }`
     * @complexity Time O(1) to create, O(S) to iterate, Space O(h), where S is subtree size and h its height.
     * @warning Lazy range. Invalidated if the tree topology changes during iteration.
     */
    class DynamicComponentTree::SubtreeNodeRange {
    private:
        const DynamicComponentTree *tree_ = nullptr;
        index_t rootNodeId_ = invalid_index;
        std::size_t expectedVersion_ = 0;

    public:
        SubtreeNodeRange() = default;
        SubtreeNodeRange(const DynamicComponentTree *tree, index_t rootNodeId)
                : tree_(tree), rootNodeId_(rootNodeId), expectedVersion_(tree ? tree->topologyVersion : 0) {
        }

        SubtreeNodeIterator begin() const { return SubtreeNodeIterator(tree_, rootNodeId_, expectedVersion_); }
        SubtreeNodeIterator end() const { return SubtreeNodeIterator(); }
    };

    /**
     * @brief Range over the descendants of a node, excluding the node itself.
     * @usage `for (auto n : tree.getDescendants(nodeId)) { ... }`
     * @complexity Time O(1) to create, O(S) to iterate, Space O(h), where S is subtree size and h its height.
     * @warning Lazy range. Invalidated if the tree topology changes during iteration.
     */
    class DynamicComponentTree::DescendantNodeRange {
    private:
        const DynamicComponentTree *tree_ = nullptr;
        index_t rootNodeId_ = invalid_index;

    public:
        DescendantNodeRange() = default;
        DescendantNodeRange(const DynamicComponentTree *tree, index_t rootNodeId)
                : tree_(tree), rootNodeId_(rootNodeId) {
        }

        SubtreeNodeIterator begin() const {
            auto it = SubtreeNodeIterator(tree_, rootNodeId_, tree_ ? tree_->topologyVersion : 0);
            if (it != SubtreeNodeIterator()) {
                ++it;
            }
            return it;
        }

        SubtreeNodeIterator end() const { return SubtreeNodeIterator(); }
    };

    /**
     * @brief Iterator over the proper parts directly owned by a node.
     */
    class DynamicComponentTree::ProperPartsIterator {
    private:
        const DynamicComponentTree *tree_ = nullptr;
        index_t current_ = invalid_index;
        std::size_t expectedVersion_ = 0;

    public:
        using value_type = index_t;
        using difference_type = std::ptrdiff_t;
        using pointer = const index_t *;
        using reference = const index_t &;
        using iterator_category = std::forward_iterator_tag;

        ProperPartsIterator() = default;
        ProperPartsIterator(const DynamicComponentTree *tree, index_t current, std::size_t expectedVersion)
                : tree_(tree), current_(current), expectedVersion_(expectedVersion) {
        }

        index_t operator*() const {
            tree_->checkProperPartIteratorVersion(expectedVersion_);
            return current_;
        }

        ProperPartsIterator &operator++() {
            tree_->checkProperPartIteratorVersion(expectedVersion_);
            if (current_ != invalid_index) {
                current_ = tree_->nextProperPart[(size_t) current_];
            }
            return *this;
        }

        bool operator==(const ProperPartsIterator &other) const {
            return tree_ == other.tree_ && current_ == other.current_;
        }

        bool operator!=(const ProperPartsIterator &other) const { return !(*this == other); }
    };

    /**
     * @brief Range over the proper parts directly owned by a node.
     * @usage `for (auto p : tree.getProperParts(nodeId)) { ... }`
     * @complexity Time O(1) to create, O(k) to iterate, Space O(1), where k is the number of proper parts.
     * @warning Lazy range. Invalidated if proper-part ownership changes during iteration.
     */
    class DynamicComponentTree::ProperPartsRange {
    private:
        const DynamicComponentTree *tree_ = nullptr;
        index_t first_ = invalid_index;
        std::size_t expectedVersion_ = 0;

    public:
        ProperPartsRange() = default;
        ProperPartsRange(const DynamicComponentTree *tree, index_t first)
                : tree_(tree), first_(first), expectedVersion_(tree ? tree->properPartVersion : 0) {
        }

        ProperPartsIterator begin() const { return ProperPartsIterator(tree_, first_, expectedVersion_); }
        ProperPartsIterator end() const { return ProperPartsIterator(tree_, invalid_index, expectedVersion_); }
    };

    /**
     * @brief Iterator over an internal rooted subtree in breadth-first order.
     */
    class DynamicComponentTree::BreadthFirstNodeIterator {
    private:
        const DynamicComponentTree *tree_ = nullptr;
        std::vector<index_t> queue_;
        size_t head_ = 0;
        std::size_t expectedVersion_ = 0;

        bool empty() const { return head_ >= queue_.size(); }

    public:
        using value_type = index_t;
        using difference_type = std::ptrdiff_t;
        using pointer = const index_t *;
        using reference = const index_t &;
        using iterator_category = std::input_iterator_tag;

        BreadthFirstNodeIterator() = default;
        BreadthFirstNodeIterator(const DynamicComponentTree *tree, index_t rootNodeId, std::size_t expectedVersion)
                : tree_(tree), expectedVersion_(expectedVersion) {
            if (!tree_ || rootNodeId < tree_->getNumTotalProperParts() || rootNodeId >= tree_->getGlobalIdSpaceSize()) {
                head_ = 0;
                return;
            }
            queue_.push_back(rootNodeId);
        }

        index_t operator*() const {
            tree_->checkTopologyIteratorVersion(expectedVersion_);
            return queue_[head_];
        }

        BreadthFirstNodeIterator &operator++() {
            tree_->checkTopologyIteratorVersion(expectedVersion_);
            if (empty()) {
                return *this;
            }
            const auto nodeId = queue_[head_++];
            auto nodeLocal = tree_->localOf(nodeId);
            auto childLocal = tree_->firstChild[(size_t) nodeLocal];
            while (childLocal != invalid_index) {
                queue_.push_back(tree_->globalOf(childLocal));
                childLocal = tree_->nextSibling[(size_t) childLocal];
            }
            return *this;
        }

        bool operator==(const BreadthFirstNodeIterator &other) const { return empty() == other.empty(); }
        bool operator!=(const BreadthFirstNodeIterator &other) const { return !(*this == other); }
    };

    /**
     * @brief Range over an internal rooted subtree in breadth-first order.
     * @usage `for (auto n : tree.getBreadthFirstNodes()) { ... }`
     * @complexity Time O(1) to create, O(S) to iterate, Space O(w), where S is subtree size and w the frontier width.
     * @warning Lazy range. Invalidated if the tree topology changes during iteration.
     */
    class DynamicComponentTree::BreadthFirstNodeRange {
    private:
        const DynamicComponentTree *tree_ = nullptr;
        index_t rootNodeId_ = invalid_index;
        std::size_t expectedVersion_ = 0;

    public:
        BreadthFirstNodeRange() = default;
        BreadthFirstNodeRange(const DynamicComponentTree *tree, index_t rootNodeId)
                : tree_(tree), rootNodeId_(rootNodeId), expectedVersion_(tree ? tree->topologyVersion : 0) {
        }

        BreadthFirstNodeIterator begin() const { return BreadthFirstNodeIterator(tree_, rootNodeId_, expectedVersion_); }
        BreadthFirstNodeIterator end() const { return BreadthFirstNodeIterator(); }
    };

    inline DynamicComponentTree::AliveNodeRange DynamicComponentTree::getAliveNodeIds() const {
        return AliveNodeRange(this, getNumTotalProperParts(), getGlobalIdSpaceSize());
    }

    inline DynamicComponentTree::ChildrenRange DynamicComponentTree::getChildren(index_t nodeId) const {
        return ChildrenRange(this, firstChild[(size_t) localOf(nodeId)]);
    }

    inline DynamicComponentTree::ProperPartsRange DynamicComponentTree::getProperParts(index_t nodeId) const {
        return ProperPartsRange(this, properHead[(size_t) localOf(nodeId)]);
    }

    inline DynamicComponentTree::PostOrderNodeRange DynamicComponentTree::getPostOrderNodes() const {
        return PostOrderNodeRange(this, getRoot());
    }

    inline DynamicComponentTree::PostOrderNodeRange DynamicComponentTree::getPostOrderNodes(index_t rootNodeId) const {
        return PostOrderNodeRange(this, rootNodeId);
    }

    inline DynamicComponentTree::BreadthFirstNodeRange DynamicComponentTree::getBreadthFirstNodes() const {
        return BreadthFirstNodeRange(this, getRoot());
    }

    inline DynamicComponentTree::BreadthFirstNodeRange DynamicComponentTree::getBreadthFirstNodes(index_t rootNodeId) const {
        return BreadthFirstNodeRange(this, rootNodeId);
    }

    inline DynamicComponentTree::PathToRootRange DynamicComponentTree::getPathToRootNodes(index_t nodeId) const {
        return PathToRootRange(this, nodeId);
    }

    inline DynamicComponentTree::SubtreeNodeRange DynamicComponentTree::getNodeSubtree(index_t nodeId) const {
        return SubtreeNodeRange(this, nodeId);
    }

    inline DynamicComponentTree::DescendantNodeRange DynamicComponentTree::getDescendants(index_t nodeId) const {
        return DescendantNodeRange(this, nodeId);
    }

} // namespace hg::detail::hierarchy
