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

#include <cstdint>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "higra/detail/hierarchy/dynamic_component_tree.hpp"

namespace hg::detail::hierarchy {

    /**
     * @brief Base class for dynamic-tree scalar attribute computers.
     * @details The class defines the bottom-up reduction protocol through
     * `preProcessing`, `mergeProcessing`, and `postProcessing`, and provides the
     * execution helpers `computeAttribute(...)` and `computeAttributeOnNode(...)`.
     * Released dynamic-tree slots remain addressable, so buffers are sized on the
     * full global id space.
     */
    template<typename value_t = double>
    class DynamicComponentTreeAttributeComputer {
    public:
        using tree_type = DynamicComponentTree;
        using buffer_type = std::vector<value_t>;

        virtual ~DynamicComponentTreeAttributeComputer() = default;

        virtual void resize(const tree_type &tree, buffer_type &buffer) const {
            buffer.resize((size_t) tree.getGlobalIdSpaceSize(), value_t{});
        }

        virtual void preProcessing(index_t nodeId, const tree_type &tree, buffer_type &buffer) const = 0;

        virtual void mergeProcessing(index_t parentId, index_t childId, const tree_type &tree, buffer_type &buffer) const = 0;

        virtual void postProcessing(index_t nodeId, const tree_type &tree, buffer_type &buffer) const = 0;

        /**
         * @brief Incremental-update hooks called by the tree on structural edits.
         * @details The default implementations are no-ops so concrete computers can
         * override only the events they actually exploit.
         */
        virtual void onMoveProperParts(index_t, index_t, const tree_type &) const { }
        virtual void onMoveProperPart(index_t, index_t, index_t, const tree_type &) const { }
        virtual void onNodeRemoved(index_t, const tree_type &) const { }

        /**
         * @brief Computes the attribute on the full tree by a bottom-up pass.
         */
        void computeAttribute(const tree_type &tree, buffer_type &buffer) const {
            resize(tree, buffer);
            for (auto nodeId: tree.getPostOrderNodes()) {
                preProcessing(nodeId, tree, buffer);
                for (auto childId: tree.getChildren(nodeId)) {
                    mergeProcessing(nodeId, childId, tree, buffer);
                }
                postProcessing(nodeId, tree, buffer);
            }
        }

        /**
         * @brief Recomputes one node assuming its direct children are already up to date.
         */
        void computeAttributeOnNode(const tree_type &tree, index_t nodeId, buffer_type &buffer) const {
            preProcessing(nodeId, tree, buffer);
            for (auto childId: tree.getChildren(nodeId)) {
                mergeProcessing(nodeId, childId, tree, buffer);
            }
            postProcessing(nodeId, tree, buffer);
        }
    };

    /**
     * @brief Incremental area computer for the internal dynamic component tree.
     * @details Computes the canonical increasing area attribute by counting the
     * proper parts directly owned by each node and accumulating child areas. The
     * single-node refresh contract is therefore straightforward: once child areas are
     * current, recomputing a parent requires only its direct proper-part count and the
     * cached values of its direct children.
     */
    template<typename value_t = double>
    class DynamicComponentTreeAreaAttributeComputer : public DynamicComponentTreeAttributeComputer<value_t> {
    public:
        using tree_type = typename DynamicComponentTreeAttributeComputer<value_t>::tree_type;
        using buffer_type = typename DynamicComponentTreeAttributeComputer<value_t>::buffer_type;

        void preProcessing(index_t nodeId, const tree_type &tree, buffer_type &buffer) const override {
            buffer[(size_t) nodeId] = (value_t) tree.getNumProperParts(nodeId);
        }

        void mergeProcessing(index_t parentId, index_t childId, const tree_type &, buffer_type &buffer) const override {
            buffer[(size_t) parentId] += buffer[(size_t) childId];
        }

        void postProcessing(index_t, const tree_type &, buffer_type &) const override { }
    };


    /**
     * @brief Scalar measures derivable from an axis-aligned bounding box.
     */
    enum class DynamicComponentTreeBoundingBoxMeasure {
        width,
        height,
        diagonal_length
    };

    /**
     * @brief Incremental bounding-box attribute computer for dynamic component trees.
     * @details The constructor receives a pixel-to-point functor so the tree backend
     * stays independent from any specific image embedding. The resulting scalar can
     * be the box width, height, or diagonal length.
     *
     * The computer keeps per-node mutable auxiliary buffers: `local_` stores the box of
     * direct proper parts together with extremum multiplicities, while `subtree_` stores
     * the accumulated box after child merges. Consequently, a single-node refresh with
     * `computeAttributeOnNode(...)` is valid only when applied bottom-up on the affected
     * path, after all edited children have themselves been refreshed.
     */
    template<typename pixel_to_point_t, typename value_t = double>
    class DynamicComponentTreeBoundingBoxAttributeComputer : public DynamicComponentTreeAttributeComputer<value_t> {
    public:
        using tree_type = typename DynamicComponentTreeAttributeComputer<value_t>::tree_type;
        using buffer_type = typename DynamicComponentTreeAttributeComputer<value_t>::buffer_type;

    private:
        /**
         * @brief Cached subtree box produced during the current bottom-up reduction.
         */
        struct DynamicComponentTreeBoundingBoxState {
            index_t xmin = 0;
            index_t xmax = -1;
            index_t ymin = 0;
            index_t ymax = -1;
            std::uint8_t empty = 1;
        };

        /**
         * @brief Cached box of direct proper parts plus enough metadata for local updates.
         */
        struct DynamicComponentTreeBoundingBoxLocalSummary {
            index_t xmin = 0;
            index_t xmax = -1;
            index_t ymin = 0;
            index_t ymax = -1;
            index_t xminCount = 0;
            index_t xmaxCount = 0;
            index_t yminCount = 0;
            index_t ymaxCount = 0;
            index_t properPartCount = 0;
            std::uint8_t empty = 1;
            std::uint8_t dirty = 1;
        };

        pixel_to_point_t pixelToPoint;
        DynamicComponentTreeBoundingBoxMeasure measure;
        mutable std::vector<DynamicComponentTreeBoundingBoxLocalSummary> local_;
        mutable std::vector<DynamicComponentTreeBoundingBoxState> subtree_;

        /**
         * @brief Restores an empty local box and clears all extremum multiplicities.
         */
        void resetLocalBox(DynamicComponentTreeBoundingBoxLocalSummary &local) const {
            local.xmin = 0;
            local.xmax = -1;
            local.ymin = 0;
            local.ymax = -1;
            local.xminCount = 0;
            local.xmaxCount = 0;
            local.yminCount = 0;
            local.ymaxCount = 0;
            local.properPartCount = 0;
            local.empty = 1;
        }

        /**
         * @brief Resets one node-local cache entry and marks it as already synchronized.
         */
        void resetLocalSummary(index_t nodeId) const {
            auto &local = local_[(size_t) nodeId];
            resetLocalBox(local);
            local.dirty = 0;
        }

        /**
         * @brief Clears the subtree cache associated with one node.
         */
        void resetSubtreeSummary(index_t nodeId) const {
            auto &subtree = subtree_[(size_t) nodeId];
            subtree.xmin = 0;
            subtree.xmax = -1;
            subtree.ymin = 0;
            subtree.ymax = -1;
            subtree.empty = 1;
        }

        /**
         * @brief Enlarges a local box with one proper part and updates extremum counts.
         */
        void expandLocalBoxWithPixel(DynamicComponentTreeBoundingBoxLocalSummary &local, index_t pixelId) const {
            const auto point = pixelToPoint(pixelId);
            const auto y = (index_t) point.first;
            const auto x = (index_t) point.second;
            if (local.empty) {
                local.xmin = x;
                local.xmax = x;
                local.ymin = y;
                local.ymax = y;
                local.xminCount = 1;
                local.xmaxCount = 1;
                local.yminCount = 1;
                local.ymaxCount = 1;
                local.empty = 0;
                return;
            }

            if (x < local.xmin) {
                local.xmin = x;
                local.xminCount = 1;
            } else if (x == local.xmin) {
                ++local.xminCount;
            }

            if (x > local.xmax) {
                local.xmax = x;
                local.xmaxCount = 1;
            } else if (x == local.xmax) {
                ++local.xmaxCount;
            }

            if (y < local.ymin) {
                local.ymin = y;
                local.yminCount = 1;
            } else if (y == local.ymin) {
                ++local.yminCount;
            }

            if (y > local.ymax) {
                local.ymax = y;
                local.ymaxCount = 1;
            } else if (y == local.ymax) {
                ++local.ymaxCount;
            }
        }

        /**
         * @brief Rebuilds the local summary from the current proper parts of the node.
         */
        void rebuildLocalBox(index_t nodeId, const tree_type &tree) const {
            auto &local = local_[(size_t) nodeId];
            resetLocalBox(local);
            for (auto pixelId: tree.getProperParts(nodeId)) {
                expandLocalBoxWithPixel(local, pixelId);
            }
            local.properPartCount = tree.getNumProperParts(nodeId);
            local.dirty = 0;
        }

        /**
         * @brief Lazily refreshes the local summary when a previous edit invalidated it.
         */
        void ensureLocalSummary(index_t nodeId, const tree_type &tree) const {
            auto &local = local_[(size_t) nodeId];
            const auto properPartCount = tree.getNumProperParts(nodeId);
            if (!local.dirty && local.properPartCount == properPartCount) {
                return;
            }
            rebuildLocalBox(nodeId, tree);
        }

        /**
         * @brief Seeds the subtree cache of a node with its own local bounding box.
         */
        void copyLocalToSubtree(index_t nodeId) const {
            const auto &local = local_[(size_t) nodeId];
            auto &subtree = subtree_[(size_t) nodeId];
            subtree.xmin = local.xmin;
            subtree.xmax = local.xmax;
            subtree.ymin = local.ymin;
            subtree.ymax = local.ymax;
            subtree.empty = local.empty;
        }

        /**
         * @brief Unions a child subtree box into an accumulated parent subtree box.
         */
        void mergeSubtreeStates(DynamicComponentTreeBoundingBoxState &target, const DynamicComponentTreeBoundingBoxState &source) const {
            if (source.empty) {
                return;
            }
            if (target.empty) {
                target = source;
                return;
            }
            target.xmin = std::min(target.xmin, source.xmin);
            target.xmax = std::max(target.xmax, source.xmax);
            target.ymin = std::min(target.ymin, source.ymin);
            target.ymax = std::max(target.ymax, source.ymax);
            target.empty = 0;
        }

        /**
         * @brief Merges two local summaries after moving all proper parts from source to target.
         */
        void mergeLocalBoxes(DynamicComponentTreeBoundingBoxLocalSummary &target, const DynamicComponentTreeBoundingBoxLocalSummary &source) const {
            if (source.empty) {
                return;
            }

            if (target.empty) {
                target = source;
                return;
            }

            if (source.xmin < target.xmin) {
                target.xmin = source.xmin;
                target.xminCount = source.xminCount;
            } else if (source.xmin == target.xmin) {
                target.xminCount += source.xminCount;
            }

            if (source.xmax > target.xmax) {
                target.xmax = source.xmax;
                target.xmaxCount = source.xmaxCount;
            } else if (source.xmax == target.xmax) {
                target.xmaxCount += source.xmaxCount;
            }

            if (source.ymin < target.ymin) {
                target.ymin = source.ymin;
                target.yminCount = source.yminCount;
            } else if (source.ymin == target.ymin) {
                target.yminCount += source.yminCount;
            }

            if (source.ymax > target.ymax) {
                target.ymax = source.ymax;
                target.ymaxCount = source.ymaxCount;
            } else if (source.ymax == target.ymax) {
                target.ymaxCount += source.ymaxCount;
            }

            target.properPartCount += source.properPartCount;
            target.empty = 0;
            target.dirty = 0;
        }

    public:
        explicit DynamicComponentTreeBoundingBoxAttributeComputer(pixel_to_point_t pixelToPoint,
                                                                  DynamicComponentTreeBoundingBoxMeasure measure = DynamicComponentTreeBoundingBoxMeasure::diagonal_length)
            : pixelToPoint(std::move(pixelToPoint)), measure(measure) {
        }

        /**
         * @brief Resizes the output and auxiliary buffers on the full dynamic-tree id space.
         */
        void resize(const tree_type &tree, buffer_type &buffer) const override {
            DynamicComponentTreeAttributeComputer<value_t>::resize(tree, buffer);
            const auto size = (size_t) tree.getGlobalIdSpaceSize();
            local_.resize(size);
            subtree_.resize(size);
        }

        /**
         * @brief Initializes the node-local bounding box from its direct proper parts.
         */
        void preProcessing(index_t nodeId, const tree_type &tree, buffer_type &) const override {
            ensureLocalSummary(nodeId, tree);
            copyLocalToSubtree(nodeId);
        }

        /**
         * @brief Expands the parent bounding box with one already-computed child box.
         */
        void mergeProcessing(index_t parentId, index_t childId, const tree_type &, buffer_type &) const override {
            mergeSubtreeStates(subtree_[(size_t) parentId], subtree_[(size_t) childId]);
        }

        /**
         * @brief Converts the accumulated box extents into the requested scalar measure.
         */
        void postProcessing(index_t nodeId, const tree_type &, buffer_type &buffer) const override {
            const auto &subtree = subtree_[(size_t) nodeId];
            if (subtree.empty) {
                buffer[(size_t) nodeId] = value_t{};
                return;
            }

            const auto width = (value_t) (subtree.xmax - subtree.xmin + 1);
            const auto height = (value_t) (subtree.ymax - subtree.ymin + 1);

            switch (measure) {
                case DynamicComponentTreeBoundingBoxMeasure::width:
                    buffer[(size_t) nodeId] = width;
                    break;
                case DynamicComponentTreeBoundingBoxMeasure::height:
                    buffer[(size_t) nodeId] = height;
                    break;
                case DynamicComponentTreeBoundingBoxMeasure::diagonal_length:
                    buffer[(size_t) nodeId] = (value_t) std::sqrt(width * width + height * height);
                    break;
            }
        }

        /**
         * @brief Updates local caches after a bulk transfer of all proper parts from source to target.
         */
        void onMoveProperParts(index_t targetId, index_t sourceId, const tree_type &tree) const override {
            ensureLocalSummary(targetId, tree);
            ensureLocalSummary(sourceId, tree);
            mergeLocalBoxes(local_[(size_t) targetId], local_[(size_t) sourceId]);
            resetLocalSummary(sourceId);
        }

        /**
         * @brief Updates local caches after moving a single proper part between two nodes.
         * @details The target cache is updated eagerly. The source cache stays exact as long as
         * the removed pixel does not exhaust one extremum; otherwise it is marked dirty and
         * rebuilt lazily on the next access.
         */
        void onMoveProperPart(index_t targetId, index_t sourceId, index_t pixelId, const tree_type &tree) const override {
            ensureLocalSummary(targetId, tree);
            if (sourceId != invalid_index) {
                ensureLocalSummary(sourceId, tree);
            }
            expandLocalBoxWithPixel(local_[(size_t) targetId], pixelId);
            local_[(size_t) targetId].properPartCount += 1;
            local_[(size_t) targetId].dirty = 0;
            if (sourceId != invalid_index) {
                auto &source = local_[(size_t) sourceId];
                if (source.properPartCount <= 1) {
                    resetLocalSummary(sourceId);
                    return;
                }

                const auto point = pixelToPoint(pixelId);
                const auto y = (index_t) point.first;
                const auto x = (index_t) point.second;
                --source.properPartCount;

                bool exhaustsXmin = false;
                bool exhaustsXmax = false;
                bool exhaustsYmin = false;
                bool exhaustsYmax = false;

                if (!source.dirty) {
                    if (x == source.xmin) {
                        if (source.xminCount <= 1) {
                            exhaustsXmin = true;
                        } else {
                            --source.xminCount;
                        }
                    }

                    if (x == source.xmax) {
                        if (source.xmaxCount <= 1) {
                            exhaustsXmax = true;
                        } else {
                            --source.xmaxCount;
                        }
                    }

                    if (y == source.ymin) {
                        if (source.yminCount <= 1) {
                            exhaustsYmin = true;
                        } else {
                            --source.yminCount;
                        }
                    }

                    if (y == source.ymax) {
                        if (source.ymaxCount <= 1) {
                            exhaustsYmax = true;
                        } else {
                            --source.ymaxCount;
                        }
                    }
                }

                source.dirty = source.dirty || exhaustsXmin || exhaustsXmax || exhaustsYmin || exhaustsYmax;
            }
        }

        /**
         * @brief Clears cached summaries of a removed node so stale boxes cannot be reused.
         */
        void onNodeRemoved(index_t nodeId, const tree_type &) const override {
            resetLocalSummary(nodeId);
            resetSubtreeSummary(nodeId);
        }

    
    };

} // namespace hg::detail::hierarchy
