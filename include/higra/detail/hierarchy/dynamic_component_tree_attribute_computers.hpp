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
     * The computer keeps per-node mutable auxiliary buffers (`xmin`, `xmax`, `ymin`,
     * `ymax`, `empty`) so `preProcessing`, `mergeProcessing`, and `postProcessing` can
     * share intermediate state. Consequently, a single-node refresh with
     * `computeAttributeOnNode(...)` is valid only when applied bottom-up on the affected
     * path, after all edited children have themselves been refreshed.
     */
    template<typename pixel_to_point_t, typename value_t = double>
    class DynamicComponentTreeBoundingBoxAttributeComputer : public DynamicComponentTreeAttributeComputer<value_t> {
    public:
        using tree_type = typename DynamicComponentTreeAttributeComputer<value_t>::tree_type;
        using buffer_type = typename DynamicComponentTreeAttributeComputer<value_t>::buffer_type;

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
            xmin.resize(size, 0);
            xmax.resize(size, 0);
            ymin.resize(size, 0);
            ymax.resize(size, 0);
            empty.resize(size, true);
        }

        /**
         * @brief Initializes the node-local bounding box from its direct proper parts.
         */
        void preProcessing(index_t nodeId, const tree_type &tree, buffer_type &) const override {
            xmin[(size_t) nodeId] = std::numeric_limits<index_t>::max();
            xmax[(size_t) nodeId] = std::numeric_limits<index_t>::lowest();
            ymin[(size_t) nodeId] = std::numeric_limits<index_t>::max();
            ymax[(size_t) nodeId] = std::numeric_limits<index_t>::lowest();
            empty[(size_t) nodeId] = true;

            for (auto pixelId: tree.getProperParts(nodeId)) {
                const auto point = pixelToPoint(pixelId);
                const auto y = (index_t) point.first;
                const auto x = (index_t) point.second;
                xmin[(size_t) nodeId] = std::min(xmin[(size_t) nodeId], x);
                xmax[(size_t) nodeId] = std::max(xmax[(size_t) nodeId], x);
                ymin[(size_t) nodeId] = std::min(ymin[(size_t) nodeId], y);
                ymax[(size_t) nodeId] = std::max(ymax[(size_t) nodeId], y);
                empty[(size_t) nodeId] = false;
            }
        }

        /**
         * @brief Expands the parent bounding box with one already-computed child box.
         */
        void mergeProcessing(index_t parentId, index_t childId, const tree_type &, buffer_type &) const override {
            if (empty[(size_t) childId]) {
                return;
            }
            if (empty[(size_t) parentId]) {
                xmin[(size_t) parentId] = xmin[(size_t) childId];
                xmax[(size_t) parentId] = xmax[(size_t) childId];
                ymin[(size_t) parentId] = ymin[(size_t) childId];
                ymax[(size_t) parentId] = ymax[(size_t) childId];
                empty[(size_t) parentId] = false;
                return;
            }

            xmin[(size_t) parentId] = std::min(xmin[(size_t) parentId], xmin[(size_t) childId]);
            xmax[(size_t) parentId] = std::max(xmax[(size_t) parentId], xmax[(size_t) childId]);
            ymin[(size_t) parentId] = std::min(ymin[(size_t) parentId], ymin[(size_t) childId]);
            ymax[(size_t) parentId] = std::max(ymax[(size_t) parentId], ymax[(size_t) childId]);
        }

        /**
         * @brief Converts the accumulated box extents into the requested scalar measure.
         */
        void postProcessing(index_t nodeId, const tree_type &, buffer_type &buffer) const override {
            if (empty[(size_t) nodeId]) {
                buffer[(size_t) nodeId] = value_t{};
                return;
            }

            const auto width = (value_t) (xmax[(size_t) nodeId] - xmin[(size_t) nodeId] + 1);
            const auto height = (value_t) (ymax[(size_t) nodeId] - ymin[(size_t) nodeId] + 1);

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

    private:
        pixel_to_point_t pixelToPoint;
        DynamicComponentTreeBoundingBoxMeasure measure;
        mutable std::vector<index_t> xmin;
        mutable std::vector<index_t> xmax;
        mutable std::vector<index_t> ymin;
        mutable std::vector<index_t> ymax;
        mutable std::vector<bool> empty;
    };

} // namespace hg::detail::hierarchy
