/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#pragma once

#include <deque>
#include <limits>
#include <algorithm>
#include "xtensor/xindex_view.hpp"
#include "xtensor/xview.hpp"
#include "higra/accumulator/accumulator.hpp"
#include "higra/accumulator/tree_accumulator.hpp"
#include "higra/graph.hpp"
#include "higra/hierarchy/hierarchy_core.hpp"



namespace hg {

    namespace tree_energy_optimization_internal {

        /**
         * One piece of a piecewise energy linear function.
         * A piece represents the line with the given slope and passing through the point (origin_x, origin_y).
         */
        template<typename value_type=double>
        class piecewise_linear_energy_function_piece {
        public:

            piecewise_linear_energy_function_piece(value_type origin_x, value_type origin_y, value_type slope) :
                    m_origin_x(origin_x), m_origin_y(origin_y), m_slope(slope) {}

            double operator()(value_type x) const {
                return m_origin_y + m_slope * (x - m_origin_x);
            }

            value_type &origin_x() {
                return m_origin_x;
            }

            const value_type &origin_x() const {
                return m_origin_x;
            }

            value_type &origin_y() {
                return m_origin_y;
            }

            const value_type &origin_y() const {
                return m_origin_y;
            }

            value_type &slope() {
                return m_slope;
            }

            const value_type &slope() const {
                return m_slope;
            }

            bool
            operator==(const piecewise_linear_energy_function_piece<value_type> &rhs) const {
                return almost_equal(m_origin_x, rhs.m_origin_x) &&
                       almost_equal(m_origin_y, rhs.m_origin_y) &&
                       almost_equal(m_slope, rhs.m_slope);
            }


            bool
            operator!=(const piecewise_linear_energy_function_piece<value_type> &rhs) const {
                return !(*this == rhs);
            }


        private:

            static
            bool almost_equal(value_type a, value_type b) {
                double epsilon = 1e-5;
                return fabs(a - b) < epsilon;
            }

            value_type m_origin_x;
            value_type m_origin_y;
            value_type m_slope;
        };

        /**
         * Piecewise linear energy function as modelled in:
         * 
         *  Laurent Guigues, Jean Pierre Cocquerez, Hervé Le Men. Scale-sets Image Analysis. International
         *  Journal of Computer Vision, Springer Verlag, 2006, 68 (3), pp.289-317
         *  
         * An energy function is a concave non decreasing piecewise linear positive function.
         */
        template<typename value_type=double>
        class piecewise_linear_energy_function {

        public:
            using lp_t = piecewise_linear_energy_function_piece<value_type>;
            using self_type = piecewise_linear_energy_function<value_type>;

            piecewise_linear_energy_function() {};

            piecewise_linear_energy_function(const lp_t &piece) {
                pieces.push_back(piece);
            }

            piecewise_linear_energy_function(lp_t &&piece) {
                pieces.push_back(std::forward<lp_t>(piece));
            }

            piecewise_linear_energy_function(std::initializer_list<lp_t> pieces_l) : pieces(pieces_l) {}

            void add_piece(const lp_t &piece) {
                pieces.push_back(piece);
            }

            void add_piece(lp_t &&piece) {
                pieces.push_back(std::forward<lp_t>(piece));
            }

            /**
             * Computes the sum between two piecewise_linear_energy_function
             * The computation is by default limited to the max_pieces largest pieces (right most)
             * 
             * @param other 
             * @return 
             */
            self_type sum(const self_type &other, int max_pieces = 10) const {

                if (other.size() == 0) {
                    return self_type(*this);
                } else if (size() == 0) {
                    return self_type(other);
                }

                self_type result = self_type();
                int count = 0;
                int i1 = pieces.size() - 1;
                int i2 = other.pieces.size() - 1;
                while (i1 >= 0 && i2 >= 0 && count < max_pieces) {
                    const auto &piece1 = pieces[i1];
                    const auto &piece2 = other.pieces[i2];
                    auto new_slope = piece1.slope() + piece2.slope();
                    value_type new_origin_x, new_origin_y;
                    if (piece1.origin_x() >= piece2.origin_x()) {
                        new_origin_x = piece1.origin_x();
                        new_origin_y = piece1.origin_y() + piece2(piece1.origin_x());
                        if (piece1.origin_x() == piece2.origin_x()) {
                            i2--;
                        }
                        i1--;
                    } else {
                        new_origin_x = piece2.origin_x();
                        new_origin_y = piece2.origin_y() + piece1(piece2.origin_x());
                        i2--;
                    }

                    result.pieces.push_front(
                            lp_t(new_origin_x, new_origin_y, new_slope));
                    count++;
                }

                if (result.pieces.size() > 0) {
                    auto &first_piece = result.pieces[0];
                    if (first_piece.origin_x() > 0) {
                        first_piece.origin_y() -= first_piece.slope() * first_piece.origin_x();
                        first_piece.origin_x() = 0;
                    }
                }
                return result;
            }

            /**
             * Infimum between the current piecewise linear energy function and the given linear piece.
             *
             * Returns the abscissa of the intersection between the two functions and infinity if no intersection exists
             *
             * PRECONDITION :
             *   this->size() > 0
             *   (*this)[0].origin_x() == 0
             *   linear_piece.origin_x() == 0
             *   (*this)[this->size() - 1].slope() >= linear_piece.slope()
             *
             *
             * Warning: Modification is done in place
             */
            double infimum(const lp_t &linear_piece) {
                int i = pieces.size() - 1;

                auto &last_piece = pieces[i];
                if (linear_piece.slope() == last_piece.slope()) {
                    auto y = linear_piece(last_piece.origin_x());
                    if (y > last_piece.origin_y()) {
                        return std::numeric_limits<value_type>::infinity();
                    } else if (y == last_piece.origin_y()) {
                        return last_piece.origin_x();
                    } else {
                        pieces.pop_back();
                        i--;
                    }
                }

                value_type xi = 0;
                bool flag = true;
                while (i >= 0 && flag) {
                    auto &piece = pieces[i];
                    xi = (linear_piece.origin_x() * linear_piece.slope() - piece.origin_x() * piece.slope() -
                          (linear_piece.origin_y() - piece.origin_y())) / (linear_piece.slope() - piece.slope());
                    if (xi > piece.origin_x()) {
                        flag = false;
                    } else {
                        pieces.pop_back();
                    }
                    i--;
                }
                pieces.push_back(lp_t(xi, linear_piece(xi), linear_piece.slope()));
                return xi;
            }

            bool
            operator==(const self_type &rhs) const {
                if (pieces.size() != rhs.pieces.size())
                    return false;
                return std::equal(pieces.begin(), pieces.end(), rhs.pieces.begin());
            }


            bool
            operator!=(const self_type &rhs) const {
                return !(*this == rhs);
            }

            size_t size() const {
                return pieces.size();
            }

            const auto begin() const {
                return pieces.cbegin();
            }

            const auto end() const {
                return pieces.cend();
            }

            auto begin() {
                return pieces.begin();
            }

            auto end() {
                return pieces.end();
            }

            const auto operator[](index_t i) const {
                return pieces[i];
            }

            auto operator[](index_t i) {
                return pieces[i];
            }

        private:
            std::deque<lp_t> pieces;
        };
    }

    /**
     * Computes the labelisation of the input tree leaves corresponding to the optimal cut according to the given energy attribute.
     *
     * Given a node i, the value energy_attribute(i) represents the energy fo the partial partition composed of the single region i.
     * Given a node i, the energy of the partial partition composed of the children of i is given by accumulator(energy_attribute(children(i))).
     *
     * This function computes the partition (ie. a set of node forming a cut of the tree) that has a minimal energy
     * according to the definition above.
     *
     * The algorithm used is based on dynamic programming and runs in linear time w.r.t. to the number of nodes in the tree.
     *
     * See:
     *
     *  Laurent Guigues, Jean Pierre Cocquerez, Hervé Le Men. Scale-sets Image Analysis. International
     *  Journal of Computer Vision, Springer Verlag, 2006, 68 (3), pp.289-317
     *
     * and
     *
     *  Bangalore Ravi Kiran, Jean Serra. Global-local optimizations by hierarchical cuts and climbing energies.
     *  Pattern Recognition Letters, Elsevier, 2014, 47 (1), pp.12-24.
     *
     * @tparam tree_type input tree type
     * @tparam T energy attribute type
     * @tparam accumulator_type accumulator type
     * @param tree input tree
     * @param xenergy_attribute 1d array of energy attribute for the input tree
     * @param accumulator accumulator used to define how children energies are combined in order to obtain the energy of the corresponding partial partition
     * @return a 1d integer array with num_leaves(tree) elements representing the minimal energy partition
     */
    template<typename tree_type,
            typename T,
            typename accumulator_type=hg::accumulator_sum>
    auto labelisation_optimal_cut_from_energy(const tree_type &tree,
                                              const xt::xexpression<T> &xenergy_attribute,
                                              const accumulator_type accumulator = hg::accumulator_sum()) {
        using value_type = typename T::value_type;
        auto &energy_attribute = xenergy_attribute.derived_cast();
        hg_assert_node_weights(tree, energy_attribute);
        hg_assert_1d_array(energy_attribute);

        array_1d<bool> optimal_nodes = array_1d<bool>::from_shape({num_vertices(tree)});
        array_1d<value_type> optimal_energy = array_1d<value_type>::from_shape({num_vertices(tree)});

        auto energy_children = accumulate_parallel(tree, energy_attribute, accumulator);

        // forward pass
        xt::view(optimal_nodes, xt::range(0, num_leaves(tree))) = true;
        xt::view(optimal_energy, xt::range(0, num_leaves(tree))) = xt::view(energy_attribute,
                                                                            xt::range(0, num_leaves(tree)));

        for (auto i: leaves_to_root_iterator(tree, leaves_it::exclude)) {
            if (energy_attribute(i) <= energy_children(i)) {
                optimal_nodes(i) = true;
                optimal_energy(i) = energy_attribute(i);
            } else {
                optimal_nodes(i) = false;
                optimal_energy(i) = energy_children(i);
            }
        }

        //  backtracking and labelisation
        array_1d<index_t> labels = array_1d<index_t>(optimal_nodes.shape(), invalid_index);
        index_t count = 0;
        for (auto i: root_to_leaves_iterator(tree)) {
            if (labels(i) == invalid_index && optimal_nodes(i)) {
                labels(i) = count++;
            }
            if (labels(i) != invalid_index) {
                for (auto c: children_iterator(i, tree)) {
                    labels(c) = labels(i);
                }
            }
        }
        return xt::eval(xt::view(labels, xt::range(0, num_leaves(tree))));
    };

    /**
     * Transforms the given hierarchy into its optimal energy cut hierarchy for the given energy terms.
     * In the optimal energy cut hierarchy, any horizontal cut corresponds to an optimal energy cut in the original
     * hierarchy.
     *
     * Each node i of the tree is associated to a data fidelity energy D(i) and a regularization energy R(i).
     * The algorithm construct a new hierarchy with associated altitudes such that the horizontal cut of level lambda
     * is the optimal cut for the energy attribute D + lambda * R of the input tree (see function labelisation_optimal_cut_from_energy).
     * In other words, the horizontal cut of level lambda in the result is the cut of the input composed of the nodes N such that
     * sum_{r in N} D(r) + lambda * R(r) is minimal.
     *
     * PRECONDITION: the regularization energy R must be sub additive: for each node i: R(i) <= sum_{c in children(i)} R(c)
     *
     * The algorithm runs in linear time O(n)
     *
     * See:
     *
     *  Laurent Guigues, Jean Pierre Cocquerez, Hervé Le Men. Scale-sets Image Analysis. International
     *  Journal of Computer Vision, Springer Verlag, 2006, 68 (3), pp.289-317
     *
     * @tparam tree_type
     * @tparam T
     * @param tree
     * @param xdata_fidelity_attribute
     * @param xregularization_attribute
     * @return
     */
    template<typename tree_type,
            typename T>
    auto
    hierarchy_to_optimal_energy_cut_hierarchy(const tree_type &tree,
                                              const xt::xexpression<T> &xdata_fidelity_attribute,
                                              const xt::xexpression<T> &xregularization_attribute) {
        auto &data_fidelity_attribute = xdata_fidelity_attribute.derived_cast();
        auto &regularization_attribute = xregularization_attribute.derived_cast();
        hg_assert_node_weights(tree, data_fidelity_attribute);
        hg_assert_node_weights(tree, regularization_attribute);
        hg_assert_1d_array(data_fidelity_attribute);
        hg_assert_1d_array(regularization_attribute);

        using lep_t = hg::tree_energy_optimization_internal::piecewise_linear_energy_function_piece<double>;
        using lef_t = hg::tree_energy_optimization_internal::piecewise_linear_energy_function<double>;

        std::vector <lef_t> optimal_energies{};
        array_1d<double> apparition_scales = array_1d<double>::from_shape({num_vertices(tree)});

        for (auto i: leaves_iterator(tree)) {
            optimal_energies.emplace_back(lep_t(0, data_fidelity_attribute(i), regularization_attribute(i)));
            apparition_scales(i) = -data_fidelity_attribute(i) / regularization_attribute(i);
        }

        for (auto i: leaves_to_root_iterator(tree, leaves_it::exclude)) {
            optimal_energies.push_back(optimal_energies[child(0, i, tree)]);
            for (index_t c = 1; c < (index_t)num_children(i, tree); c++) {
                optimal_energies[i] = optimal_energies[i].sum(optimal_energies[child(c, i, tree)]);
            }
            apparition_scales(i) = optimal_energies[i].infimum({0, data_fidelity_attribute(i), regularization_attribute(i)});
        }

        for (auto i: root_to_leaves_iterator(tree, leaves_it::include, root_it::exclude)) {
            apparition_scales(i) = (std::max)(0.0, (std::min)(apparition_scales(i), apparition_scales(parent(i, tree))));
        }

        auto apparition_scales_parents = propagate_parallel(tree, apparition_scales);
        auto qfz = simplify_tree(tree, xt::equal(apparition_scales, apparition_scales_parents));
        auto &qfz_tree = qfz.tree;
        auto &node_map = qfz.node_map;
        auto qfz_apparition_scales = xt::eval(xt::index_view(apparition_scales, node_map));

        return make_node_weighted_tree(std::move(qfz_tree), std::move(qfz_apparition_scales));
    };

}