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
#include "xtensor/xnoalias.hpp"
#include "higra/accumulator/accumulator.hpp"
#include "higra/accumulator/tree_accumulator.hpp"
#include "higra/graph.hpp"
#include "higra/hierarchy/hierarchy_core.hpp"
#include "higra/hierarchy/binary_partition_tree.hpp"


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
                int i1 = (int)pieces.size() - 1;
                int i2 = (int)other.pieces.size() - 1;
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
                int i = (int)pieces.size() - 1;

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

        // stupid template metaprogramming for bpt function
        template<bool vectorial>
        struct container_bpt {

        };

        template<>
        struct container_bpt<true> {
            typedef array_2d<double> type;

            template<typename T>
            static auto init(const T &a) {
                type res = type::from_shape({a.shape()[0] * 2 - 1, a.shape()[1]});
                xt::view(res, xt::range(0, a.shape()[0]), xt::all()) = a;
                return res;
            }
        };

        template<>
        struct container_bpt<false> {
            typedef array_1d<double> type;

            template<typename T>
            static auto init(const T &a) {
                type res = type::from_shape({a.size() * 2 - 1});
                xt::view(res, xt::range(0, a.size())) = a;
                return res;
            }
        };

        template<bool vectorial>
        struct computation_helper {
        };

        template<>
        struct computation_helper<true> {

            template<typename T>
            static
            void add(T &a, index_t res, index_t i, index_t j) {
                for (index_t c = 0; c < (index_t) a.shape()[1]; c++) {
                    a(res, c) = a(i, c) + a(j, c);
                }
            }

            template<typename T, typename Q>
            static
            auto
            data_fidelity(const T &m, const T &m2, const Q &area, index_t i) {
                double res = 0;
                for (index_t c = 0; c < (index_t) m.shape()[1]; c++) {
                    res += m2(i, c) - m(i, c) * m(i, c) / area(i);
                }
                return res;
            }

            template<typename Q, typename T, typename R>
            static
            auto
            apparition_scale(const Q &oe, const T &area, const T &perimeter, const R &m, const R &m2,
                             index_t i, index_t j, double edge_length) {
                auto e = oe[i].sum(oe[j]);
                double a = area(i) + area(j);
                double data_fidelity = 0;
                for (index_t c = 0; c < (index_t) m.shape()[1]; c++) {
                    double mean = m(i, c) + m(j, c);
                    double mean2 = m2(i, c) + m2(j, c);
                    data_fidelity += mean2 - mean * mean / a;
                }

                auto v = e.infimum({0,
                                    data_fidelity,
                                    perimeter(i) + perimeter(j) - 2 * edge_length});
                return v;
            }

        };

        template<>
        struct computation_helper<false> {

            template<typename T>
            static
            void add(T &a, index_t res, index_t i, index_t j) {
                a(res) = a(i) + a(j);
            }

            template<typename T>
            static
            auto
            data_fidelity(const T &m, const T &m2, const T &area, index_t i) {
                return m2(i) - m(i) * m(i) / area(i);
            }

            template<typename Q, typename T, typename R>
            static
            auto
            apparition_scale(const Q &oe, const T &area, const T &perimeter, const R &m, const R &m2,
                             index_t i, index_t j, double edge_length) {
                auto e = oe[i].sum(oe[j]);

                double mean = m(i) + m(j);
                double mean2 = m2(i) + m2(j);
                double a = area(i) + area(j);

                auto v = e.infimum({0,
                                    mean2 - mean * mean / a,
                                    perimeter(i) + perimeter(j) - 2 * edge_length});
                return v;
            }

        };

        /**
         * Weighting function for binary partition tree based on Mumfor-Shah energy function
         *
         * Consider using the helper factory function make_binary_partition_tree_MumfordShah_linkage
         *
         * @tparam T
        */
        template<bool vectorial, typename graph_type, typename value_type=double>
        struct binary_partition_tree_MumfordShah_linkage_weighting_functor {
            using ctype = typename container_bpt<vectorial>::type;

            using lep_t = piecewise_linear_energy_function_piece<double>;
            using lef_t = piecewise_linear_energy_function<double>;

            std::vector<lef_t> m_optimal_energies{};
            const graph_type &m_graph;
            array_1d<double> m_area;
            array_1d<double> m_perimeter;
            array_1d<double> m_edge_length;
            ctype m_sum;
            ctype m_sum2;

            template<typename T1, typename T2, typename T3, typename T4, typename T5>
            binary_partition_tree_MumfordShah_linkage_weighting_functor(
                    const graph_type &graph,
                    const xt::xexpression<T1> &xvertex_area,
                    const xt::xexpression<T2> &xsum_vertex_weights,
                    const xt::xexpression<T3> &xsum_square_vertex_weights,
                    const xt::xexpression<T4> &xvertex_perimeter,
                    const xt::xexpression<T5> &xedge_length) :
                    m_graph(graph),
                    m_edge_length(xedge_length) {
                auto &vertex_area = xvertex_area.derived_cast();
                auto &sum_vertex_weights = xsum_vertex_weights.derived_cast();
                auto &sum_square_vertex_weights = xsum_square_vertex_weights.derived_cast();
                auto &vertex_perimeter = xvertex_perimeter.derived_cast();

                size_t num_nodes = vertex_area.size();
                size_t num_nodes_final = num_nodes * 2 - 1;
                m_area = array_1d<double>::from_shape({num_nodes_final});
                xt::noalias(xt::view(m_area, xt::range(0, num_nodes))) = vertex_area;
                m_perimeter = array_1d<double>::from_shape({num_nodes_final});
                xt::noalias(xt::view(m_perimeter, xt::range(0, num_nodes))) = vertex_perimeter;
                m_sum = container_bpt<vectorial>::init(sum_vertex_weights);
                m_sum2 = container_bpt<vectorial>::init(sum_square_vertex_weights);

                for (index_t i = 0; i < (index_t) num_nodes; i++) {
                    m_optimal_energies.emplace_back(
                            lep_t{0, computation_helper<vectorial>::data_fidelity(m_sum, m_sum2, m_area, i),
                                  m_perimeter(i)});
                }
            }

            auto weight_initial_edges() {
                array_1d<double> edge_weights = array_1d<double>::from_shape({num_edges(m_graph)});
                for (auto e: edge_iterator(m_graph)) {
                    auto s = source(e, m_graph);
                    auto t = target(e, m_graph);
                    edge_weights(e) = computation_helper<vectorial>::apparition_scale(
                            m_optimal_energies, m_area, m_perimeter, m_sum, m_sum2,
                            s, t, m_edge_length(e));
                }
                return edge_weights;
            }

            template<typename graph_t, typename neighbours_t>
            void operator()(const graph_t &g,
                            index_t fusion_edge_index,
                            index_t new_region,
                            index_t merged_region1,
                            index_t merged_region2,
                            neighbours_t &new_neighbours) {
                // compute attributes of the new region
                m_area(new_region) = m_area(merged_region1) + m_area(merged_region2);
                m_perimeter(new_region) =
                        m_perimeter(merged_region1) + m_perimeter(merged_region2) -
                        2 * m_edge_length(fusion_edge_index);
                computation_helper<vectorial>::add(m_sum, new_region, merged_region1, merged_region2);
                computation_helper<vectorial>::add(m_sum2, new_region, merged_region1, merged_region2);

                // compute energy of new region
                m_optimal_energies.push_back(
                        m_optimal_energies[merged_region1].sum(m_optimal_energies[merged_region2]));
                m_optimal_energies[new_region].infimum(
                        {0,
                         computation_helper<vectorial>::data_fidelity(m_sum, m_sum2, m_area, new_region),
                         m_perimeter(new_region)});

                // update weights of edges linking the new region
                for (auto &n: new_neighbours) {

                    // compute the length of the edge linking the new region to one of its neighbour
                    double new_edge_length;
                    if (n.num_edges() > 1) {
                        new_edge_length = m_edge_length[n.first_edge_index()] + m_edge_length[n.second_edge_index()];
                    } else {
                        new_edge_length = m_edge_length[n.first_edge_index()];
                    }
                    m_edge_length[n.new_edge_index()] = new_edge_length;

                    // the weight of the new edge is equal to the apparition scale of the region create by the merging of
                    // the two extremities of the edge
                    n.new_edge_weight() = (std::max)(0.0, computation_helper<vectorial>::apparition_scale(
                            m_optimal_energies, m_area, m_perimeter, m_sum, m_sum2,
                            new_region, n.neighbour_vertex(), new_edge_length));

                }
            }
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
     *  Laurent Guigues, Jean Pierre Cocquerez, H
     *  ervé Le Men. Scale-sets Image Analysis. International
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
        HG_TRACE();
        using value_type = typename T::value_type;
        auto &energy_attribute = xenergy_attribute.derived_cast();
        hg_assert_node_weights(tree, energy_attribute);
        hg_assert_1d_array(energy_attribute);

        tree.compute_children();
        array_1d<bool> optimal_nodes = array_1d<bool>::from_shape({num_vertices(tree)});
        array_1d<value_type> optimal_energy = array_1d<value_type>::from_shape({num_vertices(tree)});

        auto output_view = make_light_axis_view<false>(optimal_energy);
        auto acc = accumulator.template make_accumulator<false>(output_view);

        // forward pass
        xt::view(optimal_nodes, xt::range(0, num_leaves(tree))) = true;
        xt::noalias(xt::view(optimal_energy, xt::range(0, num_leaves(tree)))) =
                xt::view(energy_attribute, xt::range(0, num_leaves(tree)));

        for (auto i: leaves_to_root_iterator(tree, leaves_it::exclude)) {
            output_view.set_position(i);
            acc.set_storage(output_view);
            acc.initialize();
            for (auto c: children_iterator(i, tree)) {
                acc.accumulate(&optimal_energy(c));
            }
            acc.finalize();
            if (energy_attribute(i) <= optimal_energy(i)) {
                optimal_nodes(i) = true;
                optimal_energy(i) = energy_attribute(i);
            } else {
                optimal_nodes(i) = false;
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
     * @param tree Input tree
     * @param xdata_fidelity_attribute Data fidelity energy (1d array)
     * @param xregularization_attribute Regularization energy (1d array)
     * @param approximation_piecewise_linear_function Maximum number of pieces used in the approximated piecewise linear model for the energy.
     * @return a node_weighted_tree
     */
    template<typename tree_type,
            typename T>
    auto
    hierarchy_to_optimal_energy_cut_hierarchy(const tree_type &tree,
                                              const xt::xexpression<T> &xdata_fidelity_attribute,
                                              const xt::xexpression<T> &xregularization_attribute,
                                              const int approximation_piecewise_linear_function = 10) {
        HG_TRACE();
        auto &data_fidelity_attribute = xdata_fidelity_attribute.derived_cast();
        auto &regularization_attribute = xregularization_attribute.derived_cast();
        hg_assert_node_weights(tree, data_fidelity_attribute);
        hg_assert_node_weights(tree, regularization_attribute);
        hg_assert_1d_array(data_fidelity_attribute);
        hg_assert_1d_array(regularization_attribute);
        hg_assert(approximation_piecewise_linear_function > 0,
                  "approximation_piecewise_linear_function must be strictly positive.");


        using lep_t = hg::tree_energy_optimization_internal::piecewise_linear_energy_function_piece<double>;
        using lef_t = hg::tree_energy_optimization_internal::piecewise_linear_energy_function<double>;

        tree.compute_children();
        std::vector<lef_t> optimal_energies{};
        array_1d<double> apparition_scales = array_1d<double>::from_shape({num_vertices(tree)});

        for (auto i: leaves_iterator(tree)) {
            optimal_energies.emplace_back(lep_t(0, data_fidelity_attribute(i), regularization_attribute(i)));
            apparition_scales(i) = -data_fidelity_attribute(i) / regularization_attribute(i);
        }

        for (auto i: leaves_to_root_iterator(tree, leaves_it::exclude)) {
            optimal_energies.push_back(optimal_energies[child(0, i, tree)]);
            for (index_t c = 1; c < (index_t) num_children(i, tree); c++) {
                optimal_energies[i] = optimal_energies[i].sum(optimal_energies[child(c, i, tree)],
                                                              approximation_piecewise_linear_function);
            }
            apparition_scales(i) = optimal_energies[i].infimum(
                    {0, data_fidelity_attribute(i), regularization_attribute(i)});
        }

        for (auto i: root_to_leaves_iterator(tree, leaves_it::include, root_it::exclude)) {
            apparition_scales(i) = (std::max)(0.0,
                                              (std::min)(apparition_scales(i), apparition_scales(parent(i, tree))));
        }

        auto apparition_scales_parents = propagate_parallel(tree, apparition_scales);
        auto qfz = simplify_tree(tree, xt::equal(apparition_scales, apparition_scales_parents));
        auto &qfz_tree = qfz.tree;
        auto &node_map = qfz.node_map;
        auto qfz_apparition_scales = xt::eval(xt::index_view(apparition_scales, node_map));

        return make_node_weighted_tree(std::move(qfz_tree), std::move(qfz_apparition_scales));
    };

    /**
     * Compute the binary partition tree, i.e. the agglomerative clustering, according to the Mumford-Shah energy
     * with a constant piecewise model.
     *
     * The distance between two regions is equal to the apparition scale of the merged region.
     *
     * See:
     *
     *  Laurent Guigues, Jean Pierre Cocquerez, Hervé Le Men. Scale-sets Image Analysis. International
     *  Journal of Computer Vision, Springer Verlag, 2006, 68 (3), pp.289-317
     *
     * @tparam graph_t
     * @tparam T1
     * @tparam T2
     * @tparam T3
     * @tparam T4
     * @tparam T5
     * @param graph Input graph
     * @param xvertex_perimeter Perimeter of each vertex of the input graph
     * @param xvertex_area Area of each vertex of the input graph
     * @param xvertex_values Sum of values inside the region represented by each vertex of the input graph.
     * @param xsquared_vertex_values Sum of the squared values inside the region represented by each vertex of the input graph.
     * @param xedge_length Length of the frontier represented by each edge.
     * @return a node_weighted_tree
     */
    template<typename graph_t,
            typename T1, typename T2, typename T3, typename T4, typename T5>
    auto binary_partition_tree_MumfordShah_energy(
            const graph_t &graph,
            const xt::xexpression<T1> &xvertex_perimeter,
            const xt::xexpression<T2> &xvertex_area,
            const xt::xexpression<T3> &xvertex_values,
            const xt::xexpression<T4> &xsquared_vertex_values,
            const xt::xexpression<T5> &xedge_length) {

        auto &vertex_perimeter = xvertex_perimeter.derived_cast();
        hg_assert_vertex_weights(graph, vertex_perimeter);
        hg_assert_1d_array(vertex_perimeter);
        auto &vertex_area = xvertex_area.derived_cast();
        hg_assert_vertex_weights(graph, vertex_area);
        hg_assert_1d_array(vertex_area);
        auto &vertex_values = xvertex_values.derived_cast();
        hg_assert_vertex_weights(graph, vertex_values);
        hg_assert(vertex_values.dimension() <= 2, "Vertex values can be scalar or vectorial.");
        auto &squared_vertex_values = xsquared_vertex_values.derived_cast();
        hg_assert_same_shape(vertex_values, squared_vertex_values);
        auto &edge_length = xedge_length.derived_cast();
        hg_assert_edge_weights(graph, edge_length);
        hg_assert_1d_array(edge_length);

        if (vertex_values.dimension() == 1) {
            auto wf = tree_energy_optimization_internal::
            binary_partition_tree_MumfordShah_linkage_weighting_functor<false, graph_t>(
                    graph,
                    vertex_area,
                    vertex_values,
                    squared_vertex_values,
                    vertex_perimeter,
                    edge_length
            );
            auto edge_weights = wf.weight_initial_edges();
            return binary_partition_tree(graph, edge_weights, wf);
        } else {
            auto wf = tree_energy_optimization_internal::
            binary_partition_tree_MumfordShah_linkage_weighting_functor<true, graph_t>(
                    graph,
                    vertex_area,
                    vertex_values,
                    squared_vertex_values,
                    vertex_perimeter,
                    edge_length
            );
            auto edge_weights = wf.weight_initial_edges();
            auto res = binary_partition_tree(graph, edge_weights, wf);

            return res;
        }

    }
}