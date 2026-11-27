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

#include "../graph.hpp"
#include "../accumulator/tree_accumulator.hpp"

namespace hg {

    /**
     * Compute the area of each node of the given tree.
     *
     * Each leaf of the tree corresponds to a vertex of the leaf graph (for example a pixel) and has an area \f$a(v)\f$.
     * The area \f$A(n)\f$ of a node \f$n\f$ is the total area of the region it represents, i.e. the sum of the areas of
     * the leaves of the subtree rooted in \f$n\f$:
     *
     * \f[
     *     A(n) = \sum_{v \in leaves(n)} a(v)
     * \f]
     *
     * Equivalently, the area of a leaf is its own area, and the area of an internal node is the sum of the areas of its
     * children:
     *
     * \f[
     *     A(n) = \begin{cases}
     *         a(n) & \textrm{if } n \textrm{ is a leaf}\\
     *         \sum_{c \in children(n)} A(c) & \textrm{otherwise}
     *     \end{cases}
     * \f]
     *
     * The area of the root is thus the total area of all the leaves.
     *
     * The area of each leaf is given by leaf_area: a 1d array with one element per leaf.
     *
     * Complexity: The tree accumulation takes \f$\mathcal{O}(N)\f$ time, where \f$N\f$ is the number of nodes in the
     * tree.
     *
     * @tparam tree_t tree type
     * @tparam T element type of leaf_area
     * @param tree input tree
     * @param leaf_area area of the leaves of the input tree (1d array with one element per leaf)
     * @return a 1d array representing the area of each node of the tree
     */
    template<typename tree_t, typename T>
    auto attribute_area(const tree_t &tree, array_view<const T> leaf_area) {
        hg_assert_leaf_weights(tree, leaf_area);

        return accumulate_sequential(tree, leaf_area, accumulator_sum());
    }

    /**
     * Compute the area of each node of the given tree.
     *
     * Each leaf of the tree corresponds to a vertex of the leaf graph (for example a pixel) and has an area \f$a(v)\f$.
     * The area \f$A(n)\f$ of a node \f$n\f$ is the total area of the region it represents, i.e. the sum of the areas of
     * the leaves of the subtree rooted in \f$n\f$:
     *
     * \f[
     *     A(n) = \sum_{v \in leaves(n)} a(v)
     * \f]
     *
     * Equivalently, the area of a leaf is its own area, and the area of an internal node is the sum of the areas of its
     * children:
     *
     * \f[
     *     A(n) = \begin{cases}
     *         a(n) & \textrm{if } n \textrm{ is a leaf}\\
     *         \sum_{c \in children(n)} A(c) & \textrm{otherwise}
     *     \end{cases}
     * \f]
     *
     * The area of the root is thus the total area of all the leaves.
     *
     * In this function, the area of each leaf is 1, and the area of a node is then its number of leaves.
     *
     * Complexity: The tree accumulation takes \f$\mathcal{O}(N)\f$ time, where \f$N\f$ is the number of nodes in the
     * tree.
     *
     * @tparam tree_t tree type
     * @param tree input tree
     * @return a 1d array of integers representing the area of each node of the tree
     */
    template<typename tree_t>
    auto attribute_area(const tree_t &tree) {
        array_1d<index_t> leaf_area = hg::ones<index_t>({num_leaves(tree)});
        return attribute_area(tree, leaf_area);
    }

    /**
     * Compute the volume under the curve of each node of the given tree.
     *
     * The volume \f$V(n)\f$ of a node \f$n\f$ is defined recursively as:
     *
     * \f[
     *     V(n) = A(n) \times |alt(n) - alt(parent(n))| + \sum_{c \in children(n)} V(c)
     * \f]
     *
     * where \f$A(n)\f$ is the area of \f$n\f$ and \f$alt(n)\f$ is its altitude. The recursive formula applies to the
     * non-leaf nodes. The volume of a leaf is 0.
     *
     * The root of the tree is its own parent: its altitude difference with its parent is null, hence the root does not
     * add any volume of its own and its volume is the sum of the volumes of its children.
     *
     * Complexity: The volume accumulation takes \f$\mathcal{O}(N)\f$ time, where \f$N\f$ is the number of nodes in the
     * tree.
     *
     * @tparam tree_t tree type
     * @tparam T1 element type of node_altitude
     * @tparam T2 element type of node_area
     * @param tree input tree
     * @param node_altitude node altitudes of the input tree (1d array with one element per node of the tree)
     * @param node_area area of the nodes of the input tree (provided by attribute_area on tree)
     * @return a 1d array of double representing the volume of each node of the tree
     */
    template<typename tree_t, typename T1, typename T2>
    auto attribute_volume(const tree_t &tree, array_view<const T1> node_altitude,
                          array_view<const T2> node_area) {
        hg_assert_node_weights(tree, node_area);
        hg_assert_1d_array(node_area);
        hg_assert_node_weights(tree, node_altitude);
        hg_assert_1d_array(node_altitude);
        tree.compute_children();
        auto &parent = tree.parents();
        array_1d<double> volume = hg::empty<double>({tree.num_vertices()});
        volume.slice(0, num_leaves(tree)).fill(0);
        for (auto i: leaves_to_root_iterator(tree, leaves_it::exclude)) {
            volume(i) = std::fabs(node_altitude(i) - node_altitude(parent(i))) * node_area(i);
            for (auto c: children_iterator(i, tree)) {
                volume(i) += volume(c);
            }
        }
        return volume;
    }

    /**
     * Compute the depth of each node of the given tree.
     *
     * The depth \f$d(n)\f$ of a node \f$n\f$ is the number of edges on the path from \f$n\f$ to the root, that is the
     * number of ancestors of \f$n\f$ other than \f$n\f$ itself:
     *
     * \f[
     *     d(n) = \begin{cases}
     *         0 & \textrm{if } n \textrm{ is the root}\\
     *         d(parent(n)) + 1 & \textrm{otherwise}
     *     \end{cases}
     * \f]
     *
     * Complexity: The computation takes \f$\mathcal{O}(N)\f$ time, where \f$N\f$ is the number of nodes in the tree.
     *
     * @tparam tree_t tree type
     * @param tree input tree
     * @return a 1d array of integers representing the depth of each node of the tree
     */
    template<typename tree_t>
    auto attribute_depth(const tree_t &tree) {
        array_1d<index_t> depth = hg::empty<index_t>({tree.num_vertices()});
        if (num_vertices(tree) == 0) {
            return depth;
        }
        depth(tree.root()) = 0;
        for (auto i: root_to_leaves_iterator(tree, leaves_it::include, root_it::exclude)) {
            depth(i) = depth(parent(i, tree)) + 1;
        }
        return depth;
    };

    /**
     * Compute the height of each node of the given tree.
     *
     * In a tree \f$T\f$, given that the altitudes of the nodes vary monotonically from the leaves to the root, the
     * height \f$h(n)\f$ of a non-leaf node \f$n\f$ of \f$T\f$ is the difference between the altitude of a reference
     * node \f$ref(n)\f$ and the altitude of the deepest non-leaf node of the subtree of \f$T\f$ rooted in \f$n\f$. For
     * increasing altitudes, the deepest node is the one with the smallest altitude:
     *
     * \f[
     *     h(n) = alt(ref(n)) - \min_{m \in subtree(n) \setminus leaves(n)} alt(m)
     * \f]
     *
     * and for decreasing altitudes, it is the one with the largest altitude:
     *
     * \f[
     *     h(n) = \max_{m \in subtree(n) \setminus leaves(n)} alt(m) - alt(ref(n))
     * \f]
     *
     * where \f$subtree(n)\f$ is the set of the nodes of the subtree rooted in \f$n\f$ (including \f$n\f$ and its
     * leaves). The height is thus always non-negative. The height of a leaf is 0.
     *
     * In this function, the reference node is the parent of \f$n\f$: \f$ref(n) = parent(n)\f$ (the root being its own
     * parent).
     *
     * Possible values of increasing_altitudes are:
     *
     * - true: the altitudes are increasing from the leaves to the root (i.e. for any node \f$n\f$,
     *   \f$alt(n) \leq alt(parent(n))\f$), as in a min-tree;
     * - false: the altitudes are decreasing from the leaves to the root (i.e. for any node \f$n\f$,
     *   \f$alt(n) \geq alt(parent(n))\f$), as in a max-tree.
     *
     * Complexity: The computation takes \f$\mathcal{O}(N)\f$ time, where \f$N\f$ is the number of nodes in the tree.
     *
     * @tparam tree_t tree type
     * @tparam T element type of altitudes
     * @param tree input tree
     * @param altitudes node altitudes of the input tree, monotonic from the leaves to the root (1d array with one
     *     element per node of the tree)
     * @param increasing_altitudes true if the altitudes are increasing from the leaves to the root, false if they are
     *     decreasing (see above)
     * @return a 1d array representing the height of each node of the tree
     */
    template<typename tree_t, typename T>
    auto
    attribute_height(const tree_t &tree,
                     array_view<const T> altitudes,
                     bool increasing_altitudes) {
        hg_assert_node_weights(tree, altitudes);
        hg_assert_1d_array(altitudes);
        using value_type = T;
        // type of the difference of two altitudes (promoted for types smaller than int)
        using result_type = decltype(std::declval<value_type>() - std::declval<value_type>());

        tree.compute_children();
        const auto &parents = tree.parents();
        const index_t num_l = (index_t) num_leaves(tree);
        const index_t num_v = (index_t) num_vertices(tree);
        if (increasing_altitudes) {
            auto min_depth = hg::empty_like(altitudes);
            for (index_t i = 0; i < num_l; i++) {
                min_depth(i) = altitudes(parents(i));
            }
            for (auto n: leaves_to_root_iterator(tree, leaves_it::exclude)) {
                min_depth(n) = (std::numeric_limits<value_type>::max)();
                bool flag = true;
                for (auto c: children_iterator(n, tree)) {
                    if (!is_leaf(c, tree)) {
                        flag = false;
                        if (min_depth(c) < min_depth(n)) {
                            min_depth(n) = min_depth(c);
                        }
                    }
                }
                if (flag) {
                    min_depth(n) = altitudes(n);
                }
            }
            auto height = hg::empty<result_type>(altitudes.shape());
            for (index_t i = 0; i < num_v; i++) {
                height(i) = altitudes(parents(i)) - min_depth(i);
            }
            return height;
        } else {
            auto max_depth = hg::empty_like(altitudes);
            for (index_t i = 0; i < num_l; i++) {
                max_depth(i) = altitudes(parents(i));
            }
            for (auto n: leaves_to_root_iterator(tree, leaves_it::exclude)) {
                max_depth(n) = std::numeric_limits<value_type>::lowest();
                bool flag = true;
                for (auto c: children_iterator(n, tree)) {
                    if (!is_leaf(c, tree)) {
                        flag = false;
                        if (max_depth(c) > max_depth(n)) {
                            max_depth(n) = max_depth(c);
                        }
                    }
                }
                if (flag) {
                    max_depth(n) = altitudes(n);
                }
            }
            auto height = hg::empty<result_type>(altitudes.shape());
            for (index_t i = 0; i < num_v; i++) {
                height(i) = max_depth(i) - altitudes(parents(i));
            }
            return height;
        }
    };

    /**
     * Identify the nodes of a hierarchy that represent extrema (minima or maxima).
     *
     * An extremum (minimum or maximum) of the hierarchy \f$T\f$ with altitudes \f$alt\f$ is a non-leaf node \f$n\f$ of
     * \f$T\f$ such that the altitude of any non-leaf node of the subtree rooted in \f$n\f$ is equal to the altitude of
     * \f$n\f$, and such that \f$n\f$ is the root or the altitude of the parent of \f$n\f$ is different from the
     * altitude of \f$n\f$. Formally, a non-leaf node \f$n\f$ is an extremum if:
     *
     * \f[
     *     \forall m \in subtree(n) \setminus leaves(n),\ alt(m) = alt(n)
     *     \qquad\textrm{and}\qquad
     *     \big(n = root \ \textrm{ or } \ alt(parent(n)) \neq alt(n)\big)
     * \f]
     *
     * where \f$subtree(n)\f$ is the set of the nodes of the subtree rooted in \f$n\f$ (including \f$n\f$ and its
     * leaves). In particular, the root is an extremum if and only if all the non-leaf nodes of the tree have the same
     * altitude.
     *
     * The result is a boolean array such that \f$result(n)\f$ is true if the node \f$n\f$ is an extremum and false
     * otherwise. Leaves are never extrema, and the altitudes of the leaves are not taken into account. The function
     * does not depend on whether the altitudes are increasing or decreasing: the extrema are minima for increasing
     * altitudes and maxima for decreasing altitudes.
     *
     * Complexity: The computation takes \f$\mathcal{O}(N)\f$ time, where \f$N\f$ is the number of nodes in the tree.
     *
     * @tparam tree_t tree type
     * @tparam T element type of altitudes
     * @param tree input tree
     * @param altitudes node altitudes of the input tree (1d array with one element per node of the tree)
     * @return a 1d boolean array representing, for each node of the tree, whether it is an extremum
     */
    template<typename tree_t, typename T>
    auto
    attribute_extrema(const tree_t &tree,
                      array_view<const T> altitudes) {
        hg_assert_node_weights(tree, altitudes);
        hg_assert_1d_array(altitudes);

        array_1d<bool> extrema = hg::ones<bool>({num_vertices(tree)});

        const auto num_leaves = (hg::index_t) hg::num_leaves(tree);
        extrema.slice(0, num_leaves).fill(0);

        for (auto n: leaves_to_root_iterator(tree, leaves_it::exclude, root_it::exclude)) {
            bool p_non_canonical = altitudes(n) == altitudes(parent(n, tree));
            extrema(parent(n, tree)) = extrema(n) && p_non_canonical && extrema(parent(n, tree));
            extrema(n) = extrema(n) && !p_non_canonical;
        }
        return extrema;
    }

    /**
     * Compute the extinction value of each node of the given tree for the given attribute.
     *
     * Consider a tree \f$T\f$ with increasing altitudes \f$alt\f$ and an increasing attribute \f$att\f$
     * (\f$att(n) \leq att(parent(n))\f$ for any node \f$n\f$, for example the area). Intuitively, the extinction value
     * of a minimum of \f$T\f$ (see attribute_extrema) measures how long it survives when the nodes whose attribute is
     * smaller than a threshold \f$k\f$ are removed for increasing values of \f$k\f$: a minimum disappears when it is
     * merged with a deeper minimum.
     *
     * Formally, let \f$\{M_i\}\f$ be the set of minima of the hierarchy \f$T\f$ with altitudes \f$alt\f$. Let
     * \f$\prec\f$ be a total ordering of \f$\{M_i\}\f$ such that \f$M_i \prec M_j \Rightarrow alt(M_i) \leq alt(M_j)\f$
     * (when several minima have the same altitude, they are ordered according to the order of the children in the
     * tree). Let \f$r(M_i)\f$ be the smallest node of \f$T\f$ containing \f$M_i\f$ and another minimum \f$M_j\f$ such
     * that \f$M_j \prec M_i\f$, and let \f$c(M_i)\f$ be the child of \f$r(M_i)\f$ containing \f$M_i\f$. The extinction
     * value of \f$M_i\f$ is then defined as:
     *
     * \f[
     *     ext(M_i) = \begin{cases}
     *         att(root) & \textrm{if } M_i \textrm{ is the first minimum for } \prec \textrm{ (the deepest one)}\\
     *         att(c(M_i)) & \textrm{otherwise}
     *     \end{cases}
     * \f]
     *
     * Extinction values of minima are then extended to the other nodes of the tree with the following rules:
     *
     * - the extinction value of a non-leaf node \f$n\f$ which is not included in a minimum is the largest extinction
     *   value among the minima included in \f$n\f$ (such a node always includes at least one minimum); and
     * - the extinction value of a leaf \f$n\f$ included in a minimum \f$M_i\f$ is equal to the extinction value of
     *   \f$M_i\f$. If \f$n\f$ is not included in any minimum, its extinction value is 0.
     *
     * The extinction values of the non-leaf nodes strictly included in a minimum are not covered by these rules. Such
     * nodes only exist if a non-leaf node has the same altitude as its parent: they can be removed beforehand with
     * simplify_tree, using the criterion \f$alt(n) = alt(parent(n))\f$.
     *
     * The function can also handle decreasing altitudes, in which case *minima* should be replaced by *maxima* in the
     * description above. Possible values of increasing_altitudes are:
     *
     * - true: the altitudes are increasing from the leaves to the root (i.e. for any node \f$n\f$,
     *   \f$alt(n) \leq alt(parent(n))\f$);
     * - false: the altitudes are decreasing from the leaves to the root (i.e. for any node \f$n\f$,
     *   \f$alt(n) \geq alt(parent(n))\f$).
     *
     * Complexity: The computation takes \f$\mathcal{O}(N)\f$ time, where \f$N\f$ is the number of nodes in the tree.
     *
     * @tparam tree_t tree type
     * @tparam T1 element type of altitudes
     * @tparam T2 element type of attribute
     * @param tree input tree
     * @param altitudes node altitudes of the input tree, monotonic from the leaves to the root (1d array with one
     *     element per node of the tree)
     * @param attribute node attribute of the input tree, increasing from the leaves to the root (1d array with one
     *     element per node of the tree)
     * @param increasing_altitudes true if the altitudes are increasing from the leaves to the root, false if they are
     *     decreasing (see above)
     * @return a 1d array representing the extinction value of each node of the tree
     */
    template<typename tree_t,
            typename T1,
            typename T2>
    auto attribute_extinction_value(const tree_t &tree,
                                    array_view<const T1> altitudes,
                                    array_view<const T2> attribute,
                                    bool increasing_altitudes) {
        using value_type = T1;
        hg_assert_node_weights(tree, altitudes);
        hg_assert_1d_array(altitudes);
        hg_assert_node_weights(tree, attribute);
        hg_assert_1d_array(attribute);

        tree.compute_children();

        // identify path to the deepest extrema
        array_1d<index_t> ref_son({num_vertices(tree)}, invalid_index);
        if (increasing_altitudes) {
            auto min_depth = hg::empty_like(altitudes);
            for (auto n: leaves_to_root_iterator(tree, leaves_it::exclude)) {
                min_depth(n) = (std::numeric_limits<value_type>::max)();
                bool flag = true;
                for (auto c: children_iterator(n, tree)) {
                    if (!is_leaf(c, tree)) {
                        flag = false;
                        if (min_depth(c) < min_depth(n)) {
                            min_depth(n) = min_depth(c);
                            ref_son(n) = c;
                        }
                    }
                }
                if (flag) {
                    min_depth(n) = altitudes(n);
                }
            }
        } else {
            auto max_depth = hg::empty_like(altitudes);
            for (auto n: leaves_to_root_iterator(tree, leaves_it::exclude)) {
                max_depth(n) = std::numeric_limits<value_type>::lowest();
                bool flag = true;
                for (auto c: children_iterator(n, tree)) {
                    if (!is_leaf(c, tree)) {
                        flag = false;
                        if (max_depth(c) > max_depth(n)) {
                            max_depth(n) = max_depth(c);
                            ref_son(n) = c;
                        }
                    }
                }
                if (flag) {
                    max_depth(n) = altitudes(n);
                }
            }
        }

        // extinction of non leaf nodes
        array_1d<T2> extinction = array_1d<T2>::from_shape(
                {num_vertices(tree)});
        extinction(root(tree)) = attribute(root(tree));
        for (auto n: root_to_leaves_iterator(tree, leaves_it::exclude, root_it::exclude)) {
            auto pn = parent(n, tree);
            if (n == ref_son(pn)) {
                extinction(n) = extinction(pn);
            } else {
                extinction(n) = attribute(n);
            }
        }

        auto extrema = attribute_extrema(tree, altitudes);
        auto indices = hg::arange<index_t>(num_vertices(tree));
        if (!extrema(root(tree))) {
            indices(root(tree)) = -1;
        }
        array_1d<bool> not_extrema = array_1d<bool>::from_shape({num_vertices(tree)});
        for (index_t i = 0; i < (index_t) num_vertices(tree); i++) {
            not_extrema(i) = !extrema(i);
        }
        auto extrema_leaves = propagate_sequential(tree, indices, not_extrema);
        for (auto n: leaves_iterator(tree)) {
            if (extrema_leaves(n) != -1) {
                extinction(n) = extinction(extrema_leaves(n));
            } else {
                extinction(n) = 0;
            }
        }

        return extinction;
    };

    /**
     * Compute the dynamics of each node of the given tree.
     *
     * The dynamics is the extinction value (see attribute_extinction_value) for the attribute height (see
     * attribute_height, with the parent of each node as reference).
     *
     * Consider a tree \f$T\f$ with increasing altitudes \f$alt\f$. The dynamics of a minimum \f$M\f$ of \f$T\f$ (see
     * attribute_extrema) is the difference between the altitude of the smallest node \f$r(M)\f$ containing \f$M\f$ and
     * a deeper minimum, and the altitude of \f$M\f$. If no such node exists (\f$M\f$ is the deepest minimum), the
     * dynamics of \f$M\f$ is the difference between the altitude of the root and the altitude of \f$M\f$:
     *
     * \f[
     *     dyn(M) = \begin{cases}
     *         alt(root) - alt(M) & \textrm{if } M \textrm{ is the deepest minimum}\\
     *         alt(r(M)) - alt(M) & \textrm{otherwise}
     *     \end{cases}
     * \f]
     *
     * The order between the minima, and in particular between minima of the same altitude, is the one defined in
     * attribute_extinction_value. The dynamics of the other nodes are obtained with the rules of
     * attribute_extinction_value: a non-leaf node which is not included in a minimum takes the largest dynamics among
     * the minima it contains, a leaf takes the dynamics of the minimum containing it, and 0 if it is not included in
     * any minimum.
     *
     * The function can also handle decreasing altitudes, in which case *minima* should be replaced by *maxima*, and the
     * differences above by \f$alt(M) - alt(root)\f$ and \f$alt(M) - alt(r(M))\f$. Possible values of
     * increasing_altitudes are:
     *
     * - true: the altitudes are increasing from the leaves to the root (i.e. for any node \f$n\f$,
     *   \f$alt(n) \leq alt(parent(n))\f$);
     * - false: the altitudes are decreasing from the leaves to the root (i.e. for any node \f$n\f$,
     *   \f$alt(n) \geq alt(parent(n))\f$).
     *
     * Complexity: The computation takes \f$\mathcal{O}(N)\f$ time, where \f$N\f$ is the number of nodes in the tree
     * (computation of attribute_height followed by the computation of attribute_extinction_value).
     *
     * @tparam tree_t tree type
     * @tparam T element type of altitudes
     * @param tree input tree
     * @param altitudes node altitudes of the input tree, monotonic from the leaves to the root (1d array with one
     *     element per node of the tree)
     * @param increasing_altitudes true if the altitudes are increasing from the leaves to the root, false if they are
     *     decreasing (see above)
     * @return a 1d array representing the dynamics of each node of the tree
     */
    template<typename tree_t, typename T>
    auto attribute_dynamics(const tree_t &tree,
                            array_view<const T> altitudes,
                            bool increasing_altitudes) {
        using value_type = T;
        hg_assert_node_weights(tree, altitudes);
        hg_assert_1d_array(altitudes);

        auto height = attribute_height(tree, altitudes, increasing_altitudes);

        return attribute_extinction_value(tree, altitudes, height, increasing_altitudes);
    };

    /**
     * Compute the sibling of each node of the given tree.
     *
     * The children of a node \f$p\f$ are ordered. Let \f$n\f$ be the \f$k\f$-th child of \f$p\f$ (starting from 0)
     * among the \f$numc(p)\f$ children of \f$p\f$. The sibling of \f$n\f$ is the node:
     *
     * \f[
     *     sibling(n) = child\big((k + skip) \bmod numc(p),\ p\big)
     * \f]
     *
     * where \f$child(i, p)\f$ is the \f$i\f$-th child of \f$p\f$ (starting from 0). The children of a node are thus
     * visited in a circular way: with skip equal to 1, the sibling of a node is the next child of its parent, and the
     * sibling of the last child is the first one; with skip equal to -1, the sibling of a node is the previous child of
     * its parent. The root is not the child of any node: its sibling is itself.
     *
     * The sibling attribute makes it possible to emulate a circular linked list among the children of a node. In a
     * binary tree, with skip equal to 1, the sibling of a node is its only brother.
     *
     * Complexity: The computation takes \f$\mathcal{O}(N)\f$ time, where \f$N\f$ is the number of nodes in the tree.
     *
     * @tparam tree_t tree type
     * @param tree input tree
     * @param skip offset, in the circular list of the children of the parent, between a node and its sibling (integer,
     *     possibly negative): 1 (default) for the next child, -1 for the previous one
     * @return a 1d array of node indices representing the sibling of each node of the tree
     */
    template<typename tree_t>
    auto attribute_sibling(const tree_t &tree, index_t skip = 1) {

        tree.compute_children();
        array_1d<index_t> attribute = hg::empty<index_t>({num_vertices(tree)});
        for (auto n: leaves_to_root_iterator(tree, leaves_it::exclude, root_it::include)) {
            index_t nchs = num_children(n, tree);
            for (index_t i = 0; i < nchs; i++) {

                index_t j = (i + skip) % nchs;
                if (j < 0) { // stupid c modulo
                    j += nchs;
                }

                attribute(child(i, n, tree)) = child(j, n, tree);
            }
        }
        attribute(root(tree)) = root(tree);
        return attribute;
    }

    /**
     * Computes the length of the contour (perimeter) of each node of the given component tree.
     *
     * Each leaf \f$v\f$ of the tree is a vertex of base_graph with a perimeter \f$p(v)\f$, and each edge \f$e\f$ of
     * base_graph has a length \f$l(e)\f$: the length of the border shared by the two vertices it connects. The contour
     * length \f$C(n)\f$ of a node \f$n\f$ is the sum of the perimeters of its leaves, minus twice the length of the
     * borders shared by two of its leaves:
     *
     * \f[
     *     C(n) = \sum_{v \in leaves(n)} p(v) \; - \; 2 \sum_{e = \{i, j\} \in E,\ i, j \in leaves(n)} l(e)
     * \f]
     *
     * where \f$E\f$ is the edge set of base_graph. The contour of a leaf is its own perimeter.
     *
     * This function requires a component tree (for example a min-tree or a max-tree), in which two adjacent leaves
     * always belong to nested nodes. It does not give correct results for a tree of shapes whose interpolated pixels
     * have been removed: two children of a node may then contain adjacent leaves.
     *
     * Complexity: \f$\mathcal{O}(N + M)\f$, where \f$N\f$ is the number of nodes of the tree and \f$M\f$ is the number
     * of edges of base_graph.
     *
     * @tparam tree_t tree type
     * @tparam graph_t graph type
     * @tparam T1 element type of vertex_perimeter
     * @tparam T2 element type of edge_length
     * @param tree input component tree
     * @param base_graph graph on the leaves of the input tree
     * @param vertex_perimeter perimeter of each vertex of base_graph
     * @param edge_length length of each edge of base_graph (length of the border shared by the two vertices it
     *     connects)
     * @return an array of double with the contour length of each node of the tree
     */
    template<typename tree_t, typename graph_t, typename T1, typename T2>
    auto attribute_contour_length_component_tree(
            const tree_t &tree,
            const graph_t &base_graph,
            array_view<const T1> vertex_perimeter,
            array_view<const T2> edge_length) {
        hg_assert_component_tree(tree);
        hg_assert_1d_array(vertex_perimeter);
        hg_assert_leaf_weights(tree, vertex_perimeter);
        hg_assert_1d_array(edge_length);
        hg_assert_edge_weights(base_graph, edge_length);

        array_1d<double> res = array_1d<double>::from_shape({num_vertices(tree)});
        hg::assign(res.slice(0, num_leaves(tree)), vertex_perimeter);
        array_1d<bool> visited({num_leaves(tree)}, false);
        tree.compute_children();

        for (auto i: leaves_to_root_iterator(tree, leaves_it::exclude)) {
            res(i) = 0;
            for (auto c: children_iterator(i, tree)) {
                res(i) += res(c);
                if (is_leaf(c, tree)) {
                    for (auto e: out_edge_iterator(c, base_graph)) {
                        if (visited(target(e, base_graph))) {
                            res(i) -= 2.0 * edge_length(e);
                        }
                    }
                    visited(c) = true;
                }
            }
        }
        return res;
    }


    /**
     * Compute the rank of each node in the ordered list of children of its parent.
     *
     * Given a non-root node \f$n\f$ whose parent is \f$p\f$, the attribute value of \f$n\f$ is the rank of \f$n\f$ in
     * the list of children of \f$p\f$ (starting from 0):
     *
     * \f[
     *     cn(n) = i \iff n = child(i, p)
     * \f]
     *
     * where \f$child(i, p)\f$ is the \f$i\f$-th child of \f$p\f$ (starting from 0). The root of the tree is not the
     * child of any node: it takes the value -1.
     *
     * Complexity: The computation takes \f$\mathcal{O}(N)\f$ time, where \f$N\f$ is the number of nodes in the tree.
     *
     * @tparam tree_t tree type
     * @param tree input tree
     * @return a 1d array of integers representing the rank of each node of the tree among the children of its parent
     */
    template<typename tree_t>
    auto attribute_child_number(const tree_t &tree) {
        tree.compute_children();
        array_1d<index_t> res = hg::empty<index_t>({num_vertices(tree)});
        for (auto i: leaves_to_root_iterator(tree, leaves_it::exclude)) {
            for (index_t c = 0; c < (index_t) num_children(i, tree); c++) {
                auto cc = child(c, i, tree);
                res(cc) = c;
            }
        }
        res(root(tree)) = invalid_index;
        return res;
    }


    /**
     * Computes, for each node of a first tree, the smallest node of a second tree containing it.
     *
     * The two trees \f$t_1\f$ and \f$t_2\f$ must be defined on the same domain, i.e. they share the same set of leaves.
     * For each node \f$n\f$ of \f$t_1\f$, the result is the smallest node of \f$t_2\f$ containing all the leaves of the
     * subtree of \f$t_1\f$ rooted in \f$n\f$, that is the lowest common ancestor in \f$t_2\f$ of these leaves:
     *
     * \f[
     *     ses(n) = lca_{t_2}\big(leaves_{t_1}(n)\big)
     * \f]
     *
     * In particular, the result for a leaf is the leaf itself.
     *
     * Complexity: \f$\mathcal{O}(N_1)\f$ lowest common ancestor computations in \f$t_2\f$, where \f$N_1\f$ is the
     * number of nodes of \f$t_1\f$.
     *
     * @tparam tree_t tree type
     * @param t1 first tree
     * @param t2 second tree, with the same leaves as t1
     * @return an array of node indices of t2 with one element per node of t1
     */
    template<typename tree_t>
    auto attribute_smallest_enclosing_shape(const tree_t &t1, const tree_t &t2) {
        array_1d<index_t> attr({num_vertices(t1)}, invalid_index);
        hg::assign(attr.slice(0, num_leaves(t1)), hg::arange<index_t>(num_leaves(t1)));

        for (auto i: leaves_to_root_iterator(t1)) {
            auto p = parent(i, t1);
            if (attr(p) == -1) {
                attr(p) = attr(i);
            } else {
                attr(p) = lowest_common_ancestor(attr(p), attr(i), t2);
            }
        }

        return attr;
    }

    namespace tree_attribute_internal {
        /**
         * Unsigned integer type, at least as wide as unsigned int, in which the arithmetic of the integer type V
         * wraps around instead of overflowing (V itself for non integral types).
         */
        template<typename V, bool = std::is_integral<V>::value>
        struct wrapping_type {
            using type = V;
        };

        template<typename V>
        struct wrapping_type<V, true> {
            using type = std::common_type_t<unsigned int, std::make_unsigned_t<V>>;
        };

        template<typename V>
        using wrapping_type_t = typename wrapping_type<V>::type;
    }

    /**
     * Compute, for each node, the sum of the products of the weights of every pair of its children.
     *
     * Given a tree \f$T\f$ with node weights \f$w\f$, the children pair sum product for a node \f$n\f$ sums, for every
     * unordered pair \f$\{c_i, c_j\}\f$ of distinct children of \f$n\f$, the product of the node weights of \f$c_i\f$
     * and \f$c_j\f$. Formally:
     *
     * \f[
     *     res(n) = \sum_{i=0}^{numc(n) - 1} \sum_{j=0}^{i - 1} w(child(i, n)) \times w(child(j, n))
     * \f]
     *
     * where \f$numc(n)\f$ is the number of children of \f$n\f$ and \f$child(i, n)\f$ is the \f$i\f$-th child of the
     * node \f$n\f$ (starting from 0). The value of a leaf, which has no child, is 0, as well as the value of a node
     * with a single child.
     *
     * If the node weights are vectors, the computation is done independently for each component. The result is thus an
     * array with the same shape as node_weights.
     *
     * Complexity: The computation takes \f$\mathcal{O}(\sum_{n} numc(n)^2 \times d)\f$ time, the sum being taken over
     * all the nodes \f$n\f$ of the tree, where \f$d\f$ is the dimension of the node weights (\f$d = 1\f$ for scalar
     * weights). This is \f$\mathcal{O}(N \times C \times d)\f$, where \f$N\f$ is the number of nodes in the tree and
     * \f$C\f$ is the maximal number of children of a node.
     *
     * With floating point node weights, each product is computed in \f$T\f$, then converted to value_type. With
     * integer node weights, each product is computed in value_type; if value_type is also an integer type, products
     * and sums that do not fit in value_type wrap around modulo \f$2^n\f$, where \f$n\f$ is the number of bits of
     * value_type.
     *
     * @tparam tree_t tree type
     * @tparam T element type of node_weights
     * @tparam value_type value type of the result
     * @param tree input tree
     * @param node_weights node weights of the input tree, scalar or vectorial (array whose first dimension is the
     *     number of nodes of the tree)
     * @return an array with the same shape and dtype as node_weights representing the children pair sum product of each
     *     node of the tree
     */
    template<typename tree_t, typename T, typename value_type=T>
    auto attribute_children_pair_sum_product(const tree_t &tree, array_view<const T> node_weights) {
        hg_assert_node_weights(tree, node_weights);

        tree.compute_children();
        array_nd<value_type> res = hg::zeros<value_type>(node_weights.shape());
        const index_t num_v = (index_t) num_vertices(tree);
        const index_t num_l = (index_t) num_leaves(tree);

        // row-major elements: node i owns the num_d values starting at i * num_d
        const T *w = node_weights.data();
        value_type *r = res.data();
        const index_t num_d = (num_v == 0) ? 0 : (index_t) res.size() / num_v;

        using namespace tree_attribute_internal;
        // Floating point weights: product in T. Integer weights: product in value_type, where integer products and
        // sums wrap around modulo 2^n (n bits of value_type) instead of overflowing.
        using product_type = std::conditional_t<!std::is_integral<T>::value, T,
                std::conditional_t<std::is_integral<value_type>::value, wrapping_type_t<value_type>, value_type>>;
        using sum_type = std::conditional_t<std::is_integral<value_type>::value,
                wrapping_type_t<value_type>, value_type>;

        for (index_t i = num_l; i < num_v; i++) {
            const index_t num_c = (index_t) num_children(i, tree);
            value_type *ri = r + i * num_d;
            for (index_t c1 = 0; c1 < num_c; c1++) {
                const T *w1 = w + child(c1, i, tree) * num_d;
                for (index_t c2 = c1 + 1; c2 < num_c; c2++) {
                    const T *w2 = w + child(c2, i, tree) * num_d;
                    for (index_t k = 0; k < num_d; k++) {
                        const auto p = static_cast<value_type>(
                                static_cast<product_type>(w1[k]) * static_cast<product_type>(w2[k]));
                        ri[k] = static_cast<value_type>(static_cast<sum_type>(ri[k]) + static_cast<sum_type>(p));
                    }
                }
            }
        }
        return res;
    }

}
