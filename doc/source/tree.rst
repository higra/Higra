.. _tree:

Trees
=====

.. important::

    ``#include "higra/graph.hpp``

The `tree` class is the fundamental structure of many hierarchical representations of graphs.
In Higra, a tree is an undirected acyclic rooted graph (see :ref:`graph`), augmented with specific functions
matching the usual semantic of trees.

As with any graph in Higra, the vertices of a tree (also called *nodes*) are represented by positive integers suitable
for array indexing.
Higra's tree ensure that vertices are are topologically sorted, i.e. that for any vertices :math:`v1` and :math:`v2`, if
:math:`v2` is an ancestor of :math:`v1`, then :math:`v1\le v2`. Moreover, whenever a tree :math:`t` is a hierarchical
representation of a graph :math:`(V, E)`, then the leaves of :math:`t` are equal to :math:`V`: i.e. there is a direct
mapping between the leaves of the tree and the vertices of the graph represented by this tree.

The base of the `tree` data structure is the `parent array`: i.e. an array that indicates for each vertex the index of
its parent (for convenience, the root of the tree is its own parent).
For example, the following tree (leaves are represented by squares, inner nodes by circles, and vertex indices are
indicated inside the nodes):

.. image:: fig/tree.svg
    :align: center

is represented by the following parent array:

.. csv-table::

       node , 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11
       parent , 7 , 7 , 8 , 8 , 8 , 9 , 9 , 11 , 10 , 10 , 10 , 11

Constructor
-----------

The ``tree`` class has a single constructor that takes a single parameter: the parent array.

Example:

.. tabs::

    .. tab:: c++

        .. code-block:: cpp
            :linenos:

            #include "higra/graph.hpp"
            using namespace hg;

            // creates the tree shown in the figure above
            tree t{7, 7, 8, 8, 8, 9, 9, 11, 10, 10, 10, 11};



    .. tab:: python

        .. code-block:: python
            :linenos:

            import higra as hg

            # creates the tree shown in the figure above
            g = hg.Tree((7, 7, 8, 8, 8, 9, 9, 11, 10, 10, 10, 11))



Basic functions
---------------

.. list-table::
    :header-rows: 1

    *   - Function
        - Returns
        - Description
    *   - ``num_leaves``
        - positive integer
        - number of leaves in the tree
    *   - ``root``
        - vertex
        - Root node (last node of the parent array)
    *   - ``parent``
        - vertex
        - Parent of the given node
    *   - ``parents``
        - array of vertices
        - the parent array
    *   - ``num_children``
        - positive integer
        - number of children of the given node


Example:

.. tabs::

    .. tab:: c++

        .. code-block:: cpp
            :linenos:

            // creates the tree shown in the figure above
            tree t{7, 7, 8, 8, 8, 9, 9, 11, 10, 10, 10, 11};

            num_leaves(t);      //  7
            root(t);            // 11
            parent(2, t);       //  8
            parents(t);         //  array {7, 7, 8, 8, 8, 9, 9, 11, 10, 10, 10, 11}
            num_children(8, t); //  3


    .. tab:: python

        .. code-block:: python
            :linenos:

            # creates the tree shown in the figure above
            g = hg.Tree((7, 7, 8, 8, 8, 9, 9, 11, 10, 10, 10, 11))

            t.num_leaves();     #  7
            t.root();           # 11
            t.parent(2);        #  8
            t.parents();        #  array {7, 7, 8, 8, 8, 9, 9, 11, 10, 10, 10, 11}
            t.num_children(8);  #  3


Iterators
---------

.. list-table::
    :header-rows: 1

    *   - Function
        - Returns
        - Description
    *   - ``children_iterator``
        - a range of iterators
        - iterator on the children of the given node
    *   - ``leaves_iterator``
        - a range of iterators
        - iterator on the leaves of the tree
    *   - ``leaves_to_root_iterator``
        - a range of iterators
        - iterator on the nodes of the tree in a topological order
    *   - ``root_to_leaves_iterator``
        - a range of iterators
        - iterator on the nodes of the tree in a reverse topological order



.. tabs::

    .. tab:: c++

        .. code-block:: cpp
            :linenos:

            // creates the tree shown in the figure above
            tree t{7, 7, 8, 8, 8, 9, 9, 11, 10, 10, 10, 11};

            for(auto n: children_iterator(t, 8)){
                ... // 2, 3, 4
            }

            for(auto n: leaves_to_root_iterator(t,
                leaves_it::include /* optional: include (default) or exclude leaves from the iterator*/,
                root_it::include /* optional: include (default) or exclude root from the iterator*/)){
                ... // 0, 1, 2, ..., 11
            }

            for(auto n: leaves_to_root_iterator(t,
                    leaves_it::exclude,
                    root_it::exclude)){
                    ... // 7, 8, 9, 10
            }

            for(auto n: root_to_leaves_iterator(t,
                leaves_it::include /* optional: include (default) or exclude leaves from the iterator*/,
                root_it::include /* optional: include (default) or exclude root from the iterator*/)){
                ... // 11, 10, 9, ..., 0
            }

            for(auto n: root_to_leaves_iterator(t,
                    leaves_it::exclude,
                    root_it::exclude)){
                    ... // 10, 9, 8, 7
            }


    .. tab:: python

        .. code-block:: python
            :linenos:

            # creates the tree shown in the figure above
            g = hg.Tree((7, 7, 8, 8, 8, 9, 9, 11, 10, 10, 10, 11))

            for n in t.children_iterator(8):
                ... # 2, 3, 4

            for n in t.leaves_to_root_iterator(
                include_leaves = True, # optional: include (default) or exclude leaves from the iterator
                include_leaves = True): # optional: include (default) or exclude root from the iterator
                ... // 0, 1, 2, ..., 11

            for n in t.leaves_to_root_iterator(
                include_leaves = False,
                include_leaves = False):
                ... // 7, 8, 9, 10

            for n in t.root_to_leaves_iterator(
                include_leaves = True, # optional: include (default) or exclude leaves from the iterator
                include_leaves = True): # optional: include (default) or exclude root from the iterator
                ... // 11, 10, 9, ..., 0

            for n in t.root_to_leaves_iterator(
                include_leaves = False,
                include_leaves = False):
                ... // 10, 9, 8, 7


