.. _graph:

Graphs
======

.. important::

    ``#include "higra/graph.hpp``

Graphs come in three flavours in Higra:

1. ``ugraph`` represents general undirected graphs (adjacency list).
2. ``tree`` represents undirected connected acyclic rooted graphs (parent array).
3. ``regular_graph`` represents implicit graphs: in such graphs edges are computed on the fly (not stored). For example, they can be used to represent pixel adjacencies in images.

This page presents common functions for the manipulation of graphs.
A dedicated page for the ``tree`` structure, see :ref:`tree`.

All functions acting on graphs have the same name in C++ and in Python.
In C++ graph methods are free functions (as in the `Boost Graph Library BGL <https://www.boost.org/doc/libs/1_67_0/libs/graph/doc/index.html>`_),
while in Python they are member functions.
For example, the function ``num_vertices`` that returns the number of vertices in a graph, will be called:

.. tabs::

    .. tab:: c++

        .. code-block:: cpp
            :linenos:

            auto n = num_vertices(graph);

    .. tab:: python

        .. code-block:: python
            :linenos:

            n = graph.num_vertices()


Vertices
--------

Graph vertices are represented by positive integers (``index_t`` in c++), suitable for array indexing. All operations are done in constant time.


.. list-table::
    :header-rows: 1

    *   - Function
        - Returns
        - Description
        - Available
    *   - ``add_vertex``
        - new vertex
        - add a new vertex to the graph
        - ``ugraph``
    *   - ``num_vertices``
        - positive integer
        - Number of vertices in the graph
        - ``regular_graph``, ``ugraph``, ``tree``

Example:

.. tabs::

    .. tab:: c++

        .. code-block:: cpp
            :linenos:

                #include "higra/graph.hpp"
                using namespace hg;

                ugraph g;
                add_vertex(g);
                add_vertex(g);

                auto nv = num_vertices(g); // 2


    .. tab:: python

        .. code-block:: python
            :linenos:

                import higra as hg

                g = hg.UndirectedGraph()
                g.add_vertex()
                g.add_vertex()

                nv = g.num_vertices() # 2


Iterating on vertices
*********************

.. list-table::
    :header-rows: 1

    *   - Function
        - Returns
        - Description
        - Available
    *   - ``vertex_iterator``
        - a range of iterators
        - iterator on all graph vertices
        - ``regular_graph``, ``ugraph``, ``tree``
    *   - ``adjacent_vertex_iterator``
        - a range of iterators
        - iterator on all vertices adjacent to the given vertex
        - ``regular_graph``, ``ugraph``, ``tree``


.. tabs::

    .. tab:: c++

        .. code-block:: cpp
            :linenos:

                ugraph g;
                ...

                for(auto v: vertex_iterator(g)){
                    ... // all vertices of g
                }

                for(auto v: adjacent_vertex_iterator(1, g)){
                    ... // all vertices adjacent to vertex 1 in g
                }


        .. tab:: python

        .. code-block:: python
            :linenos:

                g = hg.UndirectedGraph()
                ...

                for v in g.vertex_iterator():
                    ... # all vertices of g

                for v in g.adjacent_vertex_iterator(1):
                    ... # all vertices adjacent to vertex 1 in g

Edges
-----

Graph edges are represented by pairs of vertices, i.e. pairs of positive integers (``index_t`` in c++), whose first element is the source and second element is the target.
All operations are done in constant time.

.. list-table::
    :header-rows: 1

    *   - Function
        - Returns
        - Description
        - Available
    *   - ``add_edge``
        - void
        - add a new edge to the graph
        - ``ugraph``
    *   - ``num_edges``
        - positive integer
        - number of edges in the graph
        - ``regular_graph``, ``ugraph``, ``tree``
    *   - ``source``
        - vertex index
        - source vertex of an edge
        - ``regular_graph``, ``ugraph``, ``tree``
    *   - ``target``
        - vertex index
        - target vertex of an edge
        - ``regular_graph``, ``ugraph``, ``tree``

Example:

.. tabs::

    .. tab:: c++

        .. code-block:: cpp
            :linenos:

            #include "higra/graph.hpp"
            using namespace hg;

            // create a graph with 3 vertices and no edge
            ugraph g(2);

            // add an edge, between vertex 0 and 1
            add_edge(0, 1, g);
            // add an edge, between vertex 0 and 1
            add_edge(1, 2, g);

            auto ne = num_edges(g); // 2


    .. tab:: python

        .. code-block:: python
            :linenos:

            import higra as hg

            # create a graph with 3 vertices and no edge
            g = hg.UndirectedGraph(3)

            # add an edge, between vertex 0 and 1
            g.add_edge(0, 1);
            # add an edge, between vertex 0 and 1
            g.add_edge(1, 2);

            ne = g.num_edges() # 2


Iterating on edges
******************

.. list-table::
    :header-rows: 1

    *   - Function
        - Returns
        - Description
        - Available
    *   - ``edge_iterator``
        - a range of iterators
        - iterator on graph edges
        - ``regular_graph``, ``ugraph``, ``tree``
    *   - ``in_edge_iterator``
        - a range of iterators
        - iterators on all edges whose target is the given vertex
        - ``regular_graph``, ``ugraph``, ``tree``
    *   - ``out_edge_iterator``
        - a range of iterators
        - iterators on all edges whose source is the given vertex
        - ``regular_graph``, ``ugraph``, ``tree``



.. tabs::

    .. tab:: c++

        .. code-block:: cpp
            :linenos:

            ugraph g;
            ...

            for(auto e: edge_iterator(g)){
                std::cout << source(e, g) << " " << target(e, g) << std::endl;
            }

            for(auto e: in_edge_iterator(1, g)){
                ... // all edges e such that target(e, g) == 1
            }

            for(auto e: out_edge_iterator(1, g)){
                ... // all edges e such that source(e, g) == 1
            }


    .. tab:: python

        .. code-block:: python
            :linenos:

            g = hg.UndirectedGraph()
            ...

            for e in g.edge_iterator():
                print(e[0], e[1]) # e[0] is the source, e[1] is the target

            for e in g.in_edge_iterator(1):
                ... # all edges e such that e[1] == 1

            for e in g.out_edge_iterator(1):
                ... # all edges e such that e[0] == 1

Edge indexes
************

``regular_graph`` and ``tree`` are also able to represent their edges by positive integers (``index_t`` in c++), suitable for array indexing.
Operations are done in constant time.

.. list-table::
    :header-rows: 1

    *   - Function
        - Returns
        - Description
        - Available
    *   - ``edge``
        - a pair of vertex indices
        - get an edge from its index
        - ``ugraph``, ``tree``
    *   - ``edge_index_iterator``
        - a range of iterators
        - iterator on the indices of every edge of the graph
        - ``ugraph``, ``tree``
    *   - ``out_edge_index_iterator``
        - a range of iterators
        - iterator on the every edge index ei that is an out-edge of the given vertex
        - ``ugraph``, ``tree``
    *   - ``in_edge_index_iterator``
        - a range of iterators
        - iterator on the every edge index ei that is an in-edge of the given vertex
        - ``ugraph``, ``tree``


.. tabs::

    .. tab:: c++

        .. code-block:: cpp
            :linenos:

            ugraph g;
            ...

            auto e = edge(0, g); // first edge of g

            for(auto ei: edge_index_iterator(g){
                ... // indices of every edge of g
            }

            for(auto ei: out_edge_index_iterator(1, g)){
                ... // indices of every edge of g whose source is vertex 1
            }

            for(auto ei: in_edge_index_iterator(1, g)){
                ... // indices of every edge of g whose target is vertex 1
            }


    .. tab:: python

        .. code-block:: python
            :linenos:

            g = hg.UndirectedGraph()
            ...

            e = g.edge(0) // first edge of g

            for ei in g.edge_index_iterator():
                ... # indices of every edge of g

            for ei in g.out_edge_index_iterator(1):
                ... # indices of every edge of g whose source is vertex 1

            for ei in g.in_edge_index_iterator(1):
                ... # indices of every edge of g whose target is vertex 1


Degrees
-------

Currently, all the graphs are undirected, meaning that the degree, the out-degree and the in-degree of a vertex are all equal.
Operations are done in constant time in ``ugraph``, ``tree``. Operations are done in time proportional to :math:`|E|/|V|` in ``regular_graph``.


.. list-table::
    :header-rows: 1

    *   - Function
        - Returns
        - Description
        - Available
    *   - ``degree``
        - a positive integer
        - number of edges containing the given vertex as either the source or the target
        - ``regular_graph``, ``ugraph``, ``tree``
    *   - ``in_degree``
        - a positive integer
        - number of edges containing the given vertex as the target
        - ``regular_graph``, ``ugraph``, ``tree``
    *   - ``degree``
        - a positive integer
        - number of edges containing the given vertex as either the source or the target
        - ``regular_graph``, ``ugraph``, ``tree``


.. tabs::

    .. tab:: c++

        .. code-block:: cpp
            :linenos:

            ugraph g;
            ...

            // degree of vertex 1
            auto d1 = degree(1, g);

            // in degree of vertex 2
            auto d2 = in_degree(2, g);

            // out degree of vertex 3
            auto d3 = out_degree(3, g);


    .. tab:: python

        .. code-block:: python
            :linenos:

            g = hg.UndirectedGraph()
            ...

            # degree of vertex 1
            d1 = g.degree(1)

            # in degree of vertex 2
            d2 = g.in_degree(2)

            # out degree of vertex 3
            d3 = g.out_degree(3)


Weighted graph
--------------

Higra enforces a strong separation between graphs and weights (on vertices or edges): a graph never stores weights.
Vertex indices and edge indices (except for ``regular_graph``) enables to have an immediate mapping between vertices
or edges and values stored in an array. The preferred storage for weights are ``xtensor`` containers in c++ and ``numpy``
arrays in python.

.. tabs::

    .. tab:: c++

        .. code-block:: cpp
            :linenos:

            // compute the sum of vertex weights adjacent to given vertex
            auto sum_adjacent_vertices_weights(
                const ugraph &g,
                const array_1d<double> &vertex_weights,
                index_t vertex){
                double result = 0;
                for(auto v: adjacent_vertex_iterator(vertex, g)){
                    result += vertex_weights[v];
                }
                return result
            }


    .. tab:: python

        .. code-block:: python
            :linenos:

            def sum_adjacent_vertices_weights(graph, vertex_weights, vertex):
                result = 0
                for v in g.adjacent_vertex_iterator(vertex);
                    result += vertex_weights[v]
                return result
