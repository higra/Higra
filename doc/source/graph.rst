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

All functions acting on graphs have the same name in C++ and in Python, except for iterators to avoid name collisions with the `Boost Graph Library (BGL) <https://www.boost.org/doc/libs/1_67_0/libs/graph/doc/index.html>`_.
In c++, graph related methods are free functions (as in BGL), while in Python they are member functions.
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
    *   - ``vertex_iterator`` (c++) ``vertices`` (python)
        - a range of iterators
        - iterator on all graph vertices
        - ``regular_graph``, ``ugraph``, ``tree``
    *   - ``adjacent_vertex_iterator`` (c++) ``adjacent_vertices`` (python)
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

                for v in g.vertices():
                    ... # all vertices of g

                for v in g.adjacent_vertices(1):
                    ... # all vertices adjacent to vertex 1 in g

Edges
-----

Graph edges are composed of a source vertex, a target vertex, and, optionally, an index.

Graphs which have indexed edges provide the following guaranties:

* edge indices of a graph ``g`` are integers (type ``index_t``) comprised between 0 (included) and ``num_edges(g)`` (excluded);
* the index of a given edge will never change during the object lifetime.

However, note that in an undirected graph, the edges ``(x, y)`` and ``(y, x)`` have the same index.

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
    *   - ``add_edges``
        - void
        - add all edges given as a pair of arrays (sources, targets) to the graph
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
    *   - ``index``
        - edge index
        - the index of the given edge in the current graph
        - ``ugraph``, ``tree``
    *   - ``edge_from_index``
        - edge
        - the edge with given index (in an undirected graph, always returns the edge whose source vertex is smaller than the target vertex)
        - ``ugraph``, ``tree``
    *   - ``edge_list``
        - void
        - a pair of arrays (sources, targets) defining all the edges of the graph
        - ``ugraph``, ``tree``

Note that python's edges are simply tuples whose first value is the source vertex, second value is the target vertex,
and third (optional) value is the index.

Example:

.. tabs::

    .. tab:: c++

        .. code-block:: cpp
            :linenos:

            #include "higra/graph.hpp"
            using namespace hg;

            // create a graph with 4 vertices and no edge
            ugraph g(4);

            // add an edge, between vertex 0 and 1
            add_edge(0, 1, g);
            // add an edge, between vertex 0 and 1
            auto e = add_edge(1, 2, g);

            auto s = source(e, g); // 1
            auto t = target(e, g); // 2
            auto ei = index(e, g); // 1

            // add the two edges (3, 0) and (3, 1)
            add_edges({3, 3}, {0, 1});

            auto ne = num_edges(g); // 4

            auto edges = edge_list(g); // edges.first = {0, 1, 0, 1}, edges.second = {1, 2, 3, 3}

    .. tab:: python

        .. code-block:: python
            :linenos:

            import higra as hg

            # create a graph with 4 vertices and no edge
            g = hg.UndirectedGraph(4)

            # add an edge, between vertex 0 and 1
            g.add_edge(0, 1)
            # add an edge, between vertex 0 and 1
            e = g.add_edge(1, 2)

            s = g.source(e) # 1 or equivalently e[0]
            t = g.target(e) # 2 or equivalently e[1]
            ei = g.index(e) # 1 or equivalently e[2]

            # add the two edges (3, 0) and (3, 1)
            g.add_edges((3, 3), (0, 1));

            ne = g.num_edges() # 4

            sources, targets = g.edge_list() # sources = [0, 1, 0, 1], targets = [1, 2, 3, 3]


Iterating on edges
******************

.. list-table::
    :header-rows: 1

    *   - Function
        - Returns
        - Description
        - Available
    *   - ``edge_iterator``  (c++) ``edges`` (python)
        - a range of iterators
        - iterator on graph edges
        - ``regular_graph``, ``ugraph``, ``tree``
    *   - ``in_edge_iterator``  (c++) ``in_edges`` (python)
        - a range of iterators
        - iterators on all edges whose target is the given vertex
        - ``regular_graph``, ``ugraph``, ``tree``
    *   - ``out_edge_iterator``  (c++) ``out_edges`` (python)
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

            for e in g.edges():
                print(g.source(e), g.target(e))

            for e in g.in_edges(1):
                ... # all edges e such that g.target(e) == 1

            for e in g.out_edges(1):
                ... # all edges e such that g.source(e) == 1


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
            auto sum_adjacent_vertices_weights(const ugraph &g,
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
                for v in g.adjacent_vertices(vertex);
                    result += vertex_weights[v]
                return result
