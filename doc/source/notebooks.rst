.. _notebooks:

Python notebooks
================

The following python notebooks contain examples demonstrating Higra usage.
Data can be points, images, or meshes, or anything that can be transformed into a graph.


Component trees
---------------
These examples deal with upper and lower treshold set of vertex-weighted graphs.

================================================================= ============= ============= ============
Connected image filtering with component trees                        |v3|_        |dl3|_        |co3|_
Filtering with non-increasing criterion - The shaping framework       |v9|_        |dl9|_        |co9|_
Pattern spectra - granulometry based on connected filters             |v15|_       |dl15|_       |co15|_
================================================================= ============= ============= ============

Hierarchical segmentation
-------------------------
These examples deal with images, weights are on edges of the associated graph.

================================================================= ============= ============= ============
Visualizing hierarchical image segmentation                           |v13|_       |dl13|_       |co13|_
Watershed hierarchies                                                 |v2|_        |dl2|_        |co2|_
Hierarchy filtering                                                   |v1|_        |dl1|_        |co1|_
Computing a saliency map with the shaping framework                   |v8|_        |dl8|_        |co8|_
Multiscale Hierarchy Alignment and Combination                        |v4|_        |dl4|_        |co4|_
================================================================= ============= ============= ============

Triangular meshes
-----------------
We provide two examples.

1. The first one uses trimesh, a simple, pure-python. It can be slow, and not-memory efficient.

2. The second one uses igl, an efficient C++ geometry processing library, with python bindings.

================================================================= ============= ============= ============
Hierarchical mesh segmentation -- trimesh                             |v16|_       |dl16|_       |co17|_
Hierarchical mesh segmentation -- igl                                 |v17|_       |dl16|_       |co17|_
================================================================= ============= ============= ============

Useful tools
------------
================================================================= ============= ============= ============
Region Adjacency Graph                                                |v5|_        |dl5|_        |co5|_
Interactive object segmentation                                       |v6|_        |dl6|_        |co6|_
Contour Simplification                                                |v7|_        |dl7|_        |co7|_
================================================================= ============= ============= ============

Illustrative applications from scientific papers
------------------------------------------------
================================================================= ============= ============= ============
Points and Images - Illustrations of SoftwareX 2019 article           |v10|_       |dl10|_       |co10|_
Non-relevant node removal, on both point and images. PRL 2019         |v11|_       |dl11|_       |co11|_
Fuzzy-marker-based interactive object segmentation - DGMM 2021        |v14|_       |dl14|_       |co14|_
Astronomical object detection with the Max-Tree - MMTA 2016           |v12|_       |dl12|_       |co12|_
================================================================= ============= ============= ============



.. |v1| unicode:: &#x1f441; .. view
.. _v1: https://github.com/higra/Higra-Notebooks/blob/master/Hierarchy%20filtering.ipynb

.. |dl1| unicode:: &#x1f4be; .. download
.. _dl1: https://cdn.jsdelivr.net/gh/higra/Higra-Notebooks/Hierarchy%20filtering.ipynb

.. |co1| image:: /images/colab.png
.. _co1: https://colab.research.google.com/github/higra/Higra-Notebooks/blob/master/Hierarchy%20filtering.ipynb


.. |v2| unicode:: &#x1f441; .. view
.. _v2: https://github.com/higra/Higra-Notebooks/blob/master/Watershed%20hierarchies.ipynb

.. |dl2| unicode:: &#x1f4be; .. download
.. _dl2: https://cdn.jsdelivr.net/gh/higra/Higra-Notebooks/Watershed%20hierarchies.ipynb

.. |co2| image:: /images/colab.png
.. _co2: https://colab.research.google.com/github/higra/Higra-Notebooks/blob/master/Watershed%20hierarchies.ipynb


.. |v3| unicode:: &#x1f441; .. view
.. _v3: https://github.com/higra/Higra-Notebooks/blob/master/Connected%20image%20filtering%20with%20component%20trees.ipynb

.. |dl3| unicode:: &#x1f4be; .. download
.. _dl3: https://cdn.jsdelivr.net/gh/higra/Higra-Notebooks/Connected%20image%20filtering%20with%20component%20trees.ipynb

.. |co3| image:: /images/colab.png
.. _co3: https://colab.research.google.com/github/higra/Higra-Notebooks/blob/master/Connected%20image%20filtering%20with%20component%20trees.ipynb


.. |v4| unicode:: &#x1f441; .. view
.. _v4: https://github.com/higra/Higra-Notebooks/blob/master/Multiscale%20Hierarchy%20Alignment%20and%20Combination.ipynb

.. |dl4| unicode:: &#x1f4be; .. download
.. _dl4: https://cdn.jsdelivr.net/gh/higra/Higra-Notebooks/Multiscale%20Hierarchy%20Alignment%20and%20Combination.ipynb

.. |co4| image:: /images/colab.png
.. _co4: https://colab.research.google.com/github/higra/Higra-Notebooks/blob/master/Multiscale%20Hierarchy%20Alignment%20and%20Combination.ipynb


.. |v5| unicode:: &#x1f441; .. view
.. _v5: https://github.com/higra/Higra-Notebooks/blob/master/Region%20Adjacency%20Graph.ipynb

.. |dl5| unicode:: &#x1f4be; .. download
.. _dl5: https://cdn.jsdelivr.net/gh/higra/Higra-Notebooks/Region%20Adjacency%20Graph.ipynb

.. |co5| image:: /images/colab.png
.. _co5: https://colab.research.google.com/github/higra/Higra-Notebooks/blob/master/Region%20Adjacency%20Graph.ipynb


.. |v6| unicode:: &#x1f441; .. view
.. _v6: https://github.com/higra/Higra-Notebooks/blob/master/Interactive%20object%20segmentation.ipynb

.. |dl6| unicode:: &#x1f4be; .. download
.. _dl6: https://cdn.jsdelivr.net/gh/higra/Higra-Notebooks/Interactive%20object%20segmentation.ipynb

.. |co6| image:: /images/colab.png
.. _co6: https://colab.research.google.com/github/higra/Higra-Notebooks/blob/master/Interactive%20object%20segmentation.ipynb


.. |v7| unicode:: &#x1f441; .. view
.. _v7: https://github.com/higra/Higra-Notebooks/blob/master/Contour%20Simplification.ipynb

.. |dl7| unicode:: &#x1f4be; .. download
.. _dl7: https://cdn.jsdelivr.net/gh/higra/Higra-Notebooks/Contour%20Simplification.ipynb

.. |co7| image:: /images/colab.png
.. _co7: https://colab.research.google.com/github/higra/Higra-Notebooks/blob/master/Contour%20Simplification.ipynb


.. |v8| unicode:: &#x1f441; .. view
.. _v8: https://github.com/higra/Higra-Notebooks/blob/master/Computing%20a%20saliency%20map%20with%20the%20shaping%20framework.ipynb

.. |dl8| unicode:: &#x1f4be; .. download
.. _dl8: https://cdn.jsdelivr.net/gh/higra/Higra-Notebooks/Computing%20a%20saliency%20map%20with%20the%20shaping%20framework.ipynb

.. |co8| image:: /images/colab.png
.. _co8: https://colab.research.google.com/github/higra/Higra-Notebooks/blob/master/Computing%20a%20saliency%20map%20with%20the%20shaping%20framework.ipynb


.. |v9| unicode:: &#x1f441; .. view
.. _v9: https://github.com/higra/Higra-Notebooks/blob/master/Filtering%20with%20non%20increasing%20criterion%20-%20The%20shaping%20framework.ipynb

.. |dl9| unicode:: &#x1f4be; .. download
.. _dl9: https://cdn.jsdelivr.net/gh/higra/Higra-Notebooks/Filtering%20with%20non%20increasing%20criterion%20-%20The%20shaping%20framework.ipynb

.. |co9| image:: /images/colab.png
.. _co9: https://colab.research.google.com/github/higra/Higra-Notebooks/blob/master/Filtering%20with%20non%20increasing%20criterion%20-%20The%20shaping%20framework.ipynb



.. |v10| unicode:: &#x1f441; .. view
.. _v10: https://github.com/higra/Higra-Notebooks/blob/master/Illustrations%20of%20SoftwareX%202019%20article.ipynb

.. |dl10| unicode:: &#x1f4be; .. download
.. _dl10: https://cdn.jsdelivr.net/gh/higra/Higra-Notebooks/Illustrations%20of%20SoftwareX%202019%20article.ipynb

.. |co10| image:: /images/colab.png
.. _co10: https://colab.research.google.com/github/higra/Higra-Notebooks/blob/master/Illustrations%20of%20SoftwareX%202019%20article.ipynb


.. |v11| unicode:: &#x1f441; .. view
.. _v11: https://github.com/higra/Higra-Notebooks/blob/master/Illustrations%20of%20Pattern%20Recognition%20Letters%202019%20article.ipynb

.. |dl11| unicode:: &#x1f4be; .. download
.. _dl11: https://cdn.jsdelivr.net/gh/higra/Higra-Notebooks/Illustrations%20of%20Pattern%20Recognition%20Letters%202019%20article.ipynb

.. |co11| image:: /images/colab.png
.. _co11: https://colab.research.google.com/github/higra/Higra-Notebooks/blob/master/Illustrations%20of%20Pattern%20Recognition%20Letters%202019%20article.ipynb


.. |v12| unicode:: &#x1f441; .. view
.. _v12: https://github.com/higra/Higra-Notebooks/blob/master/Astronomical%20object%20detection%20with%20the%20Max-Tree.ipynb

.. |dl12| unicode:: &#x1f4be; .. download
.. _dl12: https://cdn.jsdelivr.net/gh/higra/Higra-Notebooks/Astronomical%20object%20detection%20with%20the%20Max-Tree.ipynb

.. |co12| image:: /images/colab.png
.. _co12: https://colab.research.google.com/github/higra/Higra-Notebooks/blob/master/Astronomical%20object%20detection%20with%20the%20Max-Tree.ipynb


.. |v13| unicode:: &#x1f441; .. view
.. _v13: https://github.com/higra/Higra-Notebooks/blob/master/Visualizing%20hierarchical%20image%20segmentations.ipynb

.. |dl13| unicode:: &#x1f4be; .. download
.. _dl13: https://cdn.jsdelivr.net/gh/higra/Higra-Notebooks/Visualizing%20hierarchical%20image%20segmentations.ipynb

.. |co13| image:: /images/colab.png
.. _co13: https://colab.research.google.com/github/higra/Higra-Notebooks/blob/master/Visualizing%20hierarchical%20image%20segmentations.ipynb


.. |v14| unicode:: &#x1f441; .. view
.. _v14: https://github.com/higra/Higra-Notebooks/blob/master/Fuzzy-marker-based%20segmentation%20using%20%20hierarchies.ipynb

.. |dl14| unicode:: &#x1f4be; .. download
.. _dl14: https://cdn.jsdelivr.net/gh/higra/Higra-Notebooks/Fuzzy-marker-based%20segmentation%20using%20%20hierarchies.ipynb

.. |co14| image:: /images/colab.png
.. _co14: https://colab.research.google.com/github/higra/Higra-Notebooks/blob/master/Fuzzy-marker-based%20segmentation%20using%20%20hierarchies.ipynb


.. |v15| unicode:: &#x1f441; .. view
.. _v15: https://github.com/higra/Higra-Notebooks/blob/master/Pattern%20spectra%20-%20granulometry%20based%20on%20connected%20filters.ipynb

.. |dl15| unicode:: &#x1f4be; .. download
.. _dl15: https://cdn.jsdelivr.net/gh/higra/Higra-Notebooks/Pattern%20spectra%20-%20granulometry%20based%20on%20connected%20filters.ipynb

.. |co15| image:: /images/colab.png
.. _co15: https://colab.research.google.com/github/higra/Higra-Notebooks/blob/master/Pattern%20spectra%20-%20granulometry%20based%20on%20connected%20filters.ipynb


.. |v16| unicode:: &#x1f441; .. view
.. _v16: https://github.com/higra/Higra-Notebooks/blob/master/Watershed_on_a_Mesh_Trimesh.ipynb

.. |dl16| unicode:: &#x1f4be; .. download
.. _dl16: https://cdn.jsdelivr.net/gh/higra/Higra-Notebooks/Watershed_on_a_Mesh_Trimesh.ipynb

.. |co16| image:: /images/colab.png
.. _co16: https://colab.research.google.com/github/higra/Higra-Notebooks/blob/master/Watershed_on_a_Mesh_Trimesh.ipynb


.. |v17| unicode:: &#x1f441; .. view
.. _v17: https://github.com/higra/Higra-Notebooks/blob/master/Watershed_on_a_Mesh_IGL.ipynb

.. |dl17| unicode:: &#x1f4be; .. download
.. _dl17: https://cdn.jsdelivr.net/gh/higra/Higra-Notebooks/Watershed_on_a_Mesh_IGL.ipynb

.. |co17| image:: /images/colab.png
.. _co17: https://colab.research.google.com/github/higra/Higra-Notebooks/blob/master/Watershed_on_a_Mesh_IGL.ipynb


