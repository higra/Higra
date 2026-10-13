############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

import unittest
import numpy as np
import higra as hg

import os
import os.path
import pathlib
import subprocess
import sys
import tempfile


def silent_remove(filename):
    try:
        os.remove(filename)
    except:
        pass


class TestTreeIO(unittest.TestCase):

    def test_treeReadWrite(self):
        filename = "testTreeIO.graph"
        silent_remove(filename)

        parents = np.asarray((5, 5, 6, 6, 6, 7, 7, 7), dtype=np.uint64)
        tree = hg.Tree(parents)

        attr1 = np.asarray((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0))
        attr2 = np.asarray((8, 7, 6, 5, 4, 3, 2, 1), dtype=np.int32)

        hg.save_tree(filename, tree, {"attr1": attr1, "attr2": attr2})

        tree, attributes = hg.read_tree(filename)
        silent_remove(filename)

        self.assertTrue(np.allclose(tree.parents(), parents))

        self.assertTrue("attr1" in attributes)
        self.assertTrue(np.allclose(attr1, attributes["attr1"]))

        self.assertTrue("attr2" in attributes)
        self.assertTrue(np.allclose(attr2, attributes["attr2"]))

        # Test without attributes
        hg.save_tree(filename, tree)

        tree, attributes = hg.read_tree(filename)
        silent_remove(filename)

        self.assertTrue(np.allclose(tree.parents(), parents))

    def test_bounded_read_failures(self):
        child = """
import sys
sys.path.insert(0, sys.argv[2])
import higra as hg
try:
    hg.read_tree(sys.argv[1])
except Exception as e:
    print(type(e).__name__, str(e))
    sys.exit(0)
sys.exit(1)
"""
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "input.tree"
            hg.save_tree(str(path), hg.Tree((2, 2, 2)), {"a": np.arange(3)})
            valid = path.read_bytes()
            parent_start = valid.index(b"END\n") + 4
            attribute_start = parent_start + 3 * np.dtype(np.intc).itemsize
            payload_start = valid.index(b"END\n", attribute_start) + 4
            cases = [None, b"", b"VERSION=1\n", b"NBNODES=3\nEND\n",
                     b"NBATTR=0\nEND\n", b"VERSION=bad\n", b"VERSION=2\n",
                     b"NBNODES=3junk\n", b"NBNODES=-1\n",
                     b"CATEGORY=x\n", b"CATEGORY=2\n", b"CATEGORY=-1\n",
                     b"NBNODES=2147483647\nNBATTR=0\nEND\n",
                     valid[:parent_start], valid[:attribute_start - 1],
                     valid[:attribute_start], valid[:payload_start - 2], valid[:-1],
                     b"VERSION=1\nNBNODES=3\nNBATTR=0\nEND\n" +
                     np.array([10, 2, 2], dtype=np.intc).tobytes()]
            path.unlink()
            for data in cases:
                with self.subTest(data=data):
                    if data is not None:
                        path.write_bytes(data)
                    result = subprocess.run([sys.executable, "-c", child, str(path),
                                             str(pathlib.Path(hg.__file__).resolve().parent.parent)],
                                            capture_output=True, text=True, timeout=15)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertTrue(result.stdout.strip())
                    self.assertIn("tree", result.stdout.lower())

    def test_binary_legacy_and_metadata(self):
        parents = np.full(27, 26, dtype=np.intc)  # newline/control bytes in native payload
        attributes = {"control": np.array([0x1a0d0a] * 27, dtype=np.uint64).view(np.float64),
                      "integers": np.arange(27, dtype=np.int32)}
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "binary.tree"
            original = hg.Tree(parents, category=hg.TreeCategory.ComponentTree)
            self.assertIsNone(hg.save_tree(str(path), original, attributes))
            written = path.read_bytes()
            fixture = (b"VERSION=1\nNBNODES=27\nNBATTR=2\nEND\n" + parents.tobytes())
            for name, values in sorted(attributes.items()):
                fixture += ("NAME=" + name + "\nEND\n").encode() + values.astype(np.float64).tobytes()
            expected = fixture.replace(b"VERSION=1\n", b"VERSION=1\nCATEGORY=0\n", 1)
            expected = expected.replace(b"NBATTR=2\n", b"NBATTR=2" + b" " * 28 + b"\n", 1)
            self.assertEqual(written, expected)
            for data, category in ((written, hg.TreeCategory.ComponentTree),
                                   (fixture, hg.TreeCategory.PartitionTree),
                                   (fixture[len(b"VERSION=1\n"):], hg.TreeCategory.PartitionTree)):
                path.write_bytes(data)
                tree, result = hg.read_tree(str(path))
                np.testing.assert_array_equal(tree.parents(), parents)
                self.assertEqual(tree.category(), category)
                for name, values in attributes.items():
                    np.testing.assert_array_equal(result[name], values)
                    self.assertEqual(result[name].dtype, np.float64)
                    self.assertIs(hg.get_attribute(tree, name), result[name])
            hg.save_tree(str(path), original)
            self.assertEqual(hg.read_tree(str(path))[1], {})

    def test_empty_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(pathlib.Path(directory) / "empty.tree")
            for category in (hg.TreeCategory.PartitionTree, hg.TreeCategory.ComponentTree):
                for attributes in ({}, {"empty": np.empty(0, dtype=np.int32)}):
                    hg.save_tree(path, hg.Tree([], category), attributes)
                    tree, result = hg.read_tree(path)
                    self.assertEqual(tree.category(), category)
                    self.assertEqual(tree.num_vertices(), 0)
                    self.assertEqual(tree.num_leaves(), 0)
                    self.assertEqual(tree.root(), -1)
                    self.assertEqual(tree.parents().size, 0)
                    self.assertEqual(set(result), set(attributes))
                    if attributes:
                        self.assertEqual(result["empty"].shape, (0,))
                        self.assertEqual(result["empty"].dtype, np.float64)
                        self.assertIs(hg.get_attribute(tree, "empty"), result["empty"])
            pathlib.Path(path).write_bytes(b"NBNODES=0\nNBATTR=0\nEND\n")
            self.assertEqual(hg.read_tree(path)[0].category(), hg.TreeCategory.PartitionTree)

    def test_output_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            for path in (directory, os.path.join(directory, "missing", "out.tree")):
                with self.assertRaises(RuntimeError):
                    hg.save_tree(path, hg.Tree((2, 2, 2)))
        if sys.platform.startswith("linux") and os.path.exists("/dev/full"):
            with self.assertRaises(RuntimeError):
                hg.save_tree("/dev/full", hg.Tree((2, 2, 2)))

    def test_print_partition_tree(self):
        tree = hg.Tree((5, 5, 6, 6, 6, 7, 7, 7))
        s = hg.print_partition_tree(tree, altitudes=np.asarray([0, 0, 0, 0, 0, 100, 1100, 20000]),
                                    scale="log",
                                    return_string=True)
        ref = ("0------\n"
               "      |\n"
               "    100------------------------------------------------------------------------------------------------\n"
               "      |                                                                                               |\n"
               "1------                                                                                               |\n"
               "                                                                                                      |\n"
               "                                                                                                   2e+04\n"
               "2-------------                                                                                        |\n"
               "             |                                                                                        |\n"
               "             |                                                                                        |\n"
               "3---------1.1e+03--------------------------------------------------------------------------------------\n"
               "             |\n"
               "             |\n"
               "             |\n"
               "4-------------")

        self.assertTrue(ref == s)

    # this is really just a non throw test...
    def test_print_partition_tree(self):
        tree = hg.Tree((5, 5, 6, 6, 6, 7, 7, 7))

        for o in ['altitudes', 'area', 'none', 'leaves']:
            for s in ["linear", "log"]:
                hg.print_partition_tree(tree,
                                        altitudes=np.asarray([0, 0, 0, 0, 0, 100, 1100, 20000]),
                                        attribute=np.asarray([0, 0, 0, 0, 0, 100, 1100, 20000]),
                                        ordering=o,
                                        scale=s,
                                        return_string=True)


if __name__ == '__main__':
    unittest.main()
