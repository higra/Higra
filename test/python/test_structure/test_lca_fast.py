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


class TestLCAFast(unittest.TestCase):

    @staticmethod
    def getTree():
        parent_relation = np.asarray((5, 5, 6, 6, 6, 7, 7, 7), dtype=np.uint64)
        return hg.Tree(parent_relation)

    def test_LCAFast(self):
        t = TestLCAFast.getTree()
        for lca_t in [hg.LCA_rmq_sparse_table, hg.LCA_rmq_sparse_table_block]:
            with self.subTest(lca_type=lca_t):
                lca = lca_t(t)

                self.assertTrue(lca.lca(0, 0) == 0)
                self.assertTrue(lca.lca(3, 3) == 3)
                self.assertTrue(lca.lca(5, 5) == 5)
                self.assertTrue(lca.lca(7, 7) == 7)
                self.assertTrue(lca.lca(0, 1) == 5)
                self.assertTrue(lca.lca(1, 0) == 5)
                self.assertTrue(lca.lca(2, 3) == 6)
                self.assertTrue(lca.lca(2, 4) == 6)
                self.assertTrue(lca.lca(3, 4) == 6)
                self.assertTrue(lca.lca(5, 6) == 7)
                self.assertTrue(lca.lca(0, 2) == 7)
                self.assertTrue(lca.lca(1, 4) == 7)
                self.assertTrue(lca.lca(2, 6) == 6)

    def test_LCA_sp_block_size(self):
        t = TestLCAFast.getTree()

        lca = hg.LCA_rmq_sparse_table_block(t, 2)

        self.assertTrue(lca.lca(0, 0) == 0)
        self.assertTrue(lca.lca(3, 3) == 3)
        self.assertTrue(lca.lca(5, 5) == 5)
        self.assertTrue(lca.lca(7, 7) == 7)
        self.assertTrue(lca.lca(0, 1) == 5)
        self.assertTrue(lca.lca(1, 0) == 5)
        self.assertTrue(lca.lca(2, 3) == 6)
        self.assertTrue(lca.lca(2, 4) == 6)
        self.assertTrue(lca.lca(3, 4) == 6)
        self.assertTrue(lca.lca(5, 6) == 7)
        self.assertTrue(lca.lca(0, 2) == 7)
        self.assertTrue(lca.lca(1, 4) == 7)
        self.assertTrue(lca.lca(2, 6) == 6)

    def test_LCAFastV(self):
        g = hg.get_4_adjacency_graph((2, 2))
        t = hg.Tree((4, 4, 5, 5, 6, 6, 6))
        for lca_t in [hg.LCA_rmq_sparse_table, hg.LCA_rmq_sparse_table_block]:
            with self.subTest(lca_type=lca_t):
                lca = lca_t(t)

                ref = (4, 6, 6, 5)
                res = lca.lca(g)
                self.assertTrue(np.all(ref == res))

    def test_LCAFastVertices(self):
        t = hg.Tree((4, 4, 5, 5, 6, 6, 6))
        for lca_t in [hg.LCA_rmq_sparse_table, hg.LCA_rmq_sparse_table_block]:
            with self.subTest(lca_type=lca_t):
                lca = lca_t(t)
                res = lca.lca((0, 0, 1, 3), (0, 3, 0, 0))
                self.assertTrue(np.all(res == (0, 6, 4, 6)))

    def test_dynamic_attributes(self):
        t = hg.Tree((4, 4, 5, 5, 6, 6, 6))
        for lca_t in [hg.LCA_rmq_sparse_table, hg.LCA_rmq_sparse_table_block]:
            with self.subTest(lca_type=lca_t):
                lca = lca_t(t)
                lca.new_attribute = 42
                self.assertTrue(lca.new_attribute == 42)

    def test_pickle(self):
        import pickle
        tree = hg.Tree((4, 4, 5, 5, 6, 6, 6))
        for lca_t in [hg.LCA_rmq_sparse_table, hg.LCA_rmq_sparse_table_block]:
            with self.subTest(lca_type=lca_t):
                lca = lca_t(tree)
                lca.test = (1, 2, 3)
                lca.__dict__.setdefault("__higra_tags__", set()).update({"foo"})

                data = pickle.dumps(lca)
                lca2 = pickle.loads(data)

                res = lca2.lca((0, 0, 1, 3), (0, 3, 0, 0))
                self.assertTrue(np.all(res == (0, 6, 4, 6)))

                self.assertTrue(getattr(lca, "test", None) == getattr(lca2, "test", None))
                self.assertTrue(lca.test == lca2.test)
                self.assertTrue("foo" in lca2.__higra_tags__)

    # Objects pickled by Higra 0.6.13 (NumPy 2, pickle protocol 2) for the tree (3, 3, 4, 4, 4):
    # pickles must remain loadable by later versions.
    _pickled_sparse_table = (
        "gAJjaGlncmEuc3RydWN0dXJlLmxjYV9mYXN0Cl9fcmVkdWNlX2N0cl9sY2Ffc3QKcQBdcQEoY251"
        "bXB5Ll9jb3JlLm11bHRpYXJyYXkKX3JlY29uc3RydWN0CnECY251bXB5Cm5kYXJyYXkKcQNLAIVx"
        "BGNfY29kZWNzCmVuY29kZQpxBVgBAAAAYnEGWAYAAABsYXRpbjFxB4ZxCFJxCYdxClJxCyhLAUsJ"
        "hXEMY251bXB5CmR0eXBlCnENWAIAAABpOHEOiYiHcQ9ScRAoSwNYAQAAADxxEU5OTkr/////Sv//"
        "//9LAHRxEmKJaAVYSAAAAAQAAAAAAAAAAwAAAAAAAAABAAAAAAAAAAMAAAAAAAAAAAAAAAAAAAAD"
        "AAAAAAAAAAQAAAAAAAAAAgAAAAAAAAAEAAAAAAAAAHETaAeGcRRScRV0cRZiaAJoA0sAhXEXaAmH"
        "cRhScRkoSwFLCYVxGmgQiWgFWEgAAAAAAAAAAAAAAAEAAAAAAAAAAgAAAAAAAAABAAAAAAAAAAIA"
        "AAAAAAAAAQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAABxG2gHhnEcUnEddHEeYmgCaANL"
        "AIVxH2gJh3EgUnEhKEsBSwWFcSJoEIloBVgoAAAABAAAAAAAAAACAAAAAAAAAAcAAAAAAAAAAQAA"
        "AAAAAAAAAAAAAAAAAHEjaAeGcSRScSV0cSZiXXEnKGgCaANLAIVxKGgJh3EpUnEqKEsBSwmFcSto"
        "DVgCAAAAdThxLImIh3EtUnEuKEsDaBFOTk5K/////0r/////SwB0cS9iiWgFWEgAAAAAAAAAAAAA"
        "AAEAAAAAAAAAAgAAAAAAAAADAAAAAAAAAAQAAAAAAAAABQAAAAAAAAAGAAAAAAAAAAcAAAAAAAAA"
        "CAAAAAAAAABxMGgHhnExUnEydHEzYmgCaANLAIVxNGgJh3E1UnE2KEsBSwiFcTdoLoloBVhAAAAA"
        "AAAAAAAAAAABAAAAAAAAAAMAAAAAAAAAAwAAAAAAAAAFAAAAAAAAAAYAAAAAAAAABgAAAAAAAAAI"
        "AAAAAAAAAHE4aAeGcTlScTp0cTtiaAJoA0sAhXE8aAmHcT1ScT4oSwFLBoVxP2guiWgFWDAAAAAA"
        "AAAAAAAAAAMAAAAAAAAABQAAAAAAAAAGAAAAAAAAAAYAAAAAAAAACAAAAAAAAABxQGgHhnFBUnFC"
        "dHFDYmgCaANLAIVxRGgJh3FFUnFGKEsBSwKFcUdoLoloBVgQAAAABgAAAAAAAAAIAAAAAAAAAHFI"
        "aAeGcUlScUp0cUtiZWWFcUxScU19cU5iLg==")

    _pickled_sparse_table_block = (
        "gAJjaGlncmEuc3RydWN0dXJlLmxjYV9mYXN0Cl9fcmVkdWNlX2N0cl9sY2Ffc3RiCnEAXXEBKGNu"
        "dW1weS5fY29yZS5tdWx0aWFycmF5Cl9yZWNvbnN0cnVjdApxAmNudW1weQpuZGFycmF5CnEDSwCF"
        "cQRjX2NvZGVjcwplbmNvZGUKcQVYAQAAAGJxBlgGAAAAbGF0aW4xcQeGcQhScQmHcQpScQsoSwFL"
        "CYVxDGNudW1weQpkdHlwZQpxDVgCAAAAaThxDomIh3EPUnEQKEsDWAEAAAA8cRFOTk5K/////0r/"
        "////SwB0cRJiiWgFWEgAAAAEAAAAAAAAAAMAAAAAAAAAAQAAAAAAAAADAAAAAAAAAAAAAAAAAAAA"
        "AwAAAAAAAAAEAAAAAAAAAAIAAAAAAAAABAAAAAAAAABxE2gHhnEUUnEVdHEWYmgCaANLAIVxF2gJ"
        "h3EYUnEZKEsBSwmFcRpoEIloBVhIAAAAAAAAAAAAAAABAAAAAAAAAAIAAAAAAAAAAQAAAAAAAAAC"
        "AAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAcRtoB4ZxHFJxHXRxHmJoAmgD"
        "SwCFcR9oCYdxIFJxIShLAUsFhXEiaBCJaAVYKAAAAAQAAAAAAAAAAgAAAAAAAAAHAAAAAAAAAAEA"
        "AAAAAAAAAAAAAAAAAABxI2gHhnEkUnEldHEmYl1xJyhLCUsCSwVoAmgDSwCFcShoCYdxKVJxKihL"
        "AUsKhXEraBCJaAVYUAAAAAAAAAAAAAAAAAAAAAAAAAACAAAAAAAAAAMAAAAAAAAABAAAAAAAAAAF"
        "AAAAAAAAAAYAAAAAAAAABgAAAAAAAAAIAAAAAAAAAAgAAAAAAAAAcSxoB4ZxLVJxLnRxL2JoAmgD"
        "SwCFcTBoCYdxMVJxMihLAUsKhXEzaBCJaAVYUAAAAAAAAAAAAAAAAQAAAAAAAAADAAAAAAAAAAMA"
        "AAAAAAAABQAAAAAAAAAFAAAAAAAAAAYAAAAAAAAABwAAAAAAAAAJAAAAAAAAAAkAAAAAAAAAcTRo"
        "B4ZxNVJxNnRxN2JdcTgoaAJoA0sAhXE5aAmHcTpScTsoSwFLBYVxPGgNWAIAAAB1OHE9iYiHcT5S"
        "cT8oSwNoEU5OTkr/////Sv////9LAHRxQGKJaAVYKAAAAAAAAAAAAAAAAwAAAAAAAAAFAAAAAAAA"
        "AAYAAAAAAAAACAAAAAAAAABxQWgHhnFCUnFDdHFEYmgCaANLAIVxRWgJh3FGUnFHKEsBSwSFcUho"
        "P4loBVggAAAAAAAAAAAAAAAFAAAAAAAAAAYAAAAAAAAACAAAAAAAAABxSWgHhnFKUnFLdHFMYmgC"
        "aANLAIVxTWgJh3FOUnFPKEsBSwKFcVBoP4loBVgQAAAABgAAAAAAAAAIAAAAAAAAAHFRaAeGcVJS"
        "cVN0cVRiZWVlhXFVUnFWfXFXYi4=")

    @unittest.skipIf(int(np.__version__.split(".")[0]) < 2, "fixture pickled with NumPy 2")
    def test_unpickle_previous_version(self):
        import base64
        import pickle
        tree = hg.Tree((3, 3, 4, 4, 4))
        v1 = np.array((0, 0, 1, 3, 4))
        v2 = np.array((1, 2, 3, 2, 4))
        expected = (3, 4, 3, 4, 4)
        cases = ((hg.LCA_rmq_sparse_table, self._pickled_sparse_table, lambda: hg.LCA_rmq_sparse_table(tree)),
                 (hg.LCA_rmq_sparse_table_block, self._pickled_sparse_table_block,
                  lambda: hg.LCA_rmq_sparse_table_block(tree, 2)))
        for lca_t, data, make in cases:
            with self.subTest(lca_type=lca_t):
                lca = pickle.loads(base64.b64decode(data))
                self.assertIsInstance(lca, lca_t)
                self.assertTrue(np.all(lca.lca(v1, v2) == expected))
                for i in range(5):
                    for j in range(5):
                        self.assertEqual(lca.lca(i, j), tree.lowest_common_ancestor(i, j))

                # same state as an object built by the current version
                self.assertTrue(self._same_state(lca._get_state(), make()._get_state()))

                # and the state survives a new pickle round trip
                lca2 = pickle.loads(pickle.dumps(lca))
                self.assertTrue(self._same_state(lca._get_state(), lca2._get_state()))

    @staticmethod
    def _same_state(s1, s2):
        if isinstance(s1, (tuple, list)):
            return isinstance(s2, (tuple, list)) and len(s1) == len(s2) and \
                   all(TestLCAFast._same_state(a, b) for a, b in zip(s1, s2))
        if isinstance(s1, np.ndarray):
            return isinstance(s2, np.ndarray) and s1.dtype == s2.dtype and s1.shape == s2.shape and \
                   np.all(s1 == s2)
        return type(s1) == type(s2) and s1 == s2

    def test_state_format(self):
        tree = hg.Tree((4, 4, 5, 5, 6, 6, 6))
        for lca_t in [hg.LCA_rmq_sparse_table, hg.LCA_rmq_sparse_table_block]:
            with self.subTest(lca_type=lca_t):
                lca = lca_t(tree)
                state = lca._get_state()
                self.assertIsInstance(state, tuple)
                self.assertEqual(len(state), 1)
                state = state[0]
                self.assertEqual(len(state), 4)
                for a in state[:3]:
                    self.assertEqual(a.dtype, np.int64)
                    self.assertEqual(a.ndim, 1)
                    self.assertTrue(a.flags.writeable)
                if lca_t is hg.LCA_rmq_sparse_table:
                    sparse_table = state[3]
                else:
                    self.assertEqual(len(state[3]), 6)
                    for v in state[3][:3]:
                        self.assertIsInstance(v, int)
                    for a in state[3][3:5]:
                        self.assertEqual(a.dtype, np.int64)
                    sparse_table = state[3][5]
                self.assertIsInstance(sparse_table, list)
                for a in sparse_table:
                    self.assertEqual(a.dtype, np.uint64)

                # the returned arrays are copies: modifying them does not change the object
                state[0][:] = 0
                self.assertTrue(np.all(lca.lca((0, 0, 1, 3), (0, 3, 0, 0)) == (0, 6, 4, 6)))

                # states made of arrays of other dtypes or layouts are converted
                state = lca._get_state()[0]

                def convert(s):
                    if isinstance(s, list):
                        return [convert(e) for e in s]
                    if isinstance(s, np.ndarray):
                        # int32, non-contiguous
                        return np.repeat(s.astype(np.int32), 2)[::2]
                    return s

                lca2 = lca_t._make_from_state((convert(state),))
                self.assertTrue(np.all(lca2.lca((0, 0, 1, 3), (0, 3, 0, 0)) == (0, 6, 4, 6)))


if __name__ == '__main__':
    unittest.main()
