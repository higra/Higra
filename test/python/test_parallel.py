############################################################################
# Copyright ESIEE Paris (2026)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

import os
import subprocess
import sys
import unittest

import numpy as np
import higra as hg


def _run_with_environment(value, code):
    """
    Run ``code`` in a new Python process importing the same higra package, with HG_NUM_THREADS set to ``value``
    (or unset if ``value`` is None). Return the completed process.
    """
    package_parent = os.path.dirname(os.path.dirname(os.path.abspath(hg.__file__)))
    env = dict(os.environ)
    env.pop("HG_NUM_THREADS", None)
    if value is not None:
        env["HG_NUM_THREADS"] = value
    env["PYTHONPATH"] = package_parent + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, "-c", "import higra as hg\n" + code],
                          cwd=package_parent, env=env, capture_output=True, text=True)


class TestParallel(unittest.TestCase):

    def tearDown(self):
        hg.set_num_threads(0)

    def test_set_get_num_threads(self):
        hg.set_num_threads(0)
        default = hg.get_num_threads()
        self.assertGreaterEqual(default, 1)

        hg.set_num_threads(1)
        self.assertEqual(hg.get_num_threads(), 1)

        hg.set_num_threads(2)
        self.assertIn(hg.get_num_threads(), (1, 2))  # 1 without multi-threading support

        hg.set_num_threads(0)
        self.assertEqual(hg.get_num_threads(), default)

    def test_set_num_threads_negative(self):
        with self.assertRaises(TypeError):
            hg.set_num_threads(-1)

    def test_results_do_not_depend_on_num_threads(self):
        rng = np.random.default_rng(1)
        keys = rng.integers(0, 100, size=300000)
        values = rng.random(300000)

        def compute():
            return hg.arg_sort(keys, stable=True), hg.sort(values.copy())

        reference = compute()
        for threads in (1, 2, 8):
            hg.set_num_threads(threads)
            result = compute()
            np.testing.assert_array_equal(result[0], reference[0])
            np.testing.assert_array_equal(result[1], reference[1])

    def test_environment_variable(self):
        code = "print(hg.get_num_threads())"
        unset = _run_with_environment(None, code)
        self.assertEqual(unset.returncode, 0, unset.stderr)
        default = int(unset.stdout)

        one = _run_with_environment("1", code)
        self.assertEqual(one.returncode, 0, one.stderr)
        self.assertEqual(int(one.stdout), 1)
        self.assertNotIn("HG_NUM_THREADS", one.stderr)

        for value in ("", "  "):
            empty = _run_with_environment(value, code)
            self.assertEqual(empty.returncode, 0, empty.stderr)
            self.assertEqual(int(empty.stdout), default)
            self.assertNotIn("HG_NUM_THREADS", empty.stderr)

        for valid in (" 1 ", "\t1\n", "01"):
            result = _run_with_environment(valid, code)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(int(result.stdout), 1)
            self.assertNotIn("HG_NUM_THREADS", result.stderr)

        largest = _run_with_environment(str(2 ** 63 - 1), code)
        self.assertEqual(largest.returncode, 0, largest.stderr)
        self.assertNotIn("HG_NUM_THREADS", largest.stderr)

        for invalid in ("0", "-2", "+2", "two", "1.5", "1_0", "1 0", "\u0661", str(2 ** 63), str(2 ** 64)):
            result = _run_with_environment(invalid, code)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(int(result.stdout), default)
            self.assertIn("HG_NUM_THREADS", result.stderr)


if __name__ == '__main__':
    unittest.main()
