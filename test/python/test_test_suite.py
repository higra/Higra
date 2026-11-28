############################################################################
# Copyright ESIEE Paris (2026)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

import ast
import os
import unittest


def find_duplicate_definitions(source, filename="<string>"):
    """
    Find functions that are defined twice in the same scope of a Python source: at module level or in
    the body of a class. The second definition silently replaces the first one, so a test defined first
    never runs.

    :param source: Python source code
    :param filename: file name used in syntax error messages
    :return: a list of ``(scope, name, first_line, second_line)`` tuples, where ``scope`` is ``"<module>"``
             or the class name
    """
    tree = ast.parse(source, filename)
    scopes = [("<module>", tree.body)]
    scopes += [(node.name, node.body) for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

    duplicates = []
    for scope_name, body in scopes:
        seen = {}
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in seen:
                    duplicates.append((scope_name, node.name, seen[node.name], node.lineno))
                seen[node.name] = node.lineno
    return duplicates


class TestTestSuite(unittest.TestCase):

    def test_find_duplicate_definitions(self):
        source = ("def f():\n"
                  "    pass\n"
                  "class A:\n"
                  "    def test_a(self):\n"
                  "        pass\n"
                  "    def test_b(self):\n"
                  "        pass\n"
                  "    def test_a(self):\n"
                  "        pass\n"
                  "def f():\n"
                  "    pass\n")
        self.assertEqual(find_duplicate_definitions(source),
                         [("<module>", "f", 1, 10), ("A", "test_a", 4, 8)])
        self.assertEqual(find_duplicate_definitions("class A:\n    def test_a(self):\n        pass\n"), [])

    def test_no_duplicate_test_definitions(self):
        root = os.path.dirname(os.path.abspath(__file__))
        problems = []
        num_files = 0
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            for filename in sorted(filenames):
                if filename.startswith("test_") and filename.endswith(".py"):
                    path = os.path.join(dirpath, filename)
                    num_files += 1
                    with open(path, encoding="utf-8") as f:
                        source = f.read()
                    for scope, name, first, second in find_duplicate_definitions(source, path):
                        problems.append("%s: %s.%s is defined at lines %d and %d"
                                        % (os.path.relpath(path, root), scope, name, first, second))

        self.assertGreater(num_files, 1)
        self.assertEqual(problems, [], "Duplicated definitions hide tests:\n" + "\n".join(problems))


if __name__ == '__main__':
    unittest.main()
