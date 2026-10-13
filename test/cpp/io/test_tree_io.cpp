/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "../test_utils.hpp"
#include "higra/io/tree_io.hpp"

namespace tree_io {

    using namespace hg;
    using namespace std;


    TEST_CASE("read and save tree", "[tree_io]") {
            array_1d<int> parent{ 5, 5, 6, 6, 6, 7, 7, 7 };

            array_1d<double> attr1{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
            array_1d<int> attr2{ 8, 7, 6, 5, 4, 3, 2, 1 };
            tree t(parent);
            ostringstream out;
            save_tree(out, t).add_attribute("attr1", attr1).add_attribute("attr2", attr2).finalize();
            string res = out.str();

            istringstream in(res);
            auto tree_attr = read_tree(in);
            auto t2 = tree_attr.first;
            auto attributes = tree_attr.second;

            REQUIRE(xt::allclose(parent, parents(t2)));

            REQUIRE(attributes.count("attr1") == 1);
            REQUIRE(xt::allclose(attributes["attr1"], attr1));

            REQUIRE(attributes.count("attr2") == 1);
            REQUIRE(xt::allclose(attributes["attr2"], attr2));
    }

    TEST_CASE("tree input failures and legacy compatibility", "[tree_io]") {
        const string header = "VERSION=1\nNBNODES=3\nNBATTR=0\nEND\n";
        int p[] = {2, 2, 2};
        const string payload(reinterpret_cast<char *>(p), sizeof(p));
        for (const auto &bad: vector<string>{"", "VERSION=1", "NBNODES=3\nEND\n",
                "NBATTR=0\nEND\n",
                "VERSION=x\n", "VERSION=2\n", "NBNODES=3x\n",
                "CATEGORY=x\n", "CATEGORY=-1\n", "CATEGORY=2\n", "CATEGORY\n",
                "NBNODES=-1\n", "NBNODES\n", "NBATTR=x\n", "NBATTR=-1\n",
                "NBNODES=9999999999999999999999\n",
                "NBNODES=2147483647\nNBATTR=0\nEND\n", header,
                header + payload.substr(0, payload.size() - 1),
                "NBNODES=3\nNBATTR=1\nEND\n" + payload + "NAME=a\n",
                "NBNODES=3\nNBATTR=1\nEND\n" + payload + "END\n",
                "NBNODES=3\nNBATTR=1\nEND\n" + payload + "NAME=a\nEND\n" + string(3 * sizeof(double) - 1, 'x'),
                "NBNODES=3\nNBATTR=0\nEND!" + payload,
                "NBNODES=3\nNBATTR=0\nEND\rX" + payload}) {
            istringstream in(bad);
            REQUIRE_THROWS(read_tree(in));
        }
        for (const auto &h: vector<string>{header, "NBNODES=3\nNBATTR=0\nEND\n",
                "VERSION=1\r\nNBNODES=3\r\nNBATTR=0\r\nEND\r\n",
                "UNKNOWN=42\nNBNODES=1\nNBNODES=3\nNBATTR=0\nEND\n"}) {
            istringstream in(h + payload);
            auto result = read_tree(in);
            REQUIRE(xt::all(xt::equal(result.first.parents(), array_1d<int>{2, 2, 2})));
            REQUIRE(result.second.empty());
            REQUIRE(result.first.category() == tree_category::partition_tree);
        }
        {
            istringstream in(header + payload.substr(0, 1));
            in.exceptions(ios::badbit | ios::failbit);
            REQUIRE_THROWS(read_tree(in));
        }
        {
            double a[] = {1, 2, 3};
            double b[] = {4, 5, 6};
            istringstream in("NBNODES=3\nNBATTR=2\nEND\n" + payload +
                            "NAME=a\nEND\n" + string(reinterpret_cast<char *>(a), sizeof(a)) +
                            "NAME=a\nEND\n" + string(reinterpret_cast<char *>(b), sizeof(b)));
            auto result = read_tree(in);
            REQUIRE(xt::allclose(result.second.at("a"), array_1d<double>{4, 5, 6}));
        }
        int bad_p[] = {10, 2, 2};
        istringstream bad_topology(header + string(reinterpret_cast<char *>(bad_p), sizeof(bad_p)));
        REQUIRE_THROWS(read_tree(bad_topology));
        REQUIRE(tree_io_internal::legacy_parent(std::numeric_limits<int>::max()) == std::numeric_limits<int>::max());
        REQUIRE_THROWS(tree_io_internal::legacy_parent(-1));
        if (std::numeric_limits<index_t>::max() > std::numeric_limits<int>::max())
            REQUIRE_THROWS(tree_io_internal::legacy_parent(static_cast<index_t>(std::numeric_limits<int>::max()) + 1));
        REQUIRE_THROWS(tree_io_internal::byte_count(std::numeric_limits<std::size_t>::max(), 8));
        REQUIRE_THROWS(tree_io_internal::byte_count(static_cast<std::size_t>(std::numeric_limits<streamsize>::max()), 2));
    }

    struct failing_buffer : stringbuf {
        bool fail_write = false;
        streamsize fail_bytes = -1;
        bool fail_seek = false;
        bool fail_flush = false;
        bool fail_tell = false;
        streamsize xsputn(const char *s, streamsize n) override {
            return fail_write || n == fail_bytes ? 0 : stringbuf::xsputn(s, n);
        }
        int_type overflow(int_type c) override {
            return fail_write ? traits_type::eof() : stringbuf::overflow(c);
        }
        pos_type seekpos(pos_type p, ios_base::openmode mode) override {
            return fail_seek ? pos_type(off_type(-1)) : stringbuf::seekpos(p, mode);
        }
        pos_type seekoff(off_type off, ios_base::seekdir dir, ios_base::openmode mode) override {
            return fail_tell ? pos_type(off_type(-1)) : stringbuf::seekoff(off, dir, mode);
        }
        int sync() override { return fail_flush ? -1 : 0; }
    };

    TEST_CASE("tree output failures and finalization", "[tree_io]") {
        tree t(array_1d<int>{2, 2, 2});
        array_1d<int> a{1, 2, 3};
        for (int failure = 0; failure < 5; ++failure) {
            failing_buffer buffer;
            ostream out(&buffer);
            auto saver = save_tree(out, t);
            if (failure == 0) buffer.fail_write = true;
            if (failure == 1) buffer.fail_seek = true;
            if (failure == 2) buffer.fail_flush = true;
            if (failure == 3) out.setstate(ios::badbit);
            if (failure == 4) {
                buffer.fail_seek = true;
                out.exceptions(ios::badbit | ios::failbit);
            }
            REQUIRE_THROWS(saver.finalize());
            // Destructor must not throw, including streams with exception masks.
        }
        {
            failing_buffer buffer;
            ostream out(&buffer);
            buffer.fail_bytes = 3 * sizeof(int);
            REQUIRE_THROWS(save_tree(out, t));
        }
        {
            failing_buffer buffer;
            ostream out(&buffer);
            auto saver = save_tree(out, t);
            buffer.fail_bytes = 3 * sizeof(double);
            REQUIRE_THROWS(saver.add_attribute("a", a));
        }
        for (int failure = 0; failure < 3; ++failure) {
            failing_buffer buffer;
            ostream out(&buffer);
            buffer.fail_write = failure == 0;
            buffer.fail_tell = failure == 1;
            buffer.fail_flush = failure == 2;
            REQUIRE_THROWS(save_tree(out, t));
        }
        {
            failing_buffer buffer;
            ostream out(&buffer);
            auto saver = save_tree(out, t);
            buffer.fail_write = true;
            REQUIRE_THROWS(saver.add_attribute("a", a));
        }
        {
            failing_buffer buffer;
            ostream out(&buffer);
            REQUIRE_THROWS_AS(([&]() {
                auto saver = save_tree(out, t);
                buffer.fail_seek = true;
                out.exceptions(ios::badbit | ios::failbit);
                throw runtime_error("original");
            })(), runtime_error);
        }
        ostringstream out;
        {
            auto saver = save_tree(out, t);
            saver.add_attribute("a", a);
        }
        istringstream in(out.str());
        REQUIRE(read_tree(in).second.count("a") == 1);
        ostringstream explicit_out;
        auto saver = save_tree(explicit_out, t);
        saver.finalize();
        const auto bytes = explicit_out.str();
        REQUIRE_NOTHROW(saver.finalize());
        REQUIRE(explicit_out.str() == bytes);
        REQUIRE_THROWS(saver.add_attribute("a", a));
    }

    TEST_CASE("tree payload spans read chunks", "[tree_io]") {
        auto p = array_1d<int>::from_shape({4100});
        p.fill(4099);
        tree t(p);
        auto a = array_1d<double>::from_shape({4100});
        a.fill(12.5);
        ostringstream out;
        save_tree(out, t).add_attribute("a", a).finalize();
        istringstream in(out.str());
        auto result = read_tree(in);
        REQUIRE(xt::all(xt::equal(result.first.parents(), p)));
        REQUIRE(xt::allclose(result.second.at("a"), a));
    }

    TEST_CASE("empty tree and category I/O", "[tree_io]") {
        for (auto category: {tree_category::partition_tree, tree_category::component_tree}) {
            for (bool with_attribute: {false, true}) {
                for (auto p: {array_1d<index_t>::from_shape({0}), array_1d<index_t>{2, 2, 2}}) {
                    tree t(p, category);
                    auto a = array_1d<double>::from_shape({p.size()});
                    a.fill(4.5);
                    ostringstream out;
                    auto saver = save_tree(out, t);
                    if (with_attribute)
                        saver.add_attribute("a", a);
                    saver.finalize();
                    istringstream in(out.str());
                    auto result = read_tree(in);
                    REQUIRE(result.first.category() == category);
                    REQUIRE(result.first.num_vertices() == t.num_vertices());
                    REQUIRE(result.first.num_leaves() == t.num_leaves());
                    REQUIRE(result.first.root() == t.root());
                    REQUIRE(result.first.parents().size() == p.size());
                    for (std::size_t i = 0; i < p.size(); ++i)
                        REQUIRE(result.first.parents()(i) == p(i));
                    REQUIRE(result.second.size() == (with_attribute ? 1 : 0));
                    if (with_attribute) {
                        REQUIRE(result.second.at("a").size() == a.size());
                        for (std::size_t i = 0; i < a.size(); ++i)
                            REQUIRE(result.second.at("a")(i) == a(i));
                    }
                }
            }
        }
        istringstream legacy_empty("NBNODES=0\nNBATTR=0\nEND\n");
        REQUIRE(read_tree(legacy_empty).first.category() == tree_category::partition_tree);
        istringstream duplicate("NBNODES=0\nNBATTR=0\nCATEGORY=1\nCATEGORY=0\nEND\n");
        REQUIRE(read_tree(duplicate).first.category() == tree_category::component_tree);
    }
}