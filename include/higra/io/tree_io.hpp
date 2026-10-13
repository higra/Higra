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
#include "xtensor/core/xexpression.hpp"
#include <istream>
#include <ostream>
#include <map>
#include <limits>
#include <algorithm>
#include <cstdint>

namespace hg {

#define HG_TREE_IO_VERSION "1"

#define HG_TREE_IO_VERSION_KEY "VERSION"
#define HG_TREE_IO_NBNODES_KEY "NBNODES"
#define HG_TREE_IO_NBATTRIBUTES_KEY "NBATTR"
#define HG_TREE_IO_HEADEREND_KEY "END"
#define HG_TREE_IO_NAME_KEY "NAME"
#define HG_TREE_IO_CATEGORY_KEY "CATEGORY"

    namespace tree_io_internal {

        inline std::streamsize byte_count(std::size_t count, std::size_t item_size) {
            hg_assert(item_size > 0 && count <= std::numeric_limits<std::size_t>::max() / item_size,
                      "Tree payload byte count overflow.");
            const auto bytes = count * item_size;
            hg_assert(bytes <= static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max()),
                      "Tree payload exceeds stream byte count range.");
            return static_cast<std::streamsize>(bytes);
        }

        inline int legacy_parent(index_t parent) {
            hg_assert(parent >= 0 && parent <= std::numeric_limits<int>::max(),
                      "Tree parent is not representable in version 1.");
            return static_cast<int>(parent);
        }

        inline int parse_count(const std::string &value, const std::string &key) {
            std::size_t end = 0;
            long long count;
            try {
                count = std::stoll(value, &end);
            } catch (const std::exception &) {
                throw std::runtime_error("Invalid tree header value for " + key + ".");
            }
            hg_assert(end == value.size() && count >= 0 && count <= std::numeric_limits<int>::max(),
                      "Invalid tree header value for " + key + ".");
            return static_cast<int>(count);
        }

        inline std::pair<std::string, std::string> read_field(std::istream &in) {
            std::string token;
            hg_assert(static_cast<bool>(in >> token), "Incomplete tree header (expected field or END).");
            auto pos = token.find('=');
            if (pos == std::string::npos)
                return {token, ""};
            return {token.substr(0, pos), token.substr(pos + 1)};
        }

        inline void read_delimiter(std::istream &in) {
            char c;
            hg_assert(static_cast<bool>(in.get(c)), "Missing tree header newline.");
            // The END line may have trailing spaces; never consume payload whitespace.
            while (c == ' ' || c == '\t')
                hg_assert(static_cast<bool>(in.get(c)), "Missing tree header newline.");
            if (c == '\r')
                hg_assert(static_cast<bool>(in.get(c)), "Missing tree header newline.");
            hg_assert(c == '\n', "Invalid tree header delimiter (expected newline).");
        }

        template<typename T>
        array_1d<T> read_payload(std::istream &in, std::size_t count, const std::string &context) {
            byte_count(count, sizeof(T));
            // Read before allocating the full array: a corrupt count in a short file
            // must not cause a huge allocation.
            std::vector<T> values;
            T chunk[4096];
            while (values.size() < count) {
                auto n = std::min(count - values.size(), std::size_t(4096));
                auto bytes = byte_count(n, sizeof(T));
                in.read(reinterpret_cast<char *>(chunk), bytes);
                hg_assert(in && in.gcount() == bytes, "Incomplete tree " + context + " payload.");
                values.insert(values.end(), chunk, chunk + n);
            }
            auto result = array_1d<T>::from_shape({count});
            std::copy(values.begin(), values.end(), result.begin());
            return result;
        }

        struct tree_saver_helper {
            using out_type = std::ostream &;

            tree_saver_helper(out_type out, const tree &t) : m_tree(t), m_out(out) {
                init();
            }

            // RAII is best effort. Call finalize explicitly to observe output errors.
            ~tree_saver_helper() noexcept {
                try {
                    finalize();
                } catch (...) {
                }
            }

            template<typename T>
            tree_saver_helper &add_attribute(const std::string &name, const xt::xexpression<T> &xarray) {
                hg_assert(!finalized, "Cannot add an attribute after tree finalization.");
                auto &array = xarray.derived_cast();
                hg_assert(array.dimension() == 1, "Only scalar attributes are supported.");
                hg_assert(array.size() == m_tree.num_vertices(), "Attribute size does not match the size of the tree.");
                hg_assert(m_num_attr < std::numeric_limits<int>::max(), "Too many tree attributes.");
                auto bytes = byte_count(array.size(), sizeof(double));
                xt::xarray<double> a = array;
                m_out << HG_TREE_IO_NAME_KEY << "=" << name << std::endl;
                m_out << HG_TREE_IO_HEADEREND_KEY << std::endl;
                hg_assert(m_out, "Failed to write tree attribute header.");
                if (bytes > 0)
                    m_out.write(reinterpret_cast<const char *>(a.data()), bytes);
                hg_assert(m_out, "Failed to write tree attribute payload.");
                ++m_num_attr;
                return *this;
            }

            void finalize() {
                if (!finalized) {
                    hg_assert(m_out, "Failed tree output before finalization.");
                    m_out.seekp(m_nb_attr_position);
                    hg_assert(m_out, "Failed to seek tree attribute count.");
                    m_out << m_num_attr;
                    hg_assert(m_out, "Failed to update tree attribute count.");
                    m_out.flush();
                    hg_assert(m_out, "Failed to flush tree output.");
                    finalized = true;
                }
            }

        private:
            void init() {
                hg_assert(m_tree.num_vertices() <= std::numeric_limits<int>::max(),
                          "Tree node count is not representable in version 1.");
                auto bytes = byte_count(static_cast<std::size_t>(m_tree.num_vertices()), sizeof(int));
                std::vector<int> p;
                p.reserve(num_vertices(m_tree));
                for (auto v: parents(m_tree)) {
                    p.push_back(legacy_parent(v));
                }
                hg_assert(m_tree.category() == tree_category::component_tree ||
                          m_tree.category() == tree_category::partition_tree, "Invalid tree category.");
                m_out << HG_TREE_IO_VERSION_KEY << "=" << HG_TREE_IO_VERSION << std::endl;
                m_out << HG_TREE_IO_CATEGORY_KEY << "="
                      << (m_tree.category() == tree_category::component_tree ? 0 : 1) << std::endl;
                m_out << HG_TREE_IO_NBNODES_KEY << "=" << m_tree.num_vertices() << std::endl;
                m_out << HG_TREE_IO_NBATTRIBUTES_KEY << "=";
                hg_assert(m_out, "Failed to write tree header.");
                m_nb_attr_position = m_out.tellp();
                hg_assert(m_out && m_nb_attr_position != std::streampos(-1), "Failed to locate tree attribute count.");
                m_out << "                             " << std::endl;
                m_out << HG_TREE_IO_HEADEREND_KEY << std::endl;
                hg_assert(m_out, "Failed to write tree header.");
                if (bytes > 0)
                    m_out.write(reinterpret_cast<const char *>(p.data()), bytes);
                hg_assert(m_out, "Failed to write tree parent payload.");
            }

            const tree &m_tree;
            out_type m_out;
            std::streampos m_nb_attr_position = std::streampos(-1);
            int m_num_attr = 0;
            bool finalized = false;
        };
    }

    /** Save version 1 native int parents/double scalar attributes, including empty
     * trees and CATEGORY (0: component tree, 1: partition tree). The stream must
     * support seeking. Call finalize() to check count update and flushing; the
     * destructor performs non-throwing best-effort finalization. */
    inline auto save_tree(std::ostream &out, const tree &t) {
        return tree_io_internal::tree_saver_helper(out, t);
    }

    /** Read version 1 (or historical version-less) mixed text/binary data.
     * Missing CATEGORY defaults to partition tree. Incomplete headers/payloads throw. */
    inline auto read_tree(std::istream &in) {
        using namespace tree_io_internal;
        int num_vertices = -1;
        int num_attributes = -1;
        auto category = tree_category::partition_tree;
        while (true) {
            auto field = read_field(in);
            const auto &key = field.first;
            if (key == HG_TREE_IO_HEADEREND_KEY)
                break;
            if (key == HG_TREE_IO_VERSION_KEY) {
                hg_assert(parse_count(field.second, key) == 1, "Unsupported tree file version.");
            } else if (key == HG_TREE_IO_CATEGORY_KEY) {
                const auto value = parse_count(field.second, key);
                hg_assert(value <= 1, "Invalid tree CATEGORY (expected 0 or 1).");
                category = value == 0 ? tree_category::component_tree : tree_category::partition_tree;
            } else if (key == HG_TREE_IO_NBNODES_KEY) {
                num_vertices = parse_count(field.second, key);
            } else if (key == HG_TREE_IO_NBATTRIBUTES_KEY) {
                num_attributes = parse_count(field.second, key);
            } else {
                HG_LOG_WARNING("Key '%s' is unknown and will be ignored.", key.c_str());
            }
        }
        hg_assert(num_vertices >= 0, "Incorrect or missing key " HG_TREE_IO_NBNODES_KEY);
        hg_assert(num_attributes >= 0, "Incorrect or missing key " HG_TREE_IO_NBATTRIBUTES_KEY);
        read_delimiter(in);
        auto parents = read_payload<int>(in, static_cast<std::size_t>(num_vertices), "parent");
        tree t(parents, category);
        std::map<std::string, array_1d<double>> attributes;
        for (int i = 0; i < num_attributes; ++i) {
            std::string name;
            while (true) {
                auto field = read_field(in);
                if (field.first == HG_TREE_IO_HEADEREND_KEY)
                    break;
                if (field.first == HG_TREE_IO_NAME_KEY) {
                    name = field.second;
                } else {
                    HG_LOG_WARNING("Key '%s' is unknown and will be ignored.", field.first.c_str());
                }
            }
            hg_assert(!name.empty(), "Incorrect or missing NAME for tree attribute " + std::to_string(i));
            read_delimiter(in);
            auto values = read_payload<double>(in, static_cast<std::size_t>(num_vertices), "attribute " + name);
            // Preserve the legacy last-payload-wins behavior for duplicate names.
            attributes[name] = std::move(values);
        }
        return std::make_pair(std::move(t), std::move(attributes));
    }
}
