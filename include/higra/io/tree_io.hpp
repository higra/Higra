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
#include "xtensor/xexpression.hpp"
#include <istream>
#include <ostream>
#include <map>

namespace hg {

#define HG_TREE_IO_VERSION "1"

#define HG_TREE_IO_VERSION_KEY "VERSION"
#define HG_TREE_IO_NBNODES_KEY "NBNODES"
#define HG_TREE_IO_NBATTRIBUTES_KEY "NBATTR"
#define HG_TREE_IO_HEADEREND_KEY "END"
#define HG_TREE_IO_NAME_KEY "NAME"

    //bool saveBPT(char * path, int nbnodes, int * parents, int numAttr, double ** attrs, char ** attrNames);
    //bool readBPT(char * path, int * nbnodes, int ** parents, int * numAttr, double *** attrs, char *** attrNames);


    namespace tree_io_internal {


        struct tree_saver_helper {

            using out_type = std::ostream &;

            tree_saver_helper(out_type out, const tree &t) : m_tree(t), m_out(out) {
                init();
            }

            ~tree_saver_helper() {
                finalize();
            }

            template<typename T>
            tree_saver_helper &add_attribute(const std::string &name, const xt::xexpression<T> &xarray) {
                auto &array = xarray.derived_cast();
                hg_assert(array.dimension() == 1, "Only scalar attributes are supported.");
                hg_assert(array.size() == m_tree.num_vertices(), "Attribute size does not match the size of the tree.");

                // force conversion to double
                // TODO avoid copy when not needed
                xt::xarray<double> a = array;

                m_num_attr++;

                m_out << HG_TREE_IO_NAME_KEY << "=" << name << std::endl;
                m_out << HG_TREE_IO_HEADEREND_KEY << std::endl;

                m_out.write(reinterpret_cast<const char *>(a.data()), std::streamsize(array.size() * sizeof(double)));

                return *this;
            }

            void finalize() {
                if (!finalized) {
                    m_out.seekp(m_nb_attr_position);
                    m_out << m_num_attr;
                    finalized = true;
                }
            }


        private:
            void init() {
                m_out << HG_TREE_IO_VERSION_KEY << "=" << HG_TREE_IO_VERSION << std::endl;
                m_out << HG_TREE_IO_NBNODES_KEY << "=" << m_tree.num_vertices() << std::endl;
                m_out << HG_TREE_IO_NBATTRIBUTES_KEY << "=";
                m_nb_attr_position = m_out.tellp();
                // this will be filled later ....
                m_out << "                             " << std::endl;
                m_out << HG_TREE_IO_HEADEREND_KEY << std::endl;

                std::vector<int> p;
                p.reserve(num_vertices(m_tree));
                for (auto v: parents(m_tree))
                    p.push_back(v);

                m_out.write(reinterpret_cast<const char *>(&p[0]), std::streamsize(p.size() * sizeof(int)));
            }


            const tree &m_tree;
            out_type m_out;
            index_t m_nb_attr_position = -1;
            index_t m_num_attr = 0;
            bool finalized = false;
        };

    }

    inline
    auto
    save_tree(std::ostream &out, const tree &t) {
        return tree_io_internal::tree_saver_helper(out, t);
    }


    inline
    auto
    read_tree(std::istream &in) {
        std::string key = "";
        char dummy;
        std::string tmp;

        int num_vertices = -1;
        int num_attributes = -1;

        while (key != HG_TREE_IO_HEADEREND_KEY) {
            int value;
            in >> tmp;
            std::size_t pos = tmp.find('=');
            if (pos != std::string::npos) {
                key = tmp.substr(0, pos);
                value = std::stoi(tmp.substr(pos + 1));
            } else {
                key = tmp;
            }

            if (key == HG_TREE_IO_VERSION_KEY) {

            } else if (key == HG_TREE_IO_NBNODES_KEY) {
                num_vertices = value;
            } else if (key == HG_TREE_IO_NBATTRIBUTES_KEY) {
                num_attributes = value;
            } else if (key == HG_TREE_IO_HEADEREND_KEY) {

            } else {
                HG_LOG_WARNING("Key '%s' is unknown and will be ignored.", key.c_str());
            }

        }

        hg_assert(num_vertices > 0, "Incorrect or missing key "
                HG_TREE_IO_NBNODES_KEY);
        hg_assert(num_attributes >= 0, "Incorrect or missing key "
                HG_TREE_IO_NBATTRIBUTES_KEY);

        in.read(&dummy, 1); // consumme the last \n left by cin...

        array_1d<int> parents;
        parents.resize({std::size_t(num_vertices)});
        in.read(reinterpret_cast<char *>(parents.data()), std::streamsize(num_vertices * sizeof(int)));

        std::map<std::string, array_1d<double>> attributes;

        for (int i = 0; i < num_attributes; i++) {
            std::string name = "";
            key = "";
            while (key != HG_TREE_IO_HEADEREND_KEY) {
                in >> tmp;
                std::size_t pos = tmp.find('=');
                std::string value;
                if (pos != std::string::npos) {
                    key = tmp.substr(0, pos);
                    value = tmp.substr(pos + 1);
                } else {
                    key = tmp;
                }

                if (key == HG_TREE_IO_NAME_KEY) {
                    name = value;
                } else if (key == HG_TREE_IO_HEADEREND_KEY) {

                } else {
                    HG_LOG_WARNING("Key '%s' is unknown and will be ignored.", key.c_str());
                }
            }
            hg_assert(name != "", "Incorrect or missing key for attribute " + std::to_string(i) + " " +
                                  HG_TREE_IO_NAME_KEY);
            in.read(&dummy, 1); // consumme the last \n left by cin...

            attributes.emplace(name, array_1d<double>::from_shape({std::size_t(num_vertices)}));
            in.read(reinterpret_cast<char *>(attributes[name].data()), std::streamsize(num_vertices * sizeof(double)));
        }

        return std::make_pair(tree(parents), std::move(attributes));
    }

}


