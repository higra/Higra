/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <boost/test/unit_test.hpp>

#include "higra/structure/fibonacci_heap.hpp"
#include "../test_utils.hpp"
#include <random>
#include <map>
#include <algorithm>

BOOST_AUTO_TEST_SUITE(fibonacci_heap_tests);

    using namespace hg;
    using namespace std;

    template<typename T>
    struct node {

        auto get_value() {
            return m_value;
        }

        node() {

        }

        node(T value, index_t index) : m_value(value), m_index(index) {

        }

        T m_value;
        index_t m_index;
    };

    template<typename T>
    struct trivial_heap {
        using value_handle = node<T>;

        bool empty() const {
            return elements.empty();
        }

        value_handle push(T value) {
            elements.emplace_back(value, m_counter++);
            return elements[elements.size() - 1];
        }

        void merge(trivial_heap &other) {
            for (auto n: other.elements)
                elements.push_back(n);
            other.elements.clear();
        }

        value_handle top() {
            return elements[imin()];
        }

        void pop() {
            if (elements.size() > 0)
                elements.erase(elements.begin() + imin());
        }

        void erase(value_handle &node) {
            auto i = find(node.m_index);
            elements.erase(elements.begin() + i);
        }

        void increase(value_handle &node, const T &value) {
            update(node, value);
        }

        void decrease(value_handle &node, const T &value) {
            update(node, value);
        }

        void update(value_handle &node, const T &value) {
            auto i = find(node.m_index);
            node.m_value = value;
            elements[i].m_value = value;
        }

        void increase(value_handle node) {
            update(node);
        }

        void decrease(value_handle node) {
            update(node);
        }

        void update(value_handle node) {
            auto i = find(node.m_index);
            elements[i].m_value = node.m_value;
        }

        void clear() {
            elements.clear();
        }

        auto size() const {
            return elements.size();
        }

    private:
        vector<node<T>> elements;

        index_t imin() {
            T minv = elements[0].get_value();
            index_t imini = 0;
            for (index_t i = 1; i < (index_t) elements.size(); i++) {
                if (elements[i].get_value() < minv) {
                    imini = i;
                    minv = elements[i].get_value();
                }
            }
            return imini;
        }

        index_t find(index_t index) {
            for (index_t i = 0; i < (index_t)elements.size(); i++)
                if (elements[i].m_index == index)
                    return i;
            throw std::runtime_error("cannot find element");
        }

        index_t m_counter = 0;
    };

    BOOST_AUTO_TEST_CASE(test_pool_one_block) {
        fibonacci_heap_internal::object_pool<hg::index_t> pool;
        hg::index_t *i1 = pool.allocate();

        hg::index_t *i2 = pool.allocate();
        BOOST_CHECK((i2 - i1) == 1);
        hg::index_t *i3 = pool.allocate();
        BOOST_CHECK((i3 - i1) == 2);
        hg::index_t *i4 = pool.allocate();
        BOOST_CHECK((i4 - i1) == 3);

        pool.free(i3);

        hg::index_t *i5 = pool.allocate();
        BOOST_CHECK((i5 - i1) == 2);
        hg::index_t *i6 = pool.allocate();
        BOOST_CHECK((i6 - i1) == 4);

        pool.free(i5);
        pool.free(i4);

        hg::index_t *i7 = pool.allocate();
        BOOST_CHECK((i7 - i1) == 3);
        hg::index_t *i8 = pool.allocate();
        BOOST_CHECK((i8 - i1) == 2);
        hg::index_t *i9 = pool.allocate();
        BOOST_CHECK(i9 - i1 == 5);
        hg::index_t *i10 = pool.allocate();
        BOOST_CHECK((i10 - i1) == 6);
    }

    BOOST_AUTO_TEST_CASE(test_pool_several_blocks) {
        fibonacci_heap_internal::object_pool<hg::index_t> pool(3);
        hg::index_t *i1 = pool.allocate();
        hg::index_t *i2 = pool.allocate();
        BOOST_CHECK(i2 - i1 == 1);
        hg::index_t *i3 = pool.allocate();
        BOOST_CHECK(i3 - i1 == 2);

        hg::index_t *i4 = pool.allocate();
        hg::index_t *i5 = pool.allocate();
        BOOST_CHECK(i5 - i4 == 1);
        hg::index_t *i6 = pool.allocate();
        BOOST_CHECK(i6 - i4 == 2);

        hg::index_t *i7 = pool.allocate();
        hg::index_t *i8 = pool.allocate();
        BOOST_CHECK(i8 - i7 == 1);

        pool.free(i6);
        pool.free(i2);
        pool.free(i4);

        hg::index_t *i9 = pool.allocate();
        BOOST_CHECK(i9 - i4 == 0);
        hg::index_t *i10 = pool.allocate();
        BOOST_CHECK(i10 - i1 == 1);
        hg::index_t *i11 = pool.allocate();
        BOOST_CHECK(i11 - i4 == 2);

        hg::index_t *i12 = pool.allocate();
        BOOST_CHECK(i12 - i7 == 2);

        hg::index_t *i13 = pool.allocate();
        hg::index_t *i14 = pool.allocate();
        BOOST_CHECK(i14 - i13 == 1);
    }

    BOOST_AUTO_TEST_CASE(test_fibonacci_heap1) {
        fibonacci_heap<hg::index_t> heap;
        heap.push(10);
        BOOST_CHECK(heap.size() == 1);
        BOOST_CHECK(!heap.empty());
        BOOST_CHECK(heap.top()->get_value() == 10);
        heap.push(15);
        BOOST_CHECK(heap.size() == 2);
        BOOST_CHECK(heap.top()->get_value() == 10);
        heap.push(8);
        BOOST_CHECK(heap.size() == 3);
        BOOST_CHECK(heap.top()->get_value() == 8);

        heap.clear();
        BOOST_CHECK(heap.size() == 0);
        BOOST_CHECK(heap.empty());
    }

    BOOST_AUTO_TEST_CASE(test_fibonacci_heap2) {
        fibonacci_heap<hg::index_t> heap;
        heap.push(10);
        heap.pop();
        BOOST_CHECK(heap.size() == 0);
        BOOST_CHECK(heap.empty());

        heap.push(10);
        heap.push(15);
        heap.push(8);
        heap.push(22);
        heap.push(17);

        BOOST_CHECK(heap.top()->get_value() == 8);
        heap.pop();

        heap.push(5);
        heap.push(19);
        heap.push(2);


        BOOST_CHECK(heap.top()->get_value() == 2);
        heap.pop();
        BOOST_CHECK(heap.top()->get_value() == 5);
        heap.pop();
        BOOST_CHECK(heap.top()->get_value() == 10);
        heap.pop();
        BOOST_CHECK(heap.top()->get_value() == 15);
        heap.pop();
        BOOST_CHECK(heap.top()->get_value() == 17);
        heap.pop();
        BOOST_CHECK(heap.top()->get_value() == 19);
        heap.pop();
        BOOST_CHECK(heap.top()->get_value() == 22);
        heap.pop();
        BOOST_CHECK(heap.size() == 0);
    }


    BOOST_AUTO_TEST_CASE(test_fibonacci_heap3) {
        fibonacci_heap<hg::index_t> heap;
        heap.push(10);
        heap.push(15);
        heap.push(8);

        fibonacci_heap<hg::index_t> heap2;
        heap.push(9);
        heap.push(7);

        heap.merge(heap2);
        BOOST_CHECK(heap.top()->get_value() == 7);
        heap.pop();
        BOOST_CHECK(heap.top()->get_value() == 8);
        heap.pop();
        BOOST_CHECK(heap.top()->get_value() == 9);
        heap.pop();
        BOOST_CHECK(heap.top()->get_value() == 10);
        heap.pop();
        BOOST_CHECK(heap.top()->get_value() == 15);
        heap.pop();
        BOOST_CHECK(heap.size() == 0);

    }

    BOOST_AUTO_TEST_CASE(test_fibonacci_heap_decrease_key) {
        fibonacci_heap<hg::index_t> heap;
        heap.push(10);
        heap.pop();
        BOOST_CHECK(heap.size() == 0);
        BOOST_CHECK(heap.empty());

        heap.push(10);
        heap.push(15);
        heap.push(8);
        auto e1 = heap.push(22);
        auto e2 = heap.push(17);

        BOOST_CHECK(heap.top()->get_value() == 8);
        heap.pop();

        heap.push(5);
        heap.push(19);
        heap.push(2);

        heap.decrease(e2, 12);
        heap.decrease(e1, 3);

        BOOST_CHECK(heap.top()->get_value() == 2);
        heap.pop();
        BOOST_CHECK(heap.top()->get_value() == 3);
        heap.pop();
        BOOST_CHECK(heap.top()->get_value() == 5);
        heap.pop();
        BOOST_CHECK(heap.top()->get_value() == 10);
        heap.pop();
        BOOST_CHECK(heap.top()->get_value() == 12);
        heap.pop();
        BOOST_CHECK(heap.top()->get_value() == 15);
        heap.pop();
        BOOST_CHECK(heap.top()->get_value() == 19);
        heap.pop();
    }

    BOOST_AUTO_TEST_CASE(test_fibonacci_heap_erase_key) {
        fibonacci_heap<hg::index_t> heap;
        heap.push(10);
        heap.pop();
        BOOST_CHECK(heap.size() == 0);
        BOOST_CHECK(heap.empty());

        heap.push(10);
        heap.push(15);
        heap.push(8);
        auto e1 = heap.push(22);
        auto e2 = heap.push(17);

        BOOST_CHECK(heap.top()->get_value() == 8);
        heap.pop();

        heap.push(5);
        heap.push(19);
        heap.push(2);

        heap.erase(e2);
        heap.erase(e1);

        BOOST_CHECK(heap.top()->get_value() == 2);
        heap.pop();
        BOOST_CHECK(heap.top()->get_value() == 5);
        heap.pop();
        BOOST_CHECK(heap.top()->get_value() == 10);
        heap.pop();
        BOOST_CHECK(heap.top()->get_value() == 15);
        heap.pop();
        BOOST_CHECK(heap.top()->get_value() == 19);
        heap.pop();
    }

    BOOST_AUTO_TEST_CASE(test_fibonacci_heap_increase_key) {
        fibonacci_heap<hg::index_t> heap;
        heap.push(10);
        heap.pop();
        BOOST_CHECK(heap.size() == 0);
        BOOST_CHECK(heap.empty());

        heap.push(10);
        heap.push(15);
        heap.push(8);
        auto e1 = heap.push(22);
        auto e2 = heap.push(17);

        BOOST_CHECK(heap.top()->get_value() == 8);
        heap.pop();

        heap.push(5);
        heap.push(19);
        heap.push(2);

        heap.increase(e2, 25);
        heap.increase(e1, 23);

        BOOST_CHECK(heap.top()->get_value() == 2);
        heap.pop();
        BOOST_CHECK(heap.top()->get_value() == 5);
        heap.pop();
        BOOST_CHECK(heap.top()->get_value() == 10);
        heap.pop();
        BOOST_CHECK(heap.top()->get_value() == 15);
        heap.pop();
        BOOST_CHECK(heap.top()->get_value() == 19);
        heap.pop();
        BOOST_CHECK(heap.top()->get_value() == 23);
        heap.pop();
        BOOST_CHECK(heap.top()->get_value() == 25);
        heap.pop();
    }

    struct rnd {
        static auto &get_rng() {
            static std::mt19937 rng(150000); //std::random_device()()
            return rng;
        }

    };

    auto random_heaps(int nbop) {
        std::uniform_int_distribution<std::mt19937::result_type> dist100(1, 100);
        std::uniform_int_distribution<std::mt19937::result_type> weights(1, 100000);

        auto &rng = rnd::get_rng();


        fibonacci_heap<hg::index_t> heap;
        trivial_heap<hg::index_t> theap;

        for (int i = 0; i < nbop; i++) {
            int op = dist100(rng);
            if (op < 80) {
                auto w = weights(rng);
                heap.push(w);
                theap.push(w);
            } else {
                if (theap.size() > 0) {
                    BOOST_CHECK(heap.top()->get_value() == theap.top().get_value());
                    BOOST_CHECK(heap.size() == theap.size());
                    heap.pop();
                    theap.pop();
                    if (theap.size() > 0) {
                        BOOST_CHECK(heap.top()->get_value() == theap.top().get_value());
                    }
                    BOOST_CHECK(heap.size() == theap.size());
                }
            }

        }
        return std::make_pair(std::move(heap), std::move(theap));
    }


    BOOST_AUTO_TEST_CASE(test_fibonacci_heap_stress_test_push_pop) {
        random_heaps(1000);
        random_heaps(1000);
        random_heaps(1000);
    }

    BOOST_AUTO_TEST_CASE(test_fibonacci_heap_stress_test_push_pop_merge) {

        std::uniform_int_distribution<std::mt19937::result_type> dist100(1, 100);
        std::uniform_int_distribution<std::mt19937::result_type> weights(1, 100000);

        auto &rng = rnd::get_rng();
        int nbop = 500;

        fibonacci_heap<hg::index_t> heap;
        trivial_heap<hg::index_t> theap;

        for (int i = 0; i < nbop; i++) {
            int op = dist100(rng);
            if (op < 60) {
                auto w = weights(rng);
                heap.push(w);
                theap.push(w);
            } else if (op < 80) {
                if (theap.size() > 0) {
                    auto fht = heap.top();
                    auto tht = theap.top();

                    BOOST_CHECK(fht->get_value() == tht.get_value());
                    BOOST_CHECK(heap.size() == theap.size());
                    heap.pop();
                    theap.pop();
                    if (theap.size() > 0) {
                        BOOST_CHECK(heap.top()->get_value() == theap.top().get_value());
                    }
                    BOOST_CHECK(heap.size() == theap.size());
                }
            } else {
                auto hh = random_heaps(100);
                heap.merge(hh.first);
                theap.merge(hh.second);
            }
        }
    }

    BOOST_AUTO_TEST_CASE(test_fibonacci_heap_stress_test_push_pop_update_erase) {

        std::uniform_int_distribution<std::mt19937::result_type> dist100(1, 100);
        std::uniform_int_distribution<std::mt19937::result_type> weights(1, 100000);

        auto &rng = rnd::get_rng();
        int nbop = 20000;

        fibonacci_heap<hg::index_t> heap;
        trivial_heap<hg::index_t> theap;

        map<fibonacci_heap<hg::index_t>::value_handle, trivial_heap<hg::index_t>::value_handle> heapset;

        for (int i = 0; i < nbop; i++) {
            int op = dist100(rng);
            if (op < 60) {
                auto w = weights(rng);
                heapset[heap.push(w)] = theap.push(w);
            } else if (op < 80) {
                if (theap.size() > 0) {
                    auto fht = heap.top();
                    auto tht = theap.top();

                    BOOST_CHECK(fht->get_value() == tht.get_value());
                    BOOST_CHECK(heap.size() == theap.size());
                    heap.pop();
                    theap.pop();
                    heapset.erase(fht);
                    if (theap.size() > 0) {
                        BOOST_CHECK(heap.top()->get_value() == theap.top().get_value());
                    }
                    BOOST_CHECK(heap.size() == theap.size());
                }

            } else if (op < 90) {
                if (heap.size() > 0) {
                    std::uniform_int_distribution<std::mt19937::result_type> dist(0, heapset.size() - 1);
                    auto nb = dist(rng);

                    auto it = heapset.begin();
                    std::advance(it, nb);
                    auto e = *it;

                    auto fhe = e.first;
                    auto the = e.second;

                    heap.erase(fhe);
                    theap.erase(the);
                    heapset.erase(fhe);
                }
            } else {
                if (heap.size() > 0) {


                    std::uniform_int_distribution<std::mt19937::result_type> dist(0, heapset.size() - 1);
                    auto nb = dist(rng);

                    auto it = heapset.begin();
                    std::advance(it, nb);
                    auto e = *it;

                    auto fhe = e.first;
                    auto the = e.second;

                    if (fhe->get_value() != 1) {
                        auto w = weights(rng);
                        heap.update(fhe, w);
                        theap.update(the, w);
                    }

                }
            }
        }
    }

BOOST_AUTO_TEST_SUITE_END();