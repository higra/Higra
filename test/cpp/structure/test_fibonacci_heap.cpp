/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "higra/structure/fibonacci_heap.hpp"
#include "../test_utils.hpp"
#include <random>
#include <map>
#include <algorithm>

namespace test_fibonacci_heap {

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
            for (index_t i = 0; i < (index_t) elements.size(); i++)
                if (elements[i].m_index == index)
                    return i;
            throw std::runtime_error("cannot find element");
        }

        index_t m_counter = 0;
    };

    TEST_CASE("memory pool 1 block", "[fibonacci_heap]") {
        fibonacci_heap_internal::object_pool<hg::index_t> pool;
        hg::index_t *i1 = pool.allocate();

        hg::index_t *i2 = pool.allocate();
        REQUIRE((i2 - i1) == 1);
        hg::index_t *i3 = pool.allocate();
        REQUIRE((i3 - i1) == 2);
        hg::index_t *i4 = pool.allocate();
        REQUIRE((i4 - i1) == 3);

        pool.free(i3);

        hg::index_t *i5 = pool.allocate();
        REQUIRE((i5 - i1) == 2);
        hg::index_t *i6 = pool.allocate();
        REQUIRE((i6 - i1) == 4);

        pool.free(i5);
        pool.free(i4);

        hg::index_t *i7 = pool.allocate();
        REQUIRE((i7 - i1) == 3);
        hg::index_t *i8 = pool.allocate();
        REQUIRE((i8 - i1) == 2);
        hg::index_t *i9 = pool.allocate();
        REQUIRE(i9 - i1 == 5);
        hg::index_t *i10 = pool.allocate();
        REQUIRE((i10 - i1) == 6);
    }

    TEST_CASE("memory pool several blocks", "[fibonacci_heap]") {
        fibonacci_heap_internal::object_pool<hg::index_t> pool(3);
        hg::index_t *i1 = pool.allocate();
        hg::index_t *i2 = pool.allocate();
        REQUIRE(i2 - i1 == 1);
        hg::index_t *i3 = pool.allocate();
        REQUIRE(i3 - i1 == 2);

        hg::index_t *i4 = pool.allocate();
        hg::index_t *i5 = pool.allocate();
        REQUIRE(i5 - i4 == 1);
        hg::index_t *i6 = pool.allocate();
        REQUIRE(i6 - i4 == 2);

        hg::index_t *i7 = pool.allocate();
        hg::index_t *i8 = pool.allocate();
        REQUIRE(i8 - i7 == 1);

        pool.free(i6);
        pool.free(i2);
        pool.free(i4);

        hg::index_t *i9 = pool.allocate();
        REQUIRE(i9 - i4 == 0);
        hg::index_t *i10 = pool.allocate();
        REQUIRE(i10 - i1 == 1);
        hg::index_t *i11 = pool.allocate();
        REQUIRE(i11 - i4 == 2);

        hg::index_t *i12 = pool.allocate();
        REQUIRE(i12 - i7 == 2);

        hg::index_t *i13 = pool.allocate();
        hg::index_t *i14 = pool.allocate();
        REQUIRE(i14 - i13 == 1);
    }

    TEST_CASE("fibonacci heap push-top-size-empty", "[fibonacci_heap]") {
        fibonacci_heap<hg::index_t> heap;
        heap.push(10);
        REQUIRE(heap.size() == 1);
        REQUIRE(!heap.empty());
        REQUIRE(heap.top()->get_value() == 10);
        heap.push(15);
        REQUIRE(heap.size() == 2);
        REQUIRE(heap.top()->get_value() == 10);
        heap.push(8);
        REQUIRE(heap.size() == 3);
        REQUIRE(heap.top()->get_value() == 8);

        heap.clear();
        REQUIRE(heap.size() == 0);
        REQUIRE(heap.empty());
    }

    TEST_CASE("fibonacci heap push-top-size-empty 2", "[fibonacci_heap]") {
        fibonacci_heap<hg::index_t> heap;
        heap.push(10);
        heap.pop();
        REQUIRE(heap.size() == 0);
        REQUIRE(heap.empty());

        heap.push(10);
        heap.push(15);
        heap.push(8);
        heap.push(22);
        heap.push(17);

        REQUIRE(heap.top()->get_value() == 8);
        heap.pop();

        heap.push(5);
        heap.push(19);
        heap.push(2);


        REQUIRE(heap.top()->get_value() == 2);
        heap.pop();
        REQUIRE(heap.top()->get_value() == 5);
        heap.pop();
        REQUIRE(heap.top()->get_value() == 10);
        heap.pop();
        REQUIRE(heap.top()->get_value() == 15);
        heap.pop();
        REQUIRE(heap.top()->get_value() == 17);
        heap.pop();
        REQUIRE(heap.top()->get_value() == 19);
        heap.pop();
        REQUIRE(heap.top()->get_value() == 22);
        heap.pop();
        REQUIRE(heap.size() == 0);
    }


    TEST_CASE("fibonacci heap push-top-size-empty 3", "[fibonacci_heap]") {
        fibonacci_heap<hg::index_t> heap;
        heap.push(10);
        heap.push(15);
        heap.push(8);

        fibonacci_heap<hg::index_t> heap2;
        heap.push(9);
        heap.push(7);

        heap.merge(heap2);
        REQUIRE(heap.top()->get_value() == 7);
        heap.pop();
        REQUIRE(heap.top()->get_value() == 8);
        heap.pop();
        REQUIRE(heap.top()->get_value() == 9);
        heap.pop();
        REQUIRE(heap.top()->get_value() == 10);
        heap.pop();
        REQUIRE(heap.top()->get_value() == 15);
        heap.pop();
        REQUIRE(heap.size() == 0);

    }

    TEST_CASE("fibonacci heap decrease key", "[fibonacci_heap]") {
        fibonacci_heap<hg::index_t> heap;
        heap.push(10);
        heap.pop();
        REQUIRE(heap.size() == 0);
        REQUIRE(heap.empty());

        heap.push(10);
        heap.push(15);
        heap.push(8);
        auto e1 = heap.push(22);
        auto e2 = heap.push(17);

        REQUIRE(heap.top()->get_value() == 8);
        heap.pop();

        heap.push(5);
        heap.push(19);
        heap.push(2);

        heap.decrease(e2, 12);
        heap.decrease(e1, 3);

        REQUIRE(heap.top()->get_value() == 2);
        heap.pop();
        REQUIRE(heap.top()->get_value() == 3);
        heap.pop();
        REQUIRE(heap.top()->get_value() == 5);
        heap.pop();
        REQUIRE(heap.top()->get_value() == 10);
        heap.pop();
        REQUIRE(heap.top()->get_value() == 12);
        heap.pop();
        REQUIRE(heap.top()->get_value() == 15);
        heap.pop();
        REQUIRE(heap.top()->get_value() == 19);
        heap.pop();
    }

    TEST_CASE("fibonacci heap erase key", "[fibonacci_heap]") {
        fibonacci_heap<hg::index_t> heap;
        heap.push(10);
        heap.pop();
        REQUIRE(heap.size() == 0);
        REQUIRE(heap.empty());

        heap.push(10);
        heap.push(15);
        heap.push(8);
        auto e1 = heap.push(22);
        auto e2 = heap.push(17);

        REQUIRE(heap.top()->get_value() == 8);
        heap.pop();

        heap.push(5);
        heap.push(19);
        heap.push(2);

        heap.erase(e2);
        heap.erase(e1);

        REQUIRE(heap.top()->get_value() == 2);
        heap.pop();
        REQUIRE(heap.top()->get_value() == 5);
        heap.pop();
        REQUIRE(heap.top()->get_value() == 10);
        heap.pop();
        REQUIRE(heap.top()->get_value() == 15);
        heap.pop();
        REQUIRE(heap.top()->get_value() == 19);
        heap.pop();
    }

    TEST_CASE("fibonacci heap increase key", "[fibonacci_heap]") {
        fibonacci_heap<hg::index_t> heap;
        heap.push(10);
        heap.pop();
        REQUIRE(heap.size() == 0);
        REQUIRE(heap.empty());

        heap.push(10);
        heap.push(15);
        heap.push(8);
        auto e1 = heap.push(22);
        auto e2 = heap.push(17);

        REQUIRE(heap.top()->get_value() == 8);
        heap.pop();

        heap.push(5);
        heap.push(19);
        heap.push(2);

        heap.increase(e2, 25);
        heap.increase(e1, 23);

        REQUIRE(heap.top()->get_value() == 2);
        heap.pop();
        REQUIRE(heap.top()->get_value() == 5);
        heap.pop();
        REQUIRE(heap.top()->get_value() == 10);
        heap.pop();
        REQUIRE(heap.top()->get_value() == 15);
        heap.pop();
        REQUIRE(heap.top()->get_value() == 19);
        heap.pop();
        REQUIRE(heap.top()->get_value() == 23);
        heap.pop();
        REQUIRE(heap.top()->get_value() == 25);
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
                    REQUIRE(heap.top()->get_value() == theap.top().get_value());
                    REQUIRE(heap.size() == theap.size());
                    heap.pop();
                    theap.pop();
                    if (theap.size() > 0) {
                        REQUIRE(heap.top()->get_value() == theap.top().get_value());
                    }
                    REQUIRE(heap.size() == theap.size());
                }
            }

        }
        return std::make_pair(std::move(heap), std::move(theap));
    }


    TEST_CASE("fibonacci heap randomized stress test push-pop", "[fibonacci_heap]") {
        random_heaps(1000);
        random_heaps(1000);
        random_heaps(1000);
    }

    TEST_CASE("fibonacci heap randomized stress test push-pop-merge", "[fibonacci_heap]") {

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

                    REQUIRE(fht->get_value() == tht.get_value());
                    REQUIRE(heap.size() == theap.size());
                    heap.pop();
                    theap.pop();
                    if (theap.size() > 0) {
                        REQUIRE(heap.top()->get_value() == theap.top().get_value());
                    }
                    REQUIRE(heap.size() == theap.size());
                }
            } else {
                auto hh = random_heaps(100);
                heap.merge(hh.first);
                theap.merge(hh.second);
            }
        }
    }

    TEST_CASE("fibonacci heap randomized stress test push-pop-update-erase", "[fibonacci_heap]") {
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

                    REQUIRE(fht->get_value() == tht.get_value());
                    REQUIRE(heap.size() == theap.size());
                    heap.pop();
                    theap.pop();
                    heapset.erase(fht);
                    if (theap.size() > 0) {
                        REQUIRE(heap.top()->get_value() == theap.top().get_value());
                    }
                    REQUIRE(heap.size() == theap.size());
                }

            } else if (op < 90) {
                if (heap.size() > 0) {
                    std::uniform_int_distribution<std::mt19937::result_type> dist(0, (index_t)heapset.size() - 1);
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
}