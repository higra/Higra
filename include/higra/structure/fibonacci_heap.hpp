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

#include <vector>
#include <stack>
#include <array>
#include "../utils.hpp"
#include <cstring>

namespace hg {

    namespace fibonacci_heap_internal {

        /**
         * A simple object pool, only support allocating objects one at a time.
         *
         * Warning : not thread safe
         *
         * @tparam T Type of objects in the pool
         */
        template<typename T>
        struct object_pool {


            union object {
                T t;
                object *next;

                object() {
                    next = nullptr;
                }

                ~object() {

                }
            };


            void free(T *element) {
                //memset(element, 0, sizeof(T));
                object *new_first = (object *) element;
                new_first->next = first_free;
                first_free = new_first;
            }

            T *allocate() {
                if (first_free == nullptr) {
                    pool.emplace_back(m_blocksize);
                    auto &block = pool[pool.size() - 1];
                    first_free = &block[0];
                    for (index_t i = 0; i < (index_t) block.size() - 1; i++)
                        block[i].next = first_free + i + 1;
                    block[block.size() - 1].next = nullptr;
                }

                object *tmp = first_free;
                first_free = first_free->next;
                return (T *) tmp;
            }

            object_pool(size_t blocksize = 4096) : m_blocksize(blocksize) {

            }

        private:
            std::vector<std::vector<object>> pool;
            size_t m_blocksize;
            object *first_free = nullptr;
        };


        template<typename T>
        struct fibonacci_heap;

        template<typename T>
        struct node {
            using self_t = node<T>;
            using pointer_t = self_t *;
        private:
            pointer_t m_previous;
            pointer_t m_next;
            pointer_t m_child;
            pointer_t m_parent;

            T m_value;

            int m_degree;
            bool m_marked;

            node() {}

            void init(T value) {
                m_value = value;
                m_previous = this;
                m_next = this;
                m_child = nullptr;
                m_parent = nullptr;
                m_degree = 0;
                m_marked = false;
            }

        public:
            friend struct fibonacci_heap<T>;

            T get_value() { return m_value; }
        };


        /**
         * Fibonacci Heap
         *
         * Warning different heaps will share the same object pool which is NOT thread safe !
         * The following method are not thread safe: push, pop, erase, clear, destructor
         *
         * @tparam T Value type, must implement operator < (ie. with a and b two values of type T, a < b must be a well formed expression)
         */
        template<typename T>
        struct fibonacci_heap final {
            using node_t = node<T>;
            using value_handle = node_t *;

        private:

            static object_pool<node_t> &s_pool() {
                static object_pool<node_t> pool{};
                return pool;
            }

            node_t *m_heap = nullptr;
            size_t m_size = 0;

        public:

            /**
             * Creates an empty min-heap
             */
            fibonacci_heap() {

            }

            /**
             * Deletes current heap
             */
            ~fibonacci_heap() {
                clear();
            }

            fibonacci_heap(const fibonacci_heap &) = delete;

            fibonacci_heap &operator=(const fibonacci_heap &) = delete;

            fibonacci_heap(fibonacci_heap &&other) {
                m_heap = other.m_heap;
                m_size = other.m_size;
                other.m_heap = nullptr;
                other.m_size = 0;
            }

            fibonacci_heap &operator=(fibonacci_heap &&other) {
                m_heap = other.m_heap;
                m_size = other.m_size;
                other.m_heap = nullptr;
                other.m_size = 0;
            }

            /**
             * Test if heap is empty
             * @return
             */
            bool empty() const {
                return m_heap == nullptr;
            }

            /**
             * Insert new value in  the heap
             *
             * Complexity O(1)
             *
             * @param value
             * @return Handle on the new value (used for increase/decrease/update/erase operations)
             */
            value_handle push(T value) {
                m_size++;
                node_t *new_node = s_pool().allocate();
                new_node->init(value);
                m_heap = m_merge(m_heap, new_node);
                return new_node;
            }

            /**
             * Merge the provided heap into the current heap.
             * The provided heap is empty after the operation.
             *
             * Complexity O(1)
             *
             * @param other
             */
            void merge(fibonacci_heap &other) {
                m_heap = m_merge(m_heap, other.m_heap);
                m_size += other.size();
                other.m_size = 0;
                other.m_heap = nullptr;
            }

            /**
             * Returns an handle on the min element of the heap
             *
             * Complexity O(1)
             *
             * @return Handle on the value (used for increase/decrease/update/erase operations)
             */
            value_handle top() {
                return m_heap;
            }

            /**
             * Removes the min element from the heap (this invalidates any previously obtained handles on this element)
             */
            void pop() {
                auto old_heap = m_heap;
                m_extract_min();
                s_pool().free(old_heap);
            }

            /**
             * Removes the given element from the heap.
             *
             * Complexity amortized O(log(n))
             *
             * @param node
             */
            void erase(value_handle node) {
                m_delete_key(node);
                s_pool().free(node);
            }


            /**
             * Decreases the value of the given element to the given value.
             *
             * Complexity amortized O(1)
             *
             * @param node
             * @param value
             */
            void decrease(value_handle node, const T &value) {
                node->m_value = value;
                decrease(node);
            }

            /**
             * Updates the heap after the value of the given element has been decreased
             *
             * Complexity amortized O(1)
             *
             * @param node
             */
            void decrease(value_handle node) {
                m_decrease_key(node);
            }

            /**
             * Increases the value of the given element to the given value.
             *
             * Complexity amortized O(log(n))
             *
             * @param node
             * @param value
             */
            void increase(value_handle node, const T &value) {
                node->m_value = value;
                increase(node);
            }

            /**
             * Updates the heap after the value of the given element has been increased
             *
             * Complexity amortized O(log(n))
             *
             * @param node
             */
            void increase(value_handle node) {
                m_increase_key(node);
            }

            /**
             * Changes the value of the given element to the given value.
             *
             * Complexity amortized O(log(n))
             *
             * @param node
             * @param value
             */
            void update(value_handle node, const T &value) {
                if (value < node->m_value)
                    decrease(node, value);
                else if (node->m_value < value) {
                    increase(node, value);
                }
            }

            /**
             * Empties the heap
             *
             * Complexity O(n)
             */
            void clear() {
                if (m_heap == nullptr)
                    return;

                std::stack<node_t *> s;
                s.push(m_heap);
                for (node_t *tmp = m_heap->m_next; tmp != m_heap; tmp = tmp->m_next) {
                    s.push(tmp);
                }
                while (!s.empty()) {
                    node_t *n = s.top();
                    s.pop();
                    node_t *c = n->m_child;
                    if (c != nullptr) {
                        s.push(c);
                        for (node_t *tmp = c->m_next; tmp != c; tmp = tmp->m_next) {
                            s.push(tmp);
                        }
                    }
                    s_pool().free(n);
                }
                m_heap = nullptr;
                m_size = 0;
            }

            auto size() const {
                return m_size;
            }

        private:
            auto m_merge(node_t *root1, node_t *root2) {
                if (root1 == nullptr)
                    return root2;
                if (root2 == nullptr)
                    return root1;

                if (root1->m_value < root2->m_value) {
                    std::swap(root1, root2);
                }

                node_t *root1_previous = root1->m_previous;
                node_t *root2_previous = root2->m_previous;

                root2->m_previous = root1_previous;
                root1_previous->m_next = root2;

                root2_previous->m_next = root1;
                root1->m_previous = root2_previous;

                return root2;
            }

            void m_link_heap(node_t *y, node_t *x) {
                y->m_previous->m_next = y->m_next;
                y->m_next->m_previous = y->m_previous;

                y->m_parent = x;
                auto *child = x->m_child;
                if (child != nullptr) {
                    y->m_next = child->m_next;
                    y->m_previous = child;
                    child->m_next->m_previous = y;
                    child->m_next = y;
                } else {
                    y->m_next = y;
                    y->m_previous = y;
                    x->m_child = y;
                }
                x->m_degree++;
                y->m_marked = false;
            }

            void m_consolidate() {
                if (m_heap == m_heap->m_next)
                    return;

                std::array<node_t *, 64> A;
                std::fill(A.begin(), A.end(), nullptr);
                auto *last = m_heap->m_previous;
                auto *x = m_heap;
                while (true) {
                    auto d = x->m_degree;
                    auto *cur = x;
                    auto *next = x->m_next;
                    while (A[d] != nullptr) {
                        auto *y = A[d];
                        if (y->m_value < x->m_value) {
                            std::swap(x, y);
                        }
                        m_link_heap(y, x);
                        A[d] = nullptr;
                        d++;
                    }
                    A[d] = x;

                    if (cur == last)
                        break;
                    x = next;
                };

                node_t *minh = nullptr;
                int i = 0;
                while (minh == nullptr && i < (index_t) A.size()) {
                    if (A[i] != nullptr) {
                        minh = A[i];
                    }
                    i++;
                }

                auto *start = minh;
                auto *pos = minh->m_next;

                while (pos != start) {
                    if (pos->m_value < minh->m_value)
                        minh = pos;
                    pos = pos->m_next;
                }
                m_heap = minh;
            }

            void m_extract_min() {
                if (m_heap != nullptr) {
                    auto *child = m_heap->m_child;
                    if (child != nullptr) {
                        do {
                            child->m_parent = nullptr;
                            child = child->m_next;
                        } while (child != m_heap->m_child);
                        m_merge(m_heap, m_heap->m_child);
                        m_heap->m_child = nullptr;
                    }

                    auto *old_heap = m_heap;
                    if (old_heap->m_next == old_heap) {
                        m_heap = nullptr;
                    } else {
                        m_heap->m_previous->m_next = m_heap->m_next;
                        m_heap->m_next->m_previous = m_heap->m_previous;
                        m_heap = old_heap->m_next;
                        m_consolidate();
                    }

                    m_size--;
                }
            }

            void m_decrease_key(node_t *node) {
                auto parent = node->m_parent;
                if (parent != nullptr && node->m_value < parent->m_value) {
                    m_cut(node, parent);
                    m_cascading_cut(parent);
                }
                if (node->m_value < m_heap->m_value)
                    m_heap = node;
            }

            void m_cut(node_t *node, node_t *parent) {
                node->m_parent = nullptr;
                if (node->m_next == node) {
                    parent->m_child = nullptr;
                } else {
                    parent->m_child = node->m_next;
                    node->m_next->m_previous = node->m_previous;
                    node->m_previous->m_next = node->m_next;
                }
                parent->m_degree--;

                node->m_next = m_heap->m_next;
                m_heap->m_next->m_previous = node;

                m_heap->m_next = node;
                node->m_previous = m_heap;

                node->m_marked = false;
            }

            void m_cascading_cut(node_t *node) {
                auto parent = node->m_parent;
                if (parent != nullptr) {
                    if (!node->m_marked) {
                        node->m_marked = true;
                    } else {
                        m_cut(node, parent);
                        m_cascading_cut(parent);
                    }
                }
            }

            void m_delete_key(node_t *node) {
                auto parent = node->m_parent;
                if (parent != nullptr) {
                    m_cut(node, parent);
                    m_cascading_cut(parent);
                }

                auto *child = node->m_child;
                if (child != nullptr) {
                    do {
                        child->m_parent = nullptr;
                        child = child->m_next;
                    } while (child != node->m_child);
                    m_merge(m_heap, node->m_child);
                    node->m_child = nullptr;
                }

                if (node->m_next == node) {
                    m_heap = nullptr;
                } else {
                    node->m_next->m_previous = node->m_previous;
                    node->m_previous->m_next = node->m_next;
                    m_heap = node->m_next;
                    m_consolidate();
                }


                m_size--;
            }

            void m_increase_key(node_t *node) {
                m_delete_key(node);
                m_size++;
                node->init(node->m_value);
                m_heap = m_merge(m_heap, node);
            }

            /*
            int m_check_integrity(node_t * node, node_t * parent){
                if(node== nullptr)
                    return 0;
                int size = 0;
                node_t * start = node;

                node_t * position = node;
                do{
                    size++;
                    if(position->m_next->m_previous != position)
                    {
                        std::cout << "error";
                    }
                    if(position->m_previous->m_next != position)
                    {
                        std::cout << "error";
                    }
                    if(position->m_parent != parent)
                    {
                        std::cout << "error";
                    }
                    if(position->m_child != nullptr){
                        size+=m_check_integrity(position->m_child, position);
                    }
                    position = position->m_next;
                }while (position!=start);
                return size;
            };*/

        };
    }

    template<typename value_type>
    using fibonacci_heap = fibonacci_heap_internal::fibonacci_heap<value_type>;

}
