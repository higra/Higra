/*
  Copyright (C) 2014 Intel Corporation
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:

  * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.
  * Neither the name of Intel Corporation nor the names of its
    contributors may be used to endorse or promote products derived
    from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
  OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
  AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
  WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
  POSSIBILITY OF SUCH DAMAGE.
*/
#include <iterator>
#include <algorithm>
#include <memory>
#include <new>
#include <oneapi/tbb/parallel_invoke.h>

namespace pss {

    namespace internal {

    template<typename RandomAccessIterator1, typename RandomAccessIterator2, typename T, typename Compare>
    void merge_task(RandomAccessIterator1 xs, RandomAccessIterator1 xe,
              RandomAccessIterator2 ys, RandomAccessIterator2 ye,
          T *zs, Compare comp) {
      const auto merge_cut_off = typename std::iterator_traits<RandomAccessIterator1>::difference_type(2048);
      const auto size = (xe - xs) + (ye - ys);

      if (size <= merge_cut_off) {
        while (xs != xe && ys != ye) {
          if (comp(*ys, *xs)) {
            new (zs++) T(std::move(*ys++));
          } else {
            new (zs++) T(std::move(*xs++));
          }
        }
        while (xs != xe) {
          new (zs++) T(std::move(*xs++));
        }
        while (ys != ye) {
          new (zs++) T(std::move(*ys++));
        }
        return;
      }

      RandomAccessIterator1 xm;
      RandomAccessIterator2 ym;
      if (xe - xs >= ye - ys) {
        xm = xs + (xe - xs) / 2;
        ym = std::lower_bound(ys, ye, *xm, comp);
      } else {
        ym = ys + (ye - ys) / 2;
        xm = std::upper_bound(xs, xe, *ym, comp);
      }

      const auto zm = zs + (xm - xs) + (ym - ys);
      oneapi::tbb::parallel_invoke(
          [&] { merge_task(xs, xm, ys, ym, zs, comp); },
          [&] { merge_task(xm, xe, ym, ye, zm, comp); });
    }

        template<typename RandomAccessIterator, typename Compare>
        void stable_sort_task(RandomAccessIterator xs, RandomAccessIterator xe, Compare comp) {
            const auto sort_cut_off = typename std::iterator_traits<RandomAccessIterator>::difference_type(4096);
            const auto size = xe - xs;

            if (size <= sort_cut_off) {
                std::stable_sort(xs, xe, comp);
                return;
            }

            const auto xm = xs + size / 2;
            oneapi::tbb::parallel_invoke(
                    [&] { stable_sort_task(xs, xm, comp); },
                    [&] { stable_sort_task(xm, xe, comp); });

                typedef typename std::iterator_traits<RandomAccessIterator>::value_type value_type;
                std::allocator<value_type> allocator;
                value_type *buffer = allocator.allocate(size);
                merge_task(xs, xm, xm, xe, buffer, comp);
                std::move(buffer, buffer + size, xs);
                for (auto index = size; index != 0; --index) {
                  buffer[index - 1].~value_type();
                }
                allocator.deallocate(buffer, size);
        }

    } // namespace internal

    template<typename RandomAccessIterator, typename Compare>
    void parallel_stable_sort(RandomAccessIterator xs, RandomAccessIterator xe, Compare comp) {
        internal::stable_sort_task(xs, xe, comp);
    }

    template<typename RandomAccessIterator>
    void parallel_stable_sort(RandomAccessIterator xs, RandomAccessIterator xe) {
        typedef typename std::iterator_traits<RandomAccessIterator>::value_type T;
        parallel_stable_sort(xs, xe, std::less<T>());
    }

} // namespace pss
