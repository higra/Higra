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

#ifdef HG_USE_TBB

#include "tbb/parallel_sort.h"
#include "tbb-ssort/parallel_stable_sort.h"

#else

#include <algorithm>

#endif

namespace hg {
    
    template<typename RandomAccessIterator, typename Compare>
    void stable_sort(RandomAccessIterator xs, RandomAccessIterator xe, Compare comp) {
#ifdef HG_USE_TBB
        pss::parallel_stable_sort(xs,  xe, comp);
#else
        std::stable_sort(xs, xe, comp);
#endif
    }


    template<typename RandomAccessIterator, typename Compare>
    void sort(RandomAccessIterator xs, RandomAccessIterator xe, Compare comp) {
#ifdef HG_USE_TBB
        tbb::parallel_sort(xs,  xe, comp);
#else
        std::sort(xs, xe, comp);
#endif
    }


    template<typename RandomAccessIterator>
    void stable_sort(RandomAccessIterator xs, RandomAccessIterator xe) {
        typedef typename std::iterator_traits<RandomAccessIterator>::value_type T;
        stable_sort(xs, xe, std::less<T>());
    }

    template<typename RandomAccessIterator>
    void sort(RandomAccessIterator xs, RandomAccessIterator xe) {
        typedef typename std::iterator_traits<RandomAccessIterator>::value_type T;
        sort(xs, xe, std::less<T>());
    }
}