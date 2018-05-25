//
// Created by perretb on 25/05/18.
//

#pragma once

namespace hg {
    struct trace {
        constexpr static bool enabled = true;
    };
}

#define HG_LOG_LEVEL_ERROR (1)
#define HG_LOG_LEVEL_WARNING (2)
#define HG_LOG_LEVEL_INFO (3)
#define HG_LOG_LEVEL_DEBUG (4)
#define HG_LOG_LEVEL_DETAIL (5)

#define HG_LOG_LEVEL_ERROR_NAME "ERROR"
#define HG_LOG_LEVEL_WARNING_NAME "WARNING"
#define HG_LOG_LEVEL_INFO_NAME "INFO"
#define HG_LOG_LEVEL_DEBUG_NAME "DEBUG"
#define HG_LOG_LEVEL_DETAIL_NAME "DETAIL"


#ifndef HG_LOG_LEVEL
#define HG_LOG_LEVEL HG_LOG_LEVEL_WARNING
#endif

#define HG_LOG_EMIT(LEVEL, M, ...)   fprintf(stdout, "[" LEVEL "] %s (%s:%d) " M, __func__, __FILE__, __LINE__, ##__VA_ARGS__);

#if HG_LOG_LEVEL <= HG_LOG_LEVEL_ERROR
#define HG_LOG_ERROR(M, ...) HG_LOG_EMIT(HG_LOG_LEVEL_ERROR_NAME, M, ##__VA_ARGS__)
#else
#define HG_LOG_ERROR(...) do{}while(0)
#endif

#if HG_LOG_LEVEL <= HG_LOG_LEVEL_WARNING
#define HG_LOG_WARNING(M, ...) HG_LOG_EMIT(HG_LOG_LEVEL_WARNING_NAME, M, ##__VA_ARGS__)
#else
#define HG_LOG_WARNING(...) do{}while(0)
#endif

#if HG_LOG_LEVEL <= HG_LOG_LEVEL_INFO
#define HG_LOG_INFO(M, ...) HG_LOG_EMIT(HG_LOG_LEVEL_INFO_NAME, M, ##__VA_ARGS__)
#else
#define HG_LOG_INFO(...) do{}while(0)
#endif

#if HG_LOG_LEVEL <= HG_LOG_LEVEL_DEBUG
#define HG_LOG_DEBUG(M, ...) HG_LOG_EMIT(HG_LOG_LEVEL_DEBUG_NAME, M, ##__VA_ARGS__)
#else
#define HG_LOG_DEBUG(...) do{}while(0)
#endif

#if HG_LOG_LEVEL <= HG_LOG_LEVEL_DETAIL
#define HG_LOG_DETAIL(M, ...) HG_LOG_EMIT(HG_LOG_LEVEL_DETAIL_NAME, M, ##__VA_ARGS__)
#else
#define HG_LOG_DETAIL(...) do{}while(0)
#endif

#define HG_TRACE do{                                \
if(hg::trace::enabled){                             \
    HG_LOG_EMIT("TRACE", "function called");        \
}                                                   \
}while(0)