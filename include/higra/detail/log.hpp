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
#include <functional>
#include <iostream>
#include <string>

namespace hg {

    struct logger {

        static const std::size_t MAX_MSG_SIZE = 8096;
        using callback_list = std::vector<std::function<void(const std::string &)>>;

        static bool &trace_enabled() {
            static bool value{false};
            return value;
        }

        static callback_list &callbacks() {
            static callback_list callbacks{
                    [](const std::string &msg) { std::cout << msg; }}; //not the perfect initialization...
            return callbacks;
        }

        template<typename ...Args>
        static void emit(const char *format, Args &&...args) {
            char message[MAX_MSG_SIZE];
            snprintf(message, MAX_MSG_SIZE, format, std::forward<Args>(args)...);
            std::string sm(message);
            for (auto c: callbacks()) {
                c(sm);
            }

        }
    };

}
#if defined(__clang__)
#define HG_PRETTY_FUNCTION __PRETTY_FUNCTION__
#elif defined(__GNUC__) || defined(__GNUG__)
#define HG_PRETTY_FUNCTION __PRETTY_FUNCTION__
#elif defined(_MSC_VER)
#define HG_PRETTY_FUNCTION __FUNCSIG__
#else
#define HG_PRETTY_FUNCTION __func__
#endif


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

//#define HG_LOG_EMIT(LEVEL, M, ...)   fprintf(stdout, "[" LEVEL "] %s (%s:%d) " M "\n", HG_PRETTY_FUNCTION, __FILE__, __LINE__, ##__VA_ARGS__);
#define HG_LOG_EMIT(LEVEL, M, ...)   hg::logger::emit("[" LEVEL "] %s (%s:%d) " M "\n", HG_PRETTY_FUNCTION, __FILE__, __LINE__, ##__VA_ARGS__)

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

#ifdef HG_ENABLE_TRACE
#define HG_TRACE(M, ...) do{                                            \
if(hg::logger::trace_enabled()){                                        \
    HG_LOG_EMIT("TRACE", "function called " M, ##__VA_ARGS__);          \
}                                                                       \
}while(0)
#else
#define HG_TRACE(...)  do{}while(0)
#endif
