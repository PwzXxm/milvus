// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <chrono>
#include <functional>
#include <mutex>
#include <string>
#include <optional>
#include <folly/concurrency/ConcurrentHashMap.h>
#include <folly/container/EvictingCacheMap.h>

#include "common/BitsetView.h"
#include "common/QueryInfo.h"
#include "common/Types.h"
#include "knowhere/index/index_node.h"
#include "log/Log.h"
#include "query/CachedSearchIterator.h"

namespace milvus {

// This is a singleton class manages V2 Search Iterators
// It is used to hold/get the Knowhere Iterator objects
// and limit the memory usage of the iterators.
// It also has the power to clean up the iterators that are not used for a long time.
//
// This class manages all iterators from different segment types
class SearchIteratorManager {
    using Mp = folly::ConcurrentHashMap<
        std::string,
        std::unique_ptr<query::CachedSearchIterator>>;
 public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;
    using ValueConstIter = Mp::ConstIterator;

    static SearchIteratorManager&
    GetInstance() {
        static SearchIteratorManager instance;
        return instance;
    }

    // Call this before using the SearchIteratorManager
    // Call it only once, this is NOT thread-safe
    // The capacity should be a power of two
    void
    Init(const int32_t capacity, const int64_t default_ttl, const int64_t clean_up_interval) {
        std::lock_guard<std::mutex> lock(mutex_);

        capacity_ = capacity;
        default_ttl_ = std::chrono::seconds(default_ttl);
        clean_up_interval_ = std::chrono::seconds(clean_up_interval);
        LOG_INFO(
            "SearchIteratorManager initialized with capacity: {} and "
            "default_ttl: {} seconds and clean_up_interval: {} seconds",
            capacity_,
            default_ttl_.count(),
            clean_up_interval_.count());
        iterators_.reserve(capacity_);
        lru_.setMaxSize(capacity_, [this](const std::string& key, const LruInfo& value) {
            iterators_.erase(key);
        });
        
        clean_up_thread_ = std::thread(&SearchIteratorManager::CleanUpExpiredIterators, this);
    }

    SearchIteratorManager(const SearchIteratorManager&) = delete;
    SearchIteratorManager& operator=(const SearchIteratorManager&) = delete;
    SearchIteratorManager(SearchIteratorManager&&) = delete;
    SearchIteratorManager& operator=(SearchIteratorManager&&) = delete;

    void
    NextBatchFrom(
        const SearchInfo& search_info,
        const std::function<std::unique_ptr<query::CachedSearchIterator>()>&
            create_iter_fn,
        SearchResult& search_result) {
        if (!search_info.iterator_v2_info_.has_value()) {
            PanicInfo(ErrorCode::UnexpectedError,
                      "The iterator_v2_info is not set.");
        }

        auto [iter_ptr, is_timed_out] =
            GetOrPut(search_info.iterator_v2_info_.value(), create_iter_fn);

        if (is_timed_out) {
            PanicInfo(ErrorCode::UnexpectedError,
                      "The search_iterator has been timed out. Try creating a new one.");
        }

        if (iter_ptr.has_value()) {
            auto cached_iter = create_iter_fn();
            cached_iter->NextBatch(search_info, search_result, true);
            return;
        }
        iter_ptr.value()->second->NextBatch(search_info, search_result, false);

        return;
    }

private:
    // This function tries to get the iterator from the cache,
    // if not found, it will create a new iterator and put it into the cache.
    // 
    // Returns:
    //    - (iter, false) the iterator found/created
    //    - (nullopt, false) if it currently full
    //    - (nullopt, true) it times out
   std::pair<std::optional<ValueConstIter>, bool /* is_timed_out */>
   GetOrPut(const SearchIteratorV2Info& search_info,
            const std::function<std::unique_ptr<query::CachedSearchIterator>()>&
                create_iter_fn) {
       try {
           const std::string& token = search_info.token;

           if (auto it = iterators_.find(token); it != iterators_.end()) {
               return UpdateLru(token, search_info.ttl)
                          ? std::make_pair(std::optional{std::move(it)}, false)
                          : std::make_pair(std::nullopt, true);
           }

           auto [it, inserted] = iterators_.insert({token, create_iter_fn()});
           if (!inserted) {
               return {std::nullopt, false};
           }

           return UpdateLru(token, search_info.ttl)
                      ? std::make_pair(std::optional{std::move(it)}, false)
                      : std::make_pair(std::nullopt, true);
       } catch (const std::bad_alloc&) {
           // exceeding max size
       }
       return {std::nullopt, false};
   }

    struct LruInfo {
        Clock::time_point expiration_time;
    };

    bool stop_clean_up_ = false;
    int32_t capacity_ = 0;
    std::chrono::seconds default_ttl_ = std::chrono::seconds(0);
    std::chrono::seconds clean_up_interval_ = std::chrono::seconds(0);
    std::mutex mutex_;
    folly::ConcurrentHashMap<std::string, std::unique_ptr<query::CachedSearchIterator>> iterators_;
    folly::EvictingCacheMap<std::string, LruInfo> lru_ =
        folly::EvictingCacheMap<std::string, LruInfo>(0);  // not thread-safe
    std::thread clean_up_thread_;

    std::string last_token = "";

    SearchIteratorManager() = default;
    ~SearchIteratorManager() {
        stop_clean_up_ = true;
        LOG_INFO("SearchIteratorManager is being destroyed, stopping clean up thread");
        if (clean_up_thread_.joinable()) {
            clean_up_thread_.join();
        }
        LOG_INFO("SearchIteratorManager is destroyed");
    }

    // return false if fails to Update LRU since the iterator is expired
    bool UpdateLru(const std::string& token, const int32_t ttl) {
        if (ttl == -1) {
            return true;
        }

        const std::chrono::seconds ttl_seconds = std::chrono::seconds(ttl);

        std::lock_guard<std::mutex> lock(mutex_);
        if(auto it = lru_.find(token); it != lru_.end()) {
            if (it->second.expiration_time >= Clock::now()) {
                return false;
            }
            it->second.expiration_time = Clock::now() + ttl_seconds;
        } else {
            lru_.insert(token, {Clock::now() + ttl_seconds});
        }
        return true;
    }

    void CleanUpExpiredIterators() {
        while (!stop_clean_up_) {
            std::this_thread::sleep_for(clean_up_interval_);

            for (auto it = lru_.begin(); it != lru_.end();) {
                if (it->second.expiration_time < Clock::now()) {
                    LOG_DEBUG("SearchIteratorManager is cleaning up expired iterator for token: {}", it->first);
                    iterators_.erase(it->first);
                } else {
                    ++it;
                }
            }
        }
    }
};
}  // namespace milvus