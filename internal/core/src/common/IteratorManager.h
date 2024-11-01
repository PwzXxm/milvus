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

#include "common/BitsetView.h"
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
class IteratorManager {
 public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    static IteratorManager&
    GetInstance() {
        static IteratorManager instance;
        return instance;
    }

    // Call this before using the IteratorManager
    // Call it only once, this is NOT thread-safe
    void
    Init(const int32_t max_iterator_num, const int64_t default_ttl, const int64_t clean_up_interval) {
        capacity_ = max_iterator_num;
        default_ttl_ = std::chrono::seconds(default_ttl);
        clean_up_interval_ = std::chrono::seconds(clean_up_interval);
        LOG_INFO(
            "IteratorManager initialized with max_iterator_num: {} and "
            "default_ttl: {} seconds and clean_up_interval: {} seconds",
            capacity_,
            default_ttl_.count(),
            clean_up_interval_.count());
        
        // clean_up_thread_ = std::thread(&IteratorManager::CleanUpExpiredIterators, this);
    }


    // TDOO: Not thread-safe
    query::CachedSearchIterator*
    Get(const std::string& token) {
        // std::lock_guard<std::mutex> lock(mutex_);

        if(auto it = iterators_.find(token); it != iterators_.end()) {
            return &it->second;
        }
        return nullptr;
    }

    void Put(const std::string& token, query::CachedSearchIterator&& iterator, std::optional<std::chrono::seconds> ttl = std::nullopt) {
        std::lock_guard<std::mutex> lock(mutex_);

        // TODO: hack here
        const std::string& cur_uuid = token.substr(0, 36);
        if (cur_uuid != last_token) {
            last_token = cur_uuid;
            iterators_.clear();
        }

        iterators_.insert({token, std::move(iterator)});
        // lru_list_.push_back({token, Clock::now() + ttl.value_or(default_ttl_)});
    }

    IteratorManager(const IteratorManager&) = delete;
    IteratorManager& operator=(const IteratorManager&) = delete;
    IteratorManager(IteratorManager&&) = delete;
    IteratorManager& operator=(IteratorManager&&) = delete;
 private:
    struct LruInfo {
        std::string token;
        Clock::time_point expiration_time;
    };

    bool stop_clean_up_ = false;
    int32_t capacity_ = 0;
    std::chrono::seconds default_ttl_ = std::chrono::seconds(0);
    std::chrono::seconds clean_up_interval_ = std::chrono::seconds(0);
    std::mutex mutex_;
    std::unordered_map<std::string, query::CachedSearchIterator> iterators_;
    std::list<LruInfo> lru_list_;
    // std::thread clean_up_thread_;

    std::string last_token = "";

    IteratorManager() = default;
    ~IteratorManager() {
        // stop_clean_up_ = true;
        // LOG_INFO("IteratorManager is being destroyed, stopping clean up thread");
        // if (clean_up_thread_.joinable()) {
        //     clean_up_thread_.join();
        // }
        // LOG_INFO("IteratorManager is destroyed");
    }

    void CleanUpExpiredIterators() {
        while (!stop_clean_up_) {
            std::this_thread::sleep_for(clean_up_interval_);

            std::lock_guard<std::mutex> lock(mutex_);
            for (auto it = lru_list_.begin(); it != lru_list_.end();) {
                if (it->expiration_time < Clock::now()) {
                    LOG_DEBUG("IteratorManager is cleaning up expired iterator for token: {}", it->token);
                    iterators_.erase(it->token);
                    it = lru_list_.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }
};
}  // namespace milvus