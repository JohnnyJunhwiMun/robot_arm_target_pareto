// ThreadPool.h
#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <atomic>

class ThreadPool {
public:
    // 생성자: num_threads개의 스레드를 생성
    ThreadPool(size_t num_threads);
    
    // 소멸자: 모든 스레드를 정리
    ~ThreadPool();
    
    // 작업을 스레드 풀에 제출
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;

private:
    // 작업 큐
    std::queue<std::function<void()>> tasks;
    
    // 스레드 벡터
    std::vector<std::thread> workers;
    
    // 동기화
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop;
};

// 생성자 구현
inline ThreadPool::ThreadPool(size_t num_threads) : stop(false) {
    for(size_t i = 0;i<num_threads;++i)
        workers.emplace_back(
            [this]
            {
                for(;;)
                {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, 
                            [this]{ return this->stop.load() || !this->tasks.empty(); });
                        if(this->stop.load() && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }

                    task();
                }
            }
        );
}

// 소멸자 구현
inline ThreadPool::~ThreadPool(){
    stop.store(true);
    condition.notify_all();
    for(std::thread &worker: workers)
        worker.join();
}

// 작업 제출 메소드 구현
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type>
{
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared< std::packaged_task<return_type()> >(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);

        // 스레드 풀이 중단되었는지 확인
        if(stop.load())
            throw std::runtime_error("enqueue on stopped ThreadPool");

        tasks.emplace([task](){ (*task)(); });
    }
    condition.notify_one();
    return res;
}

#endif // THREADPOOL_H
