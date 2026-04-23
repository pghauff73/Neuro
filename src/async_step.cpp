#include "neuro/async_step.hpp"

#include "neuro/app.hpp"

#include <chrono>
#include <utility>

namespace neuro {

bool AsyncStepRunner::busy() const {
    return busy_;
}

bool AsyncStepRunner::start(AppController & app, std::string event_text) {
    if (busy_) {
        return false;
    }
    completed_.reset();
    error_.reset();
    busy_ = true;
    future_ = std::async(std::launch::async, [&app, event_text = std::move(event_text)]() {
        return app.run_step(event_text);
    });
    return true;
}

void AsyncStepRunner::update() {
    if (!busy_) {
        return;
    }
    if (future_.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
        return;
    }
    try {
        completed_ = future_.get();
    } catch (const std::exception & e) {
        error_ = e.what();
    }
    busy_ = false;
}

void AsyncStepRunner::wait() {
    if (!busy_) {
        return;
    }
    try {
        completed_ = future_.get();
    } catch (const std::exception & e) {
        error_ = e.what();
    }
    busy_ = false;
}

std::optional<LlmDecision> AsyncStepRunner::take_completed_decision() {
    auto result = completed_;
    completed_.reset();
    return result;
}

std::optional<std::string> AsyncStepRunner::take_error() {
    auto result = error_;
    error_.reset();
    return result;
}

} // namespace neuro
