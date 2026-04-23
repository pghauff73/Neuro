#pragma once

#include "neuro/core.hpp"

#include <future>
#include <optional>
#include <string>

namespace neuro {

class AppController;

class AsyncStepRunner {
public:
    bool busy() const;
    bool start(AppController & app, std::string event_text);
    void update();
    void wait();
    std::optional<LlmDecision> take_completed_decision();
    std::optional<std::string> take_error();

private:
    std::future<LlmDecision> future_;
    bool busy_ = false;
    std::optional<LlmDecision> completed_;
    std::optional<std::string> error_;
};

} // namespace neuro
