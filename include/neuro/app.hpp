#pragma once

#include "neuro/core.hpp"

#include <memory>
#include <mutex>
#include <optional>
#include <cstdint>
#include <string>
#include <vector>

namespace neuro {

struct AppConfig {
    std::string model_path;
    std::string topic = "Untitled Topic";
    std::string state_file = "neuro_state.bin";
    std::string log_file = "neuro_runtime.log";
    int n_predict = 192;
    int n_ctx = 2048;
    int n_gpu_layers = 99;
    RuntimeTuning runtime_tuning;
    bool knowledge_node_process = true;
    bool post_knowledge_node_process = true;
    std::string knowledge_node_modes = "all";
};

struct RuntimeSelfTestCheck {
    std::string name;
    bool passed = false;
    std::string detail;
};

struct RuntimeSelfTestResult {
    bool passed = true;
    bool degraded = false;
    int iterations_requested = 1;
    int iterations_completed = 0;
    std::string json_probe_variant;
    RuntimeStageMetrics json_probe_metrics;
    RuntimeStageMetrics plan_metrics;
    RuntimeStageMetrics write_metrics;
    std::string json_probe_raw;
    std::string json_probe_clean;
    LlmDecision last_decision;
    std::vector<RuntimeSelfTestCheck> checks;
};

class AppController {
public:
    explicit AppController(AppConfig config);
    ~AppController();

    const AppConfig & config() const;
    void load_state();
    std::optional<KnowledgeNodeRun> maybe_process_initial_knowledge();
    void save_state() const;
    void save_markdown(const std::string & prefix = "body-of-work") const;
    EngineState snapshot() const;
    std::vector<std::string> configured_knowledge_modes() const;
    RuntimeTuning runtime_tuning() const;

    void set_topic(std::string topic);
    void set_runtime_tuning(RuntimeTuning tuning);
    void tick(double dt);
    void add_knowledge_entry(KnowledgeEntry entry);
    KnowledgeNodeRun process_knowledge_modes(const std::vector<std::string> & modes);
    KnowledgeNodeRun process_configured_knowledge_modes();
    LlmDecision run_step(const std::string & input_text);
    RuntimeSelfTestResult run_self_test(const std::string & event_text, int iterations);

private:
    AppConfig config_;
    mutable std::mutex mutex_;
    std::uint64_t state_revision_ = 0;
    ChemicalEngine engine_;
    std::unique_ptr<class LlamaRuntime> runtime_;
};

} // namespace neuro
