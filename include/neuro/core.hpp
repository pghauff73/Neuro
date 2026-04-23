#pragma once

#include <array>
#include <optional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace neuro {

constexpr int kChemCount = 8;

extern const std::array<std::string, kChemCount> kChemNames;
extern const std::array<std::string, 3> kLayerNames;
extern const std::array<std::string, 7> kAuthorModes;
extern const std::array<std::string, 6> kKnowledgeNodeModes;

struct KnowledgeEntry {
    std::string id;
    std::string query;
    std::string title;
    std::string url;
    std::string source_domain;
    std::string summary;
    std::string text;
    std::string mode_summary;
    std::string stabilized_summary;
    std::string processed_at;
    std::string knowledge_node_mode;
    bool retrieval_candidate = false;
    std::vector<std::string> dirty_modes;
    std::string connection_hint;
    std::vector<std::string> structure_path;
    std::vector<std::string> stress_flags;
    std::string action_hint;
    double quality_score = 0.0;
    double trust_score = 0.0;
    double retrieval_score = 0.0;
};

struct SectionNode {
    std::string node_id;
    std::string title;
    std::string content;
    std::string authoring_mode;
    std::string source_state;
    int iteration = 0;
    std::vector<std::string> parent_ids;
    std::vector<std::string> child_ids;
    std::vector<std::string> tags;
    std::string rationale;
};

struct BodyGraph {
    std::unordered_map<std::string, SectionNode> nodes;
    std::vector<std::string> insertion_order;
    std::string focus_node_id;

    void add_node(SectionNode node);
    const SectionNode * focused() const;
    std::string render_markdown() const;
};

struct ModeForecast {
    std::string mode;
    double expected_value = 0.0;
    double expected_grounding = 0.0;
    double expected_conflict = 0.0;
    double expected_progress = 0.0;
    double expected_risk = 0.0;
};

struct RuntimeStageMetrics {
    int attempts_used = 0;
    int prompt_tokens = 0;
    int output_tokens = 0;
    int64_t tokenize_ms = 0;
    int64_t context_reset_ms = 0;
    int64_t prompt_decode_ms = 0;
    int64_t decode_ms = 0;
    int64_t first_token_ms = -1;
    int64_t total_ms = 0;
    bool timed_out = false;
    bool no_first_token = false;
    std::string timed_out_stage;
};

struct RuntimeTuning {
    int plan_max_tokens = 32;
    int plan_max_attempts = 1;
    int plan_max_elapsed_ms = 2500;
    int plan_prompt_chunk_tokens = 16;
    int write_max_tokens = 64;
    int write_max_attempts = 1;
    int write_max_elapsed_ms = 6000;
    int write_prompt_chunk_tokens = 32;
    bool compact_json_schema = false;
};

struct EngineState {
    std::string topic;
    int iteration = 0;
    std::string current_state = "ACh";
    std::string next_state = "ACh";
    std::string authoring_mode = "structure";
    std::unordered_map<std::string, double> authoring_expression;
    std::array<double, kChemCount> chem_values{};
    std::unordered_map<std::string, std::array<double, kChemCount>> neuro_layers;
    std::vector<std::string> recent_events;
    std::vector<KnowledgeEntry> knowledge_entries;
    std::string knowledge_context;
    std::string last_update_title;
    std::string last_update_text;
    std::string last_prompt_packet;
    std::string last_clean_response;
    std::string last_raw_response;
    double last_confidence = 0.0;
    std::vector<std::string> last_used_evidence_ids;
    bool last_insufficient_context = false;
    std::vector<std::string> last_conflicts_detected;
    int fallback_step_streak = 0;
    int productive_step_streak = 0;
    int stagnation_step_streak = 0;
    std::vector<ModeForecast> last_mode_forecasts;
    RuntimeStageMetrics last_plan_metrics;
    RuntimeStageMetrics last_write_metrics;
    int64_t last_step_total_ms = 0;
    std::vector<std::string> runtime_log;
    BodyGraph graph;
};

struct LlmDecision {
    std::string current_state;
    std::string authoring_mode;
    std::string update_title;
    std::string update_text;
    std::string next_state;
    std::string rationale;
    double confidence = 0.0;
    std::vector<std::string> used_evidence_ids;
    bool insufficient_context = false;
    std::vector<std::string> conflicts_detected;
    bool accepted = false;
};

struct PlanDecision {
    std::string current_state;
    std::string authoring_mode;
    std::string next_state;
    std::string rationale;
    double confidence = 0.0;
    std::vector<std::string> used_evidence_ids;
    bool insufficient_context = false;
    std::vector<std::string> conflicts_detected;
    bool accepted = false;
    bool fallback_used = false;
};

struct StepContext {
    std::string user_event;
    std::vector<std::string> recent_events;
    std::vector<KnowledgeEntry> evidence_entries;
    std::string prompt_packet;
};

struct StepPlan {
    StepContext context;
    std::string plan_packet;
    std::string clean_response;
    std::string raw_response;
    PlanDecision decision;
};

struct KnowledgeNodeRun {
    std::string topic;
    int entry_count = 0;
    std::vector<std::pair<std::string, int>> processed_by_mode;
    std::string summary;
    std::unordered_map<std::string, double> chemical_adjustments;
};

std::string render_chemicals(const std::array<double, kChemCount> & chem_values);
std::vector<std::string> parse_knowledge_modes(const std::string & raw_modes);
std::string build_knowledge_context(const std::vector<KnowledgeEntry> & source_entries, const std::string & topic, int max_entries = 4);
void write_engine_state_binary(const EngineState & state, const std::string & path);
std::optional<EngineState> read_engine_state_binary(const std::string & path);
void set_runtime_log_file(std::string path);
void append_runtime_log(EngineState & state, const std::string & level, const std::string & message);
void append_runtime_log_detail(EngineState & state, const std::string & level, const std::string & label, const std::string & payload);

class LlamaRuntime;

class ChemicalEngine {
public:
    explicit ChemicalEngine(std::string topic);

    EngineState & state();
    const EngineState & state() const;

    void add_event(const std::string & text);
    void add_knowledge_entry(KnowledgeEntry entry);
    void tick(double dt_seconds);
    LlmDecision step_with_llm(const std::string & user_event, const LlamaRuntime & runtime, const RuntimeTuning & tuning);
    void persist(const std::string & prefix = "body-of-work") const;
    KnowledgeNodeRun process_knowledgebase_modes(const std::vector<std::string> & modes);

private:
    StepContext prepare_step_context(const std::string & user_event) const;
    StepPlan plan_step_with_llm(const StepContext & context, const LlamaRuntime & runtime, const RuntimeTuning & tuning) const;
    LlmDecision write_step_with_llm(const StepContext & context, const StepPlan & plan, const LlamaRuntime & runtime, const RuntimeTuning & tuning);
    StepPlan build_fallback_plan(const StepContext & context, const std::string & reason) const;
    LlmDecision build_fallback_write(const StepContext & context, const StepPlan & plan, const std::string & reason) const;
    void apply_outcome_feedback(const StepPlan & plan, const LlmDecision & decision, bool write_fallback_used);
    void log_runtime(const std::string & level, const std::string & message);
    void apply_keyword_signal(const std::string & text);
    void refresh_mode();
    void commit(const LlmDecision & decision);
    static std::string to_lower(std::string s);

    EngineState state_;
};

} // namespace neuro
