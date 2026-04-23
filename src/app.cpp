#include "neuro/app.hpp"

#include "neuro/llama_runtime.hpp"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace neuro {
namespace {

using json = nlohmann::ordered_json;

std::string trim(const std::string & s) {
    const auto begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(begin, end - begin + 1);
}

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return s;
}

std::string build_probe_grammar(bool compact_json_schema) {
    if (compact_json_schema) {
        return R"GBNF(
root ::= "{" ws "\"p\"" ws ":" ws "\"h\"" ws "," ws "\"s\"" ws ":" ws "\"o\"" ws "}" ws
ws ::= [ \t\n]{0,8}
)GBNF";
    }
    return R"GBNF(
root ::= "{" ws "\"ping\"" ws ":" ws "\"hello\"" ws "," ws "\"status\"" ws ":" ws "\"ok\"" ws "}" ws
ws ::= [ \t\n]{0,8}
)GBNF";
}

void push_check(RuntimeSelfTestResult & result, std::string name, bool passed, std::string detail, bool hard_failure = true) {
    result.checks.push_back(RuntimeSelfTestCheck{std::move(name), passed, std::move(detail)});
    if (!passed) {
        if (hard_failure) {
            result.passed = false;
        } else {
            result.degraded = true;
        }
    }
}

RuntimeStageMetrics copy_metrics(const LlamaCompletionMetrics & metrics) {
    RuntimeStageMetrics out;
    out.attempts_used = metrics.attempts_used;
    out.prompt_tokens = metrics.prompt_tokens;
    out.output_tokens = metrics.output_tokens;
    out.tokenize_ms = metrics.tokenize_ms;
    out.context_reset_ms = metrics.context_reset_ms;
    out.prompt_decode_ms = metrics.prompt_decode_ms;
    out.decode_ms = metrics.decode_ms;
    out.first_token_ms = metrics.first_token_ms;
    out.total_ms = metrics.total_ms;
    out.timed_out = metrics.timed_out;
    out.no_first_token = metrics.no_first_token;
    out.timed_out_stage = metrics.timed_out_stage;
    return out;
}

} // namespace

AppController::AppController(AppConfig config)
    : config_(std::move(config)),
      engine_(config_.topic),
      runtime_(std::make_unique<LlamaRuntime>(config_.model_path, config_.n_predict, config_.n_ctx, config_.n_gpu_layers)) {
    const std::string lowered_model = to_lower(config_.model_path);
    if (lowered_model.find("lfm2.5-350m") != std::string::npos ||
        lowered_model.find("350m-q4") != std::string::npos ||
        lowered_model.find("tinyllama-1.1b-chat-v0.3") != std::string::npos ||
        lowered_model.find("tinyllama") != std::string::npos) {
        config_.runtime_tuning.compact_json_schema = true;
        config_.runtime_tuning.plan_prompt_chunk_tokens = std::max(config_.runtime_tuning.plan_prompt_chunk_tokens, 64);
        config_.runtime_tuning.write_prompt_chunk_tokens = std::max(config_.runtime_tuning.write_prompt_chunk_tokens, 64);
        config_.runtime_tuning.plan_max_tokens = std::min(config_.runtime_tuning.plan_max_tokens, 24);
        config_.runtime_tuning.write_max_tokens = std::min(config_.runtime_tuning.write_max_tokens, 48);
    }
    set_runtime_log_file(config_.log_file);
    append_runtime_log(engine_.state(), "INFO", "runtime log initialized file=" + config_.log_file);
    append_runtime_log(engine_.state(), "INFO",
        "llm pipeline mode=staged plan_write_with_fallback compact_json=" +
        std::string(config_.runtime_tuning.compact_json_schema ? "yes" : "no"));
}

AppController::~AppController() = default;

const AppConfig & AppController::config() const {
    return config_;
}

RuntimeTuning AppController::runtime_tuning() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_.runtime_tuning;
}

void AppController::load_state() {
    auto loaded_state = read_engine_state_binary(config_.state_file);
    std::lock_guard<std::mutex> lock(mutex_);
    if (loaded_state) {
        engine_.state() = std::move(*loaded_state);
        ++state_revision_;
        append_runtime_log(engine_.state(), "INFO", "state loaded from " + config_.state_file);
    } else {
        append_runtime_log(engine_.state(), "INFO", "no prior state file at " + config_.state_file + "; starting fresh");
    }
    if (engine_.state().knowledge_context.empty()) {
        engine_.state().knowledge_context = build_knowledge_context(engine_.state().knowledge_entries, engine_.state().topic, 4);
        ++state_revision_;
        append_runtime_log(engine_.state(), "INFO", "knowledge context rebuilt after load");
    }
}

std::optional<KnowledgeNodeRun> AppController::maybe_process_initial_knowledge() {
    if (!config_.knowledge_node_process) {
        return std::nullopt;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    append_runtime_log(engine_.state(), "INFO", "initial knowledge processing requested");
    ++state_revision_;
    return engine_.process_knowledgebase_modes(parse_knowledge_modes(config_.knowledge_node_modes));
}

void AppController::save_state() const {
    std::lock_guard<std::mutex> lock(mutex_);
    write_engine_state_binary(engine_.state(), config_.state_file);
    append_runtime_log(const_cast<EngineState &>(engine_.state()), "INFO", "state saved to " + config_.state_file);
}

void AppController::save_markdown(const std::string & prefix) const {
    std::lock_guard<std::mutex> lock(mutex_);
    engine_.persist(prefix);
    append_runtime_log(const_cast<EngineState &>(engine_.state()), "INFO", "markdown exported with prefix " + prefix);
}

EngineState AppController::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return engine_.state();
}

std::vector<std::string> AppController::configured_knowledge_modes() const {
    return parse_knowledge_modes(config_.knowledge_node_modes);
}

void AppController::set_topic(std::string topic) {
    std::lock_guard<std::mutex> lock(mutex_);
    topic = trim(topic);
    if (topic.empty()) {
        topic = "Untitled Topic";
    }
    config_.topic = topic;
    engine_.state().topic = topic;
    engine_.state().knowledge_context = build_knowledge_context(engine_.state().knowledge_entries, engine_.state().topic, 4);
    ++state_revision_;
    append_runtime_log(engine_.state(), "INFO", "topic updated to " + topic);
}

void AppController::set_runtime_tuning(RuntimeTuning tuning) {
    tuning.plan_max_tokens = std::max(16, tuning.plan_max_tokens);
    tuning.plan_max_attempts = std::max(1, tuning.plan_max_attempts);
    tuning.plan_max_elapsed_ms = std::max(250, tuning.plan_max_elapsed_ms);
    tuning.plan_prompt_chunk_tokens = std::max(8, tuning.plan_prompt_chunk_tokens);
    tuning.write_max_tokens = std::max(16, tuning.write_max_tokens);
    tuning.write_max_attempts = std::max(1, tuning.write_max_attempts);
    tuning.write_max_elapsed_ms = std::max(500, tuning.write_max_elapsed_ms);
    tuning.write_prompt_chunk_tokens = std::max(8, tuning.write_prompt_chunk_tokens);

    std::lock_guard<std::mutex> lock(mutex_);
    config_.runtime_tuning = tuning;
    append_runtime_log(engine_.state(), "INFO",
        "runtime tuning updated plan_tokens=" + std::to_string(tuning.plan_max_tokens) +
        " plan_attempts=" + std::to_string(tuning.plan_max_attempts) +
        " plan_ms=" + std::to_string(tuning.plan_max_elapsed_ms) +
        " plan_chunk=" + std::to_string(tuning.plan_prompt_chunk_tokens) +
        " write_tokens=" + std::to_string(tuning.write_max_tokens) +
        " write_attempts=" + std::to_string(tuning.write_max_attempts) +
        " write_ms=" + std::to_string(tuning.write_max_elapsed_ms) +
        " write_chunk=" + std::to_string(tuning.write_prompt_chunk_tokens) +
        " compact_json=" + std::string(tuning.compact_json_schema ? "yes" : "no"));
}

void AppController::tick(double dt) {
    std::lock_guard<std::mutex> lock(mutex_);
    engine_.tick(dt);
    ++state_revision_;
    append_runtime_log(engine_.state(), "INFO", "manual tick dt=" + std::to_string(dt));
}

void AppController::add_knowledge_entry(KnowledgeEntry entry) {
    std::lock_guard<std::mutex> lock(mutex_);
    engine_.add_knowledge_entry(std::move(entry));
    ++state_revision_;
}

KnowledgeNodeRun AppController::process_knowledge_modes(const std::vector<std::string> & modes) {
    std::lock_guard<std::mutex> lock(mutex_);
    append_runtime_log(engine_.state(), "INFO", "manual knowledge processing modes=" + std::to_string(modes.size()));
    ++state_revision_;
    return engine_.process_knowledgebase_modes(modes);
}

KnowledgeNodeRun AppController::process_configured_knowledge_modes() {
    std::lock_guard<std::mutex> lock(mutex_);
    append_runtime_log(engine_.state(), "INFO", "configured knowledge processing requested");
    ++state_revision_;
    return engine_.process_knowledgebase_modes(parse_knowledge_modes(config_.knowledge_node_modes));
}

LlmDecision AppController::run_step(const std::string & input_text) {
    const std::string trimmed = trim(input_text);
    const std::string event_text = trimmed.empty() ? "Continue developing the topic." : trimmed;
    ChemicalEngine working_engine("Untitled Topic");
    std::uint64_t expected_revision = 0;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        append_runtime_log(engine_.state(), "INFO", "run_step invoked event=" + event_text);
        engine_.add_event(event_text);
        engine_.tick(0.25);
        ++state_revision_;
        expected_revision = state_revision_;
        working_engine = engine_;
    }

    const RuntimeTuning tuning = config_.runtime_tuning;
    const LlmDecision decision = working_engine.step_with_llm(event_text, *runtime_, tuning);
    if (config_.post_knowledge_node_process) {
        const KnowledgeNodeRun run = working_engine.process_knowledgebase_modes(parse_knowledge_modes(config_.knowledge_node_modes));
        append_runtime_log(working_engine.state(), "INFO", "post-step knowledge processing complete summary=" + run.summary);
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_revision_ != expected_revision) {
            append_runtime_log(engine_.state(), "WARN",
                "run_step commit skipped because state changed during inference expected_revision="
                + std::to_string(expected_revision) + " actual_revision=" + std::to_string(state_revision_));
            throw std::runtime_error("state changed during inference; discarded stale LLM result");
        }
        engine_ = std::move(working_engine);
        ++state_revision_;
    }

    return decision;
}

RuntimeSelfTestResult AppController::run_self_test(const std::string & input_text, int iterations) {
    RuntimeSelfTestResult result;
    result.iterations_requested = std::max(1, iterations);

    const std::string event_text = trim(input_text).empty()
        ? "Produce a short grounded runtime self-test confirmation."
        : trim(input_text);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        append_runtime_log(engine_.state(), "INFO", "runtime self-test started iterations=" + std::to_string(result.iterations_requested));
    }

    try {
        struct ProbeVariant {
            std::string name;
            LlamaCompletionOptions options;
        };

        const int hw_threads = std::max(1u, std::thread::hardware_concurrency());
        const int half_threads = std::max(1, hw_threads / 2);
        const int full_threads = std::max(1, hw_threads);
        const int probe_budget_ms = std::min(3000, std::max(1000, config_.runtime_tuning.plan_max_elapsed_ms));
        const bool compact_json = config_.runtime_tuning.compact_json_schema;
        std::vector<ProbeVariant> variants;
        if (compact_json) {
            variants = {
                {"chunk64-full-threads", LlamaCompletionOptions{16, 1, 0.0f, 0.0f, 1.0f, 1, 27, probe_budget_ms, 64, full_threads, full_threads, "{\"", build_probe_grammar(true)}},
                {"chunk32-full-threads", LlamaCompletionOptions{16, 1, 0.0f, 0.0f, 1.0f, 1, 17, probe_budget_ms, 32, full_threads, full_threads, "{\"", build_probe_grammar(true)}},
                {"chunk16-full-threads", LlamaCompletionOptions{16, 1, 0.0f, 0.0f, 1.0f, 1, 7, probe_budget_ms, 16, full_threads, full_threads, "{\"", build_probe_grammar(true)}},
            };
        } else {
            variants = {
                {"chunk16-half-threads", LlamaCompletionOptions{24, 1, 0.01f, 0.01f, 0.80f, 16, 7, probe_budget_ms, 16, half_threads, half_threads, "{\"", build_probe_grammar(false)}},
                {"chunk32-half-threads", LlamaCompletionOptions{24, 1, 0.01f, 0.01f, 0.80f, 16, 17, probe_budget_ms, 32, half_threads, half_threads, "{\"", build_probe_grammar(false)}},
                {"chunk64-full-threads", LlamaCompletionOptions{24, 1, 0.01f, 0.01f, 0.80f, 16, 27, probe_budget_ms, 64, full_threads, full_threads, "{\"", build_probe_grammar(false)}},
            };
        }
        const std::string probe_prompt = compact_json
            ? "Return exactly this JSON object and nothing else: {\"p\":\"h\",\"s\":\"o\"}"
            : "Return exactly this JSON object and nothing else: {\"ping\":\"hello\",\"status\":\"ok\"}";

        std::ostringstream sweep_summary;
        bool probe_ok = false;
        for (size_t i = 0; i < variants.size(); ++i) {
            const auto & variant = variants[i];
            LlamaCompletionMetrics probe_metrics;
            std::string probe_raw;
            std::string probe_clean;
            try {
                probe_clean = runtime_->complete_json(
                    probe_prompt,
                    variant.options,
                    &probe_raw,
                    &probe_metrics);
                const json parsed = json::parse(probe_clean);
                probe_ok = parsed.is_object() &&
                    (compact_json
                        ? (parsed.value("p", "") == "h" && parsed.value("s", "") == "o")
                        : (parsed.value("ping", "") == "hello" && parsed.value("status", "") == "ok"));
            } catch (const std::exception & e) {
                probe_ok = false;
                probe_clean.clear();
                if (probe_raw.empty()) {
                    probe_raw = e.what();
                }
            }

            if (i > 0) {
                sweep_summary << " | ";
            }
            sweep_summary << variant.name
                          << ":first=" << probe_metrics.first_token_ms
                          << ",out=" << probe_metrics.output_tokens
                          << ",total=" << probe_metrics.total_ms
                          << ",timeout=" << (probe_metrics.timed_out ? "yes" : "no");
            if (!probe_metrics.timed_out_stage.empty()) {
                sweep_summary << "@" << probe_metrics.timed_out_stage;
            }

            if (probe_ok) {
                result.json_probe_variant = variant.name;
                result.json_probe_raw = std::move(probe_raw);
                result.json_probe_clean = std::move(probe_clean);
                result.json_probe_metrics = copy_metrics(probe_metrics);
                break;
            }
        }

        push_check(result, "json_probe", !result.json_probe_clean.empty(),
            result.json_probe_clean.empty() ? sweep_summary.str() : ("variant=" + result.json_probe_variant + " " + sweep_summary.str()));
        if (result.json_probe_clean.empty()) {
            return result;
        }

        push_check(result,
            "json_probe_first_token",
            result.json_probe_metrics.first_token_ms >= 0 && !result.json_probe_metrics.no_first_token,
            "variant=" + result.json_probe_variant +
            " first_token_ms=" + std::to_string(result.json_probe_metrics.first_token_ms) +
            " output_tokens=" + std::to_string(result.json_probe_metrics.output_tokens));

        push_check(result,
            "json_probe_budget",
            !result.json_probe_metrics.timed_out && result.json_probe_metrics.total_ms <= probe_budget_ms,
            "variant=" + result.json_probe_variant +
            " total_ms=" + std::to_string(result.json_probe_metrics.total_ms) +
            " budget_ms=" + std::to_string(probe_budget_ms) +
            " timeout_stage=" + (result.json_probe_metrics.timed_out_stage.empty() ? "none" : result.json_probe_metrics.timed_out_stage));
    } catch (const std::exception & e) {
        push_check(result, "json_probe", false, e.what());
        return result;
    }

    RuntimeTuning tuning = config_.runtime_tuning;
    tuning.plan_max_tokens = std::min(tuning.plan_max_tokens, 32);
    tuning.plan_prompt_chunk_tokens = std::min(std::max(8, tuning.plan_prompt_chunk_tokens), 32);
    tuning.write_max_tokens = std::min(tuning.write_max_tokens, 64);
    tuning.write_prompt_chunk_tokens = std::min(std::max(8, tuning.write_prompt_chunk_tokens), 64);
    if (tuning.compact_json_schema) {
        tuning.plan_max_tokens = std::min(tuning.plan_max_tokens, 24);
        tuning.write_max_tokens = std::min(tuning.write_max_tokens, 48);
        tuning.plan_prompt_chunk_tokens = std::max(tuning.plan_prompt_chunk_tokens, 64);
        tuning.write_prompt_chunk_tokens = std::max(tuning.write_prompt_chunk_tokens, 64);
    }

    for (int i = 0; i < result.iterations_requested; ++i) {
        ChemicalEngine working_engine("Runtime Self Test");

        KnowledgeEntry seed_entry;
        seed_entry.id = "selftest-seed";
        seed_entry.query = "Runtime Self Test";
        seed_entry.title = "Runtime Self Test Seed";
        seed_entry.summary = "The system is running a runtime self-test. Produce a short grounded confirmation that the LLM plan and write stages are operating.";
        seed_entry.text = "The system is running a runtime self-test. Produce a short grounded confirmation that the LLM plan and write stages are operating.";
        seed_entry.quality_score = 0.95;
        seed_entry.trust_score = 0.95;
        seed_entry.retrieval_score = 1.0;
        working_engine.add_knowledge_entry(seed_entry);

        working_engine.add_event(event_text);
        working_engine.tick(0.25);

        const auto started = std::chrono::steady_clock::now();
        result.last_decision = working_engine.step_with_llm(event_text, *runtime_, tuning);
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started).count();
        const EngineState & state = working_engine.state();
        result.plan_metrics = state.last_plan_metrics;
        result.write_metrics = state.last_write_metrics;
        result.iterations_completed = i + 1;

        push_check(result,
            "plan_first_token",
            state.last_plan_metrics.first_token_ms >= 0 && state.last_plan_metrics.output_tokens > 0 &&
                !state.last_plan_metrics.timed_out && !state.last_plan_metrics.no_first_token,
            "first_token_ms=" + std::to_string(state.last_plan_metrics.first_token_ms) +
            " output_tokens=" + std::to_string(state.last_plan_metrics.output_tokens) +
            " total_ms=" + std::to_string(state.last_plan_metrics.total_ms) +
            " timeout_stage=" + (state.last_plan_metrics.timed_out_stage.empty() ? "none" : state.last_plan_metrics.timed_out_stage));

        push_check(result,
            "write_first_token",
            state.last_write_metrics.first_token_ms >= 0 && state.last_write_metrics.output_tokens > 0 &&
                !state.last_write_metrics.timed_out && !state.last_write_metrics.no_first_token,
            "first_token_ms=" + std::to_string(state.last_write_metrics.first_token_ms) +
            " output_tokens=" + std::to_string(state.last_write_metrics.output_tokens) +
            " total_ms=" + std::to_string(state.last_write_metrics.total_ms) +
            " timeout_stage=" + (state.last_write_metrics.timed_out_stage.empty() ? "none" : state.last_write_metrics.timed_out_stage));

        push_check(result,
            "step_budget",
            elapsed <= tuning.plan_max_elapsed_ms + tuning.write_max_elapsed_ms + 1500,
            "elapsed_ms=" + std::to_string(elapsed) +
            " budget_ms=" + std::to_string(tuning.plan_max_elapsed_ms + tuning.write_max_elapsed_ms + 1500));

        push_check(result,
            "decision_quality",
            !result.last_decision.insufficient_context && result.last_decision.confidence >= 0.35,
            "confidence=" + std::to_string(result.last_decision.confidence) +
            " insufficient_context=" + std::string(result.last_decision.insufficient_context ? "true" : "false"),
            false);
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        append_runtime_log(engine_.state(),
            result.passed && !result.degraded ? "INFO" : "WARN",
            "runtime self-test complete passed=" + std::string(result.passed ? "yes" : "no") +
            " degraded=" + std::string(result.degraded ? "yes" : "no") +
            " iterations=" + std::to_string(result.iterations_completed));
    }

    return result;
}

} // namespace neuro
