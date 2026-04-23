#include "neuro/llama_runtime.hpp"

#include "neuro/core.hpp"

#include "llama.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace neuro {
namespace {

using json = nlohmann::ordered_json;

constexpr const char * kSystemPrompt =
    "Return exactly one valid JSON object. "
    "No markdown. No explanation. "
    "Use double quotes for all keys and strings.";

constexpr int kMaxCompletionAttempts = 3;
constexpr int kMinCompletionTokens = 16;
constexpr int kDefaultMaxElapsedMs = 8000;
constexpr int kDefaultPromptChunkTokens = 32;

enum class TimeoutStage {
    None,
    Prompt,
    Generation,
};

struct AbortState {
    std::chrono::steady_clock::time_point started;
    int max_elapsed_ms = 0;
    TimeoutStage stage = TimeoutStage::None;
    bool tripped = false;
};

bool abort_if_timed_out(void * data) {
    auto * state = static_cast<AbortState *>(data);
    if (!state || state->max_elapsed_ms <= 0) {
        return false;
    }
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - state->started).count();
    if (elapsed_ms >= state->max_elapsed_ms) {
        state->tripped = true;
        return true;
    }
    return false;
}

std::string timeout_stage_name(TimeoutStage stage) {
    switch (stage) {
        case TimeoutStage::Prompt: return "prompt";
        case TimeoutStage::Generation: return "generation";
        case TimeoutStage::None:
        default: return "";
    }
}

std::string trim(const std::string & s) {
    const auto begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(begin, end - begin + 1);
}

std::string compact_excerpt(const std::string & text, size_t max_chars = 200) {
    const std::string trimmed = trim(text);
    if (trimmed.size() <= max_chars) {
        return trimmed;
    }
    return trimmed.substr(0, max_chars - 3) + "...";
}

LlamaCompletionOptions with_defaults(LlamaCompletionOptions options, int default_n_predict) {
    if (options.max_tokens <= 0) {
        options.max_tokens = default_n_predict;
    }
    if (options.max_attempts <= 0) {
        options.max_attempts = kMaxCompletionAttempts;
    }
    if (options.top_k <= 0) {
        options.top_k = 40;
    }
    if (options.top_p <= 0.0f) {
        options.top_p = 0.95f;
    }
    if (options.max_elapsed_ms <= 0) {
        options.max_elapsed_ms = kDefaultMaxElapsedMs;
    }
    if (options.prompt_chunk_tokens <= 0) {
        options.prompt_chunk_tokens = kDefaultPromptChunkTokens;
    }
    return options;
}

std::string clean_json_text(const std::string & raw) {
    std::string text = trim(raw);
    if (text.rfind("```", 0) == 0) {
        const auto first_nl = text.find('\n');
        if (first_nl != std::string::npos) {
            text = text.substr(first_nl + 1);
        }
        const auto last_ticks = text.rfind("```");
        if (last_ticks != std::string::npos) {
            text = text.substr(0, last_ticks);
        }
        text = trim(text);
    }
    const auto start = text.find('{');
    const auto end = text.rfind('}');
    if (start != std::string::npos && end != std::string::npos && end >= start) {
        text = text.substr(start, end - start + 1);
    }
    return text;
}

bool is_valid_json_object(const std::string & text) {
    try {
        const json parsed = json::parse(text);
        return parsed.is_object();
    } catch (...) {
        return false;
    }
}

} // namespace

LlamaRuntime::LlamaRuntime(std::string model_path, int n_predict, int n_ctx, int n_gpu_layers)
    : model_path_(std::move(model_path)), n_predict_(n_predict), n_ctx_(n_ctx), n_gpu_layers_(n_gpu_layers) {
    ggml_backend_load_all();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers_;
    model_ = llama_model_load_from_file(model_path_.c_str(), model_params);
    if (!model_) {
        throw std::runtime_error("unable to load model: " + model_path_);
    }
    vocab_ = llama_model_get_vocab(model_);
    if (!vocab_) {
        throw std::runtime_error("unable to get vocab");
    }
}

LlamaRuntime::~LlamaRuntime() {
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
}

std::string LlamaRuntime::complete_json(const std::string & user_prompt, std::string * raw_output) const {
    return complete_json(user_prompt, LlamaCompletionOptions{}, raw_output, nullptr);
}

std::string LlamaRuntime::complete_json(const std::string & user_prompt, const LlamaCompletionOptions & requested_options, std::string * raw_output) const {
    return complete_json(user_prompt, requested_options, raw_output, nullptr);
}

std::string LlamaRuntime::complete_json(const std::string & user_prompt, const LlamaCompletionOptions & requested_options, std::string * raw_output, LlamaCompletionMetrics * metrics) const {
    using clock = std::chrono::steady_clock;
    const auto total_started = clock::now();
    std::string last_error = "model returned empty output";
    std::string last_output_excerpt;
    std::string last_raw_output;
    const LlamaCompletionOptions options = with_defaults(requested_options, n_predict_);
    const int max_tokens = std::max(kMinCompletionTokens, options.max_tokens);
    LlamaCompletionMetrics local_metrics;

    for (int attempt = 0; attempt < options.max_attempts; ++attempt) {
        local_metrics.attempts_used = attempt + 1;
        const auto attempt_started = clock::now();
        const std::string full_prompt = attempt == 0
            ? std::string(kSystemPrompt) + "\n\n" + user_prompt
            : std::string(kSystemPrompt) + "\n\n"
                + "Retry because the previous answer was blank or invalid. "
                + "Return one compact JSON object immediately.\n\n"
                + user_prompt;
        const auto tokenize_started = clock::now();
        std::vector<llama_token> prompt_tokens = tokenize(full_prompt);
        local_metrics.tokenize_ms += std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - tokenize_started).count();
        local_metrics.prompt_tokens = static_cast<int>(prompt_tokens.size());
        if (static_cast<int>(prompt_tokens.size()) + max_tokens + 16 > n_ctx_) {
            last_error = "prompt exceeds runtime context budget";
            break;
        }
        if (std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - total_started).count() >= options.max_elapsed_ms) {
            local_metrics.timed_out = true;
            last_error = "completion timed out before decode";
            break;
        }
        const auto context_started = clock::now();
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = n_ctx_;
        const uint32_t prompt_chunk_tokens = static_cast<uint32_t>(std::max(8, options.prompt_chunk_tokens));
        ctx_params.n_batch = std::min<uint32_t>(n_ctx_, prompt_chunk_tokens);
        ctx_params.n_ubatch = std::min<uint32_t>(n_ctx_, prompt_chunk_tokens);
        const int default_thread_count = std::max(1u, std::thread::hardware_concurrency());
        ctx_params.n_threads = options.n_threads > 0 ? options.n_threads : default_thread_count;
        ctx_params.n_threads_batch = options.n_threads_batch > 0 ? options.n_threads_batch : ctx_params.n_threads;
        ctx_params.no_perf = true;
        llama_context * ctx = llama_init_from_model(model_, ctx_params);
        if (!ctx) {
            throw std::runtime_error("unable to create llama context");
        }
        local_metrics.context_reset_ms += std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - context_started).count();
        AbortState abort_state{total_started, options.max_elapsed_ms, TimeoutStage::None, false};
        llama_set_abort_callback(ctx, abort_if_timed_out, &abort_state);

        auto sampler_params = llama_sampler_chain_default_params();
        sampler_params.no_perf = true;
        llama_sampler * sampler = llama_sampler_chain_init(sampler_params);
        if (!sampler) {
            llama_free(ctx);
            throw std::runtime_error("unable to create sampler");
        }
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(options.top_k));
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(options.top_p, 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(attempt == 0 ? options.temperature : options.retry_temperature));
        if (!options.grammar.empty()) {
            llama_sampler * grammar = llama_sampler_init_grammar(vocab_, options.grammar.c_str(), "root");
            if (!grammar) {
                llama_sampler_free(sampler);
                llama_free(ctx);
                throw std::runtime_error("unable to initialize grammar sampler");
            }
            llama_sampler_chain_add(sampler, grammar);
        }
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(options.seed + attempt));

        std::string output;
        std::string raw_attempt_output;
        bool decode_failed = false;
        const auto prompt_decode_started = clock::now();
        if (llama_model_has_encoder(model_)) {
            llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
            abort_state.stage = TimeoutStage::Prompt;
            if (llama_encode(ctx, batch) != 0) {
                llama_sampler_free(sampler);
                llama_free(ctx);
                throw std::runtime_error("llama_encode failed");
            }
            llama_token decoder_start = llama_model_decoder_start_token(model_);
            if (decoder_start == LLAMA_TOKEN_NULL) {
                decoder_start = llama_vocab_bos(vocab_);
            }
            llama_batch decode_batch = llama_batch_get_one(&decoder_start, 1);
            abort_state.stage = TimeoutStage::Prompt;
            if (llama_decode(ctx, decode_batch) != 0) {
                if (abort_state.tripped) {
                    local_metrics.timed_out = true;
                    local_metrics.timed_out_stage = timeout_stage_name(abort_state.stage);
                    last_error = "completion timed out during prompt ingest";
                    decode_failed = true;
                } else {
                    llama_sampler_free(sampler);
                    llama_free(ctx);
                    throw std::runtime_error("llama_decode failed");
                }
            }
        } else {
            size_t prompt_pos = 0;
            while (prompt_pos < prompt_tokens.size()) {
                if (std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - total_started).count() >= options.max_elapsed_ms) {
                    local_metrics.timed_out = true;
                    local_metrics.timed_out_stage = "prompt";
                    last_error = "completion timed out during prompt ingest";
                    decode_failed = true;
                    break;
                }
                const size_t chunk = std::min(prompt_tokens.size() - prompt_pos, static_cast<size_t>(prompt_chunk_tokens));
                llama_batch prompt_batch = llama_batch_get_one(prompt_tokens.data() + prompt_pos, chunk);
                abort_state.stage = TimeoutStage::Prompt;
                if (llama_decode(ctx, prompt_batch) != 0) {
                    if (abort_state.tripped) {
                        local_metrics.timed_out = true;
                        local_metrics.timed_out_stage = timeout_stage_name(abort_state.stage);
                        last_error = "completion timed out during prompt ingest";
                        decode_failed = true;
                        break;
                    }
                    llama_sampler_free(sampler);
                    llama_free(ctx);
                    throw std::runtime_error("llama_decode failed");
                }
                prompt_pos += chunk;
            }
        }

        if (!decode_failed && !options.response_prefix.empty()) {
            auto prefix_tokens = tokenize(options.response_prefix, false);
            size_t prefix_pos = 0;
            while (prefix_pos < prefix_tokens.size()) {
                if (std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - total_started).count() >= options.max_elapsed_ms) {
                    local_metrics.timed_out = true;
                    local_metrics.timed_out_stage = "prompt";
                    last_error = "completion timed out during prompt ingest";
                    decode_failed = true;
                    break;
                }
                const size_t chunk = std::min(prefix_tokens.size() - prefix_pos, static_cast<size_t>(prompt_chunk_tokens));
                llama_batch prefix_batch = llama_batch_get_one(prefix_tokens.data() + prefix_pos, chunk);
                abort_state.stage = TimeoutStage::Prompt;
                if (llama_decode(ctx, prefix_batch) != 0) {
                    if (abort_state.tripped) {
                        local_metrics.timed_out = true;
                        local_metrics.timed_out_stage = timeout_stage_name(abort_state.stage);
                        last_error = "completion timed out during prompt ingest";
                        decode_failed = true;
                        break;
                    }
                    llama_sampler_free(sampler);
                    llama_free(ctx);
                    throw std::runtime_error("llama_decode failed");
                }
                prefix_pos += chunk;
            }
            if (!decode_failed) {
                for (const llama_token token : prefix_tokens) {
                    llama_sampler_accept(sampler, token);
                }
            }
            output = options.response_prefix;
            raw_attempt_output = options.response_prefix;
        }
        local_metrics.prompt_decode_ms += std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - prompt_decode_started).count();

        int output_tokens = 0;
        const auto decode_started = clock::now();
        while (!decode_failed && output_tokens < max_tokens) {
            if (std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - total_started).count() >= options.max_elapsed_ms) {
                local_metrics.timed_out = true;
                local_metrics.timed_out_stage = "generation";
                last_error = "completion timed out during decode";
                break;
            }
            llama_token token = llama_sampler_sample(sampler, ctx, -1);
            if (llama_vocab_is_eog(vocab_, token)) {
                break;
            }
            ++output_tokens;
            if (local_metrics.first_token_ms < 0) {
                local_metrics.first_token_ms = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - attempt_started).count();
            }
            char piece[256];
            const int n = llama_token_to_piece(vocab_, token, piece, sizeof(piece), 0, true);
            if (n < 0) {
                llama_sampler_free(sampler);
                llama_free(ctx);
                throw std::runtime_error("llama_token_to_piece failed");
            }
            output.append(piece, piece + n);
            raw_attempt_output.append(piece, piece + n);
            llama_batch next_batch = llama_batch_get_one(&token, 1);
            abort_state.stage = TimeoutStage::Generation;
            if (llama_decode(ctx, next_batch) != 0) {
                if (abort_state.tripped) {
                    local_metrics.timed_out = true;
                    local_metrics.timed_out_stage = timeout_stage_name(abort_state.stage);
                    last_error = "completion timed out during decode";
                    break;
                }
                llama_sampler_free(sampler);
                llama_free(ctx);
                throw std::runtime_error("llama_decode failed");
            }

            if (output.find('}') != std::string::npos && output.find('{') != std::string::npos) {
                const auto cleaned = clean_json_text(output);
                if (!cleaned.empty() && cleaned.front() == '{' && cleaned.back() == '}' && is_valid_json_object(cleaned)) {
                    if (raw_output) {
                        *raw_output = raw_attempt_output;
                    }
                    output = cleaned;
                    break;
                }
            }
        }
        local_metrics.decode_ms += std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - decode_started).count();
        local_metrics.output_tokens = output_tokens;

        llama_sampler_free(sampler);
        llama_free(ctx);
        local_metrics.no_first_token = local_metrics.first_token_ms < 0;

        if (raw_attempt_output.empty()) {
            if (!local_metrics.timed_out) {
                last_error = "model returned empty output";
            }
            continue;
        }
        last_raw_output = raw_attempt_output;

        const std::string cleaned = clean_json_text(output);
        if (!cleaned.empty() && cleaned.front() == '{' && cleaned.back() == '}' && is_valid_json_object(cleaned)) {
            if (raw_output) {
                *raw_output = raw_attempt_output;
            }
            local_metrics.total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - total_started).count();
            if (metrics) {
                *metrics = local_metrics;
            }
            return cleaned;
        }

        last_output_excerpt = compact_excerpt(raw_attempt_output);
        last_error = last_output_excerpt.empty()
            ? "model returned invalid JSON output"
            : "model returned invalid JSON output: " + last_output_excerpt;
        if (trim(raw_attempt_output).size() <= 2) {
            break;
        }
    }

    if (raw_output) {
        *raw_output = last_raw_output;
    }
    local_metrics.total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - total_started).count();
    if (metrics) {
        *metrics = local_metrics;
    }
    throw std::runtime_error(last_error);
}

std::vector<llama_token> LlamaRuntime::tokenize(const std::string & text, bool add_special) const {
    const int n_prompt = -llama_tokenize(vocab_, text.c_str(), text.size(), nullptr, 0, add_special, true);
    if (n_prompt <= 0) {
        throw std::runtime_error("failed to size tokenization");
    }
    std::vector<llama_token> tokens(n_prompt);
    const int rc = llama_tokenize(vocab_, text.c_str(), text.size(), tokens.data(), tokens.size(), add_special, true);
    if (rc < 0) {
        throw std::runtime_error("tokenization failed");
    }
    return tokens;
}

} // namespace neuro
