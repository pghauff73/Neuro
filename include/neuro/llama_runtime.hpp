#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct llama_model;
struct llama_vocab;
using llama_token = int32_t;

namespace neuro {

struct LlamaCompletionOptions {
    int max_tokens = 0;
    int max_attempts = 0;
    float temperature = 0.1f;
    float retry_temperature = 0.2f;
    float top_p = 0.95f;
    int top_k = 40;
    int seed = 1234;
    int max_elapsed_ms = 0;
    int prompt_chunk_tokens = 0;
    int n_threads = 0;
    int n_threads_batch = 0;
    std::string response_prefix;
    std::string grammar;
};

struct LlamaCompletionMetrics {
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

class LlamaRuntime {
public:
    LlamaRuntime(std::string model_path, int n_predict, int n_ctx, int n_gpu_layers);
    ~LlamaRuntime();

    LlamaRuntime(const LlamaRuntime &) = delete;
    LlamaRuntime & operator=(const LlamaRuntime &) = delete;

    std::string complete_json(const std::string & user_prompt, std::string * raw_output = nullptr) const;
    std::string complete_json(const std::string & user_prompt, const LlamaCompletionOptions & options, std::string * raw_output = nullptr) const;
    std::string complete_json(const std::string & user_prompt, const LlamaCompletionOptions & options, std::string * raw_output, LlamaCompletionMetrics * metrics) const;

private:
    std::vector<llama_token> tokenize(const std::string & text, bool add_special = true) const;

    std::string model_path_;
    int n_predict_;
    int n_ctx_;
    int n_gpu_layers_;
    llama_model * model_ = nullptr;
    const llama_vocab * vocab_ = nullptr;
};

} // namespace neuro
