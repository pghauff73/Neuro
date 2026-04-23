#pragma once

#include <string>
#include <vector>

struct llama_model;
struct llama_vocab;
using llama_token = int32_t;

namespace neuro {

class LlamaRuntime {
public:
    LlamaRuntime(std::string model_path, int n_predict, int n_ctx, int n_gpu_layers);
    ~LlamaRuntime();

    LlamaRuntime(const LlamaRuntime &) = delete;
    LlamaRuntime & operator=(const LlamaRuntime &) = delete;

    std::string complete_json(const std::string & user_prompt) const;

private:
    std::vector<llama_token> tokenize(const std::string & text) const;

    std::string model_path_;
    int n_predict_;
    int n_ctx_;
    int n_gpu_layers_;
    llama_model * model_ = nullptr;
    const llama_vocab * vocab_ = nullptr;
};

} // namespace neuro
