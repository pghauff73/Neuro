#include "neuro/llama_runtime.hpp"

#include "neuro/core.hpp"

#include "llama.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace neuro {
namespace {

constexpr const char * kSystemPrompt =
    "Return exactly one valid JSON object. "
    "No markdown. No explanation. "
    "Use double quotes for all keys and strings.";

std::string trim(const std::string & s) {
    const auto begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(begin, end - begin + 1);
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

std::string LlamaRuntime::complete_json(const std::string & user_prompt) const {
    const std::string full_prompt = std::string(kSystemPrompt) + "\n\n" + user_prompt;
    std::vector<llama_token> prompt_tokens = tokenize(full_prompt);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = std::max(n_ctx_, static_cast<int>(prompt_tokens.size()) + n_predict_ + 16);
    ctx_params.n_batch = static_cast<int>(prompt_tokens.size());
    ctx_params.no_perf = true;

    llama_context * ctx = llama_init_from_model(model_, ctx_params);
    if (!ctx) {
        throw std::runtime_error("unable to create llama context");
    }

    auto sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = true;
    llama_sampler * sampler = llama_sampler_chain_init(sampler_params);
    if (!sampler) {
        llama_free(ctx);
        throw std::runtime_error("unable to create sampler");
    }
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.1f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(1234));

    std::string output;
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    if (llama_model_has_encoder(model_)) {
        if (llama_encode(ctx, batch) != 0) {
            llama_sampler_free(sampler);
            llama_free(ctx);
            throw std::runtime_error("llama_encode failed");
        }
        llama_token decoder_start = llama_model_decoder_start_token(model_);
        if (decoder_start == LLAMA_TOKEN_NULL) {
            decoder_start = llama_vocab_bos(vocab_);
        }
        batch = llama_batch_get_one(&decoder_start, 1);
    }

    int n_pos = 0;
    while (n_pos + batch.n_tokens < static_cast<int>(prompt_tokens.size()) + n_predict_) {
        if (llama_decode(ctx, batch) != 0) {
            llama_sampler_free(sampler);
            llama_free(ctx);
            throw std::runtime_error("llama_decode failed");
        }
        n_pos += batch.n_tokens;
        llama_token token = llama_sampler_sample(sampler, ctx, -1);
        if (llama_vocab_is_eog(vocab_, token)) {
            break;
        }
        char piece[256];
        const int n = llama_token_to_piece(vocab_, token, piece, sizeof(piece), 0, true);
        if (n < 0) {
            llama_sampler_free(sampler);
            llama_free(ctx);
            throw std::runtime_error("llama_token_to_piece failed");
        }
        output.append(piece, piece + n);
        batch = llama_batch_get_one(&token, 1);

        if (output.find('}') != std::string::npos && output.find('{') != std::string::npos) {
            const auto cleaned = clean_json_text(output);
            if (!cleaned.empty() && cleaned.front() == '{' && cleaned.back() == '}') {
                output = cleaned;
                break;
            }
        }
    }

    llama_sampler_free(sampler);
    llama_free(ctx);

    if (output.empty()) {
        throw std::runtime_error("model returned empty output");
    }
    return clean_json_text(output);
}

std::vector<llama_token> LlamaRuntime::tokenize(const std::string & text) const {
    const int n_prompt = -llama_tokenize(vocab_, text.c_str(), text.size(), nullptr, 0, true, true);
    if (n_prompt <= 0) {
        throw std::runtime_error("failed to size tokenization");
    }
    std::vector<llama_token> tokens(n_prompt);
    const int rc = llama_tokenize(vocab_, text.c_str(), text.size(), tokens.data(), tokens.size(), true, true);
    if (rc < 0) {
        throw std::runtime_error("tokenization failed");
    }
    return tokens;
}

} // namespace neuro
