#pragma once

#include "neuro/core.hpp"

#include <string>

namespace neuro {

struct GuiState {
    std::string event_input;
    float tick_seconds = 0.25f;
    std::string knowledge_title;
    std::string knowledge_url;
    std::string knowledge_summary;
    std::string knowledge_text;
    std::string knowledge_modes = "all";
    std::string selected_node_id;
    std::string status_message;
    std::string error_message;
    EngineState cached_state;
};

} // namespace neuro
