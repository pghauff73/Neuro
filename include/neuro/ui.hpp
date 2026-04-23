#pragma once

#include "neuro/async_step.hpp"
#include "neuro/core.hpp"
#include "neuro/gui_state.hpp"

namespace neuro {

class AppController;

bool render_session_panel(AppController & app, const EngineState & state, GuiState & gui, bool busy);
bool render_controls_panel(AppController & app, AsyncStepRunner & runner, GuiState & gui);
bool render_chemistry_panel(const EngineState & state);
bool render_graph_panel(const EngineState & state, GuiState & gui);
bool render_knowledge_panel(AppController & app, const EngineState & state, GuiState & gui, bool busy);
bool render_diagnostics_panel(const EngineState & state, const GuiState & gui, const AsyncStepRunner & runner);

} // namespace neuro
