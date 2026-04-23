#pragma once

#include "neuro/async_step.hpp"
#include "neuro/core.hpp"
#include "neuro/gui_state.hpp"

namespace neuro {

class AppController;

void render_session_panel(AppController & app, const EngineState & state, GuiState & gui);
void render_controls_panel(AppController & app, AsyncStepRunner & runner, GuiState & gui);
void render_chemistry_panel(const EngineState & state);
void render_graph_panel(const EngineState & state, GuiState & gui);
void render_knowledge_panel(AppController & app, const EngineState & state, GuiState & gui);
void render_diagnostics_panel(const EngineState & state, const GuiState & gui, const AsyncStepRunner & runner);

} // namespace neuro
