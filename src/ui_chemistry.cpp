#include "neuro/ui.hpp"

#include "imgui.h"

#include <algorithm>
#include <vector>

namespace neuro {

void render_chemistry_panel(const EngineState & state) {
    ImGui::Begin("Chemistry");
    ImGui::Text("Iteration: %d", state.iteration);
    ImGui::Text("Current State: %s", state.current_state.c_str());
    ImGui::Text("Next State: %s", state.next_state.c_str());
    ImGui::Text("Authoring Mode: %s", state.authoring_mode.c_str());
    ImGui::Separator();

    for (int i = 0; i < kChemCount; ++i) {
        ImGui::Text("%s", kChemNames[i].c_str());
        ImGui::SameLine(120.0f);
        ImGui::ProgressBar(static_cast<float>(state.chem_values[i]), ImVec2(-FLT_MIN, 0.0f));
    }

    ImGui::Separator();
    std::vector<std::pair<std::string, double>> modes(state.authoring_expression.begin(), state.authoring_expression.end());
    std::sort(modes.begin(), modes.end(), [](const auto & a, const auto & b) {
        return a.second > b.second;
    });
    for (const auto & mode : modes) {
        ImGui::Text("%s", mode.first.c_str());
        ImGui::SameLine(120.0f);
        ImGui::ProgressBar(static_cast<float>(mode.second), ImVec2(-FLT_MIN, 0.0f));
    }
    ImGui::End();
}

} // namespace neuro
