#include "neuro/ui.hpp"

#include "imgui.h"

namespace neuro {

void render_graph_panel(const EngineState & state, GuiState & gui) {
    ImGui::Begin("Body Graph");

    if (gui.selected_node_id.empty()) {
        gui.selected_node_id = state.graph.focus_node_id;
    }

    ImGui::BeginChild("graph_list", ImVec2(260.0f, 0.0f), true);
    for (const auto & node_id : state.graph.insertion_order) {
        auto it = state.graph.nodes.find(node_id);
        if (it == state.graph.nodes.end()) {
            continue;
        }
        const bool selected = gui.selected_node_id == node_id;
        if (ImGui::Selectable(it->second.title.c_str(), selected)) {
            gui.selected_node_id = node_id;
        }
    }
    ImGui::EndChild();

    ImGui::SameLine();
    ImGui::BeginChild("graph_detail", ImVec2(0.0f, 0.0f), true);
    auto it = state.graph.nodes.find(gui.selected_node_id);
    if (it == state.graph.nodes.end()) {
        ImGui::TextUnformatted("No node selected.");
    } else {
        const SectionNode & node = it->second;
        ImGui::TextWrapped("%s", node.title.c_str());
        ImGui::Separator();
        ImGui::Text("Mode: %s", node.authoring_mode.c_str());
        ImGui::Text("State: %s", node.source_state.c_str());
        ImGui::Text("Iteration: %d", node.iteration);
        ImGui::TextWrapped("Rationale: %s", node.rationale.c_str());
        ImGui::Separator();
        ImGui::TextWrapped("%s", node.content.c_str());
    }
    ImGui::EndChild();
    ImGui::End();
}

} // namespace neuro
