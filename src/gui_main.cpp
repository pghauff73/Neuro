#include "neuro/app.hpp"
#include "neuro/async_step.hpp"
#include "neuro/gui_state.hpp"
#include "neuro/ui.hpp"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_sdl2.h"

#include <SDL.h>
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <SDL_opengles2.h>
#else
#include <SDL_opengl.h>
#endif

#include <cstdlib>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using neuro::AppConfig;

struct LaunchOptions {
    AppConfig config;
    bool test_mode = false;
    int test_iterations = 1;
    std::string test_event = "Produce a short grounded runtime self-test confirmation.";
};

void print_usage(const char * argv0) {
    std::cerr
        << "Usage: " << argv0 << " -m /path/to/model.gguf [--topic \"topic\"] [--n-predict 192] [--n-ctx 2048] [--ngl 99]\n"
        << "       [--state-file neuro_state.bin] [--knowledge-node-process] [--knowledge-node-modes all|mode,mode]\n"
        << "       [--no-post-knowledge-node-process]\n"
        << "       [--test-mode] [--test-iterations N] [--test-event \"prompt\"]\n";
}

LaunchOptions parse_args(int argc, char ** argv) {
    LaunchOptions launch;
    AppConfig & args = launch.config;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need_value = [&](const std::string & flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + flag);
            }
            return argv[++i];
        };
        if (a == "-m" || a == "--model") {
            args.model_path = need_value(a);
        } else if (a == "--topic") {
            args.topic = need_value(a);
        } else if (a == "--state-file") {
            args.state_file = need_value(a);
        } else if (a == "--n-predict") {
            args.n_predict = std::stoi(need_value(a));
        } else if (a == "--n-ctx") {
            args.n_ctx = std::stoi(need_value(a));
        } else if (a == "--ngl") {
            args.n_gpu_layers = std::stoi(need_value(a));
        } else if (a == "--knowledge-node-process") {
            args.knowledge_node_process = true;
        } else if (a == "--no-post-knowledge-node-process") {
            args.post_knowledge_node_process = false;
        } else if (a == "--knowledge-node-modes") {
            args.knowledge_node_modes = need_value(a);
        } else if (a == "--test-mode") {
            launch.test_mode = true;
        } else if (a == "--test-iterations") {
            launch.test_iterations = std::max(1, std::stoi(need_value(a)));
        } else if (a == "--test-event") {
            launch.test_event = need_value(a);
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + a);
        }
    }
    if (args.model_path.empty()) {
        throw std::runtime_error("model path is required");
    }
    return launch;
}

int run_test_mode(const LaunchOptions & launch) {
    neuro::AppController app(launch.config);
    const neuro::RuntimeSelfTestResult result = app.run_self_test(launch.test_event, launch.test_iterations);

    std::cout << "Runtime self-test\n";
    std::cout << "status: " << (result.passed ? (result.degraded ? "DEGRADED" : "PASS") : "FAIL") << "\n";
    std::cout << "iterations: " << result.iterations_completed << "/" << result.iterations_requested << "\n";
    if (!result.json_probe_variant.empty()) {
        std::cout << "json probe variant: " << result.json_probe_variant << "\n";
    }
    std::cout << "json probe total ms: " << result.json_probe_metrics.total_ms << "\n";
    std::cout << "plan total ms: " << result.plan_metrics.total_ms << "\n";
    std::cout << "write total ms: " << result.write_metrics.total_ms << "\n";
    for (const auto & check : result.checks) {
        std::cout << (check.passed ? "[PASS] " : "[FAIL] ") << check.name << ": " << check.detail << "\n";
    }
    if (!result.last_decision.update_title.empty()) {
        std::cout << "last decision title: " << result.last_decision.update_title << "\n";
    }
    if (!result.last_decision.update_text.empty()) {
        std::cout << "last decision text: " << result.last_decision.update_text << "\n";
    }

    if (!result.passed) {
        return 1;
    }
    if (result.degraded) {
        return 2;
    }
    return 0;
}

std::string decision_summary(const neuro::LlmDecision & decision) {
    std::ostringstream out;
    out << "state " << decision.current_state << " -> " << decision.next_state
        << " | mode=" << decision.authoring_mode
        << " | title=" << decision.update_title;
    return out.str();
}

void apply_dashboard_style() {
    ImGuiStyle & style = ImGui::GetStyle();
    ImVec4 * colors = style.Colors;

    ImGui::StyleColorsDark();
    style.WindowRounding = 7.0f;
    style.ChildRounding = 6.0f;
    style.FrameRounding = 4.0f;
    style.PopupRounding = 6.0f;
    style.GrabRounding = 4.0f;
    style.ScrollbarRounding = 6.0f;
    style.TabRounding = 6.0f;
    style.WindowBorderSize = 1.0f;
    style.ChildBorderSize = 1.0f;
    style.FrameBorderSize = 0.0f;
    style.ItemSpacing = ImVec2(7.0f, 5.0f);
    style.ItemInnerSpacing = ImVec2(6.0f, 4.0f);
    style.WindowPadding = ImVec2(10.0f, 9.0f);
    style.FramePadding = ImVec2(8.0f, 5.0f);
    style.CellPadding = ImVec2(6.0f, 4.0f);
    style.ScrollbarSize = 12.0f;

    colors[ImGuiCol_WindowBg] = ImVec4(0.08f, 0.10f, 0.13f, 1.0f);
    colors[ImGuiCol_ChildBg] = ImVec4(0.11f, 0.13f, 0.17f, 1.0f);
    colors[ImGuiCol_PopupBg] = ImVec4(0.11f, 0.13f, 0.17f, 1.0f);
    colors[ImGuiCol_Border] = ImVec4(0.22f, 0.27f, 0.34f, 0.85f);
    colors[ImGuiCol_FrameBg] = ImVec4(0.13f, 0.16f, 0.21f, 1.0f);
    colors[ImGuiCol_FrameBgHovered] = ImVec4(0.18f, 0.23f, 0.30f, 1.0f);
    colors[ImGuiCol_FrameBgActive] = ImVec4(0.21f, 0.27f, 0.36f, 1.0f);
    colors[ImGuiCol_TitleBg] = ImVec4(0.10f, 0.12f, 0.16f, 1.0f);
    colors[ImGuiCol_TitleBgActive] = ImVec4(0.11f, 0.14f, 0.19f, 1.0f);
    colors[ImGuiCol_MenuBarBg] = ImVec4(0.09f, 0.11f, 0.15f, 1.0f);
    colors[ImGuiCol_Button] = ImVec4(0.17f, 0.38f, 0.56f, 1.0f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.22f, 0.46f, 0.67f, 1.0f);
    colors[ImGuiCol_ButtonActive] = ImVec4(0.14f, 0.31f, 0.46f, 1.0f);
    colors[ImGuiCol_Header] = ImVec4(0.15f, 0.30f, 0.43f, 0.85f);
    colors[ImGuiCol_HeaderHovered] = ImVec4(0.21f, 0.39f, 0.55f, 0.85f);
    colors[ImGuiCol_HeaderActive] = ImVec4(0.17f, 0.34f, 0.48f, 1.0f);
    colors[ImGuiCol_Tab] = ImVec4(0.13f, 0.16f, 0.21f, 1.0f);
    colors[ImGuiCol_TabHovered] = ImVec4(0.21f, 0.39f, 0.55f, 0.90f);
    colors[ImGuiCol_TabActive] = ImVec4(0.17f, 0.38f, 0.56f, 1.0f);
    colors[ImGuiCol_TabUnfocused] = ImVec4(0.10f, 0.12f, 0.16f, 1.0f);
    colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.13f, 0.17f, 0.23f, 1.0f);
    colors[ImGuiCol_Separator] = ImVec4(0.22f, 0.27f, 0.34f, 0.85f);
    colors[ImGuiCol_TableHeaderBg] = ImVec4(0.12f, 0.16f, 0.22f, 1.0f);
    colors[ImGuiCol_TableBorderStrong] = ImVec4(0.22f, 0.27f, 0.34f, 0.95f);
    colors[ImGuiCol_TableBorderLight] = ImVec4(0.18f, 0.22f, 0.29f, 0.75f);
    colors[ImGuiCol_TableRowBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_TableRowBgAlt] = ImVec4(0.10f, 0.12f, 0.16f, 0.60f);
}

void render_metric_card(const char * title, const char * value, const char * detail, const ImVec4 & accent) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.10f, 0.13f, 0.18f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(accent.x, accent.y, accent.z, 0.65f));
    ImGui::BeginChild(title, ImVec2(0.0f, 72.0f), true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.58f, 0.66f, 0.76f, 1.0f));
    ImGui::TextUnformatted(title);
    ImGui::PopStyleColor();
    ImGui::PushStyleColor(ImGuiCol_Text, accent);
    ImGui::TextWrapped("%s", value);
    ImGui::PopStyleColor();
    if (detail != nullptr && detail[0] != '\0') {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.82f, 0.88f, 0.95f, 1.0f));
        ImGui::TextWrapped("%s", detail);
        ImGui::PopStyleColor();
    }
    ImGui::EndChild();
    ImGui::PopStyleColor(2);
}

bool render_overview_panel(const neuro::EngineState & state, const neuro::GuiState & gui, bool busy) {
    ImGui::Begin("Overview", nullptr);
    if (ImGui::IsWindowCollapsed()) {
        ImGui::End();
        return true;
    }

    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.86f, 0.91f, 0.97f, 1.0f));
    ImGui::TextUnformatted("NeuroAi Control Room");
    ImGui::PopStyleColor();
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.58f, 0.66f, 0.76f, 1.0f));
    ImGui::TextWrapped("Operational overview of runtime state, inference activity, and session continuity.");
    ImGui::PopStyleColor();
    ImGui::Spacing();

    std::string iteration_value = std::to_string(state.iteration);
    std::string iteration_detail = "Mode: " + state.authoring_mode;
    std::string state_value = state.current_state;
    std::string state_detail = "Next: " + state.next_state;
    std::string llm_value = busy ? "RUNNING" : "IDLE";
    std::string llm_detail = gui.auto_run ? "Auto-run armed" : "Awaiting operator action";
    std::string knowledge_value = std::to_string(state.knowledge_entries.size());
    std::string knowledge_detail = "Tracked knowledge entries";

    if (ImGui::BeginTable("overview_cards", 4, ImGuiTableFlags_SizingStretchSame | ImGuiTableFlags_BordersInnerV)) {
        ImGui::TableNextRow();

        ImGui::TableSetColumnIndex(0);
        render_metric_card("Topic", state.topic.c_str(), "Active topic", ImVec4(0.67f, 0.85f, 0.96f, 1.0f));

        ImGui::TableSetColumnIndex(1);
        render_metric_card("Cycle", iteration_value.c_str(), iteration_detail.c_str(), ImVec4(0.54f, 0.88f, 0.75f, 1.0f));

        ImGui::TableSetColumnIndex(2);
        render_metric_card("State", state_value.c_str(), state_detail.c_str(), ImVec4(0.98f, 0.81f, 0.49f, 1.0f));

        ImGui::TableSetColumnIndex(3);
        render_metric_card("LLM", llm_value.c_str(), llm_detail.c_str(), busy ? ImVec4(0.98f, 0.74f, 0.36f, 1.0f) : ImVec4(0.50f, 0.78f, 0.96f, 1.0f));

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        render_metric_card("Knowledge", knowledge_value.c_str(), knowledge_detail.c_str(), ImVec4(0.83f, 0.69f, 0.98f, 1.0f));

        ImGui::TableSetColumnIndex(1);
        render_metric_card("Mode", state.authoring_mode.c_str(), state.current_state.c_str(), ImVec4(0.58f, 0.82f, 0.98f, 1.0f));

        ImGui::TableSetColumnIndex(2);
        const char * status_value = gui.error_message.empty() ? "NOMINAL" : "ATTENTION";
        const char * status_detail = gui.error_message.empty() ? "No active fault state" : gui.error_message.c_str();
        render_metric_card("Alerts", status_value, status_detail, gui.error_message.empty() ? ImVec4(0.54f, 0.88f, 0.75f, 1.0f) : ImVec4(0.98f, 0.52f, 0.45f, 1.0f));

        ImGui::TableSetColumnIndex(3);
        const char * action_detail = gui.status_message.empty() ? "Ready for next command" : gui.status_message.c_str();
        render_metric_card("Last Action", busy ? "IN FLIGHT" : "COMPLETE", action_detail, ImVec4(0.74f, 0.83f, 0.96f, 1.0f));

        ImGui::EndTable();
    }

    ImGui::Spacing();
    if (!gui.error_message.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.45f, 0.40f, 1.0f));
        ImGui::TextWrapped("%s", gui.error_message.c_str());
        ImGui::PopStyleColor();
    } else if (!gui.status_message.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.60f, 0.82f, 0.98f, 1.0f));
        ImGui::TextWrapped("%s", gui.status_message.c_str());
        ImGui::PopStyleColor();
    } else {
        ImGui::TextUnformatted("Ready.");
    }

    ImGui::End();
    return false;
}

std::vector<float> allocate_panel_heights(float total_height, float gap, float collapsed_height, const std::vector<float> & preferred_heights, const std::vector<bool> & collapsed) {
    std::vector<float> heights(preferred_heights.size(), collapsed_height);
    if (preferred_heights.empty()) {
        return heights;
    }

    const float gap_total = gap * static_cast<float>(preferred_heights.size() > 0 ? preferred_heights.size() - 1 : 0);
    float remaining = total_height - gap_total;
    float preferred_total = 0.0f;
    for (size_t i = 0; i < preferred_heights.size(); ++i) {
        if (collapsed[i]) {
            remaining -= collapsed_height;
        } else {
            preferred_total += preferred_heights[i];
        }
    }
    remaining = std::max(0.0f, remaining);

    for (size_t i = 0; i < preferred_heights.size(); ++i) {
        if (!collapsed[i]) {
            if (preferred_total > 0.0f) {
                heights[i] = remaining * (preferred_heights[i] / preferred_total);
            } else {
                heights[i] = remaining;
            }
        }
    }
    return heights;
}

std::vector<float> allocate_right_panel_heights(float total_height, float gap, float collapsed_height, bool knowledge_collapsed, bool chemistry_collapsed, bool diagnostics_collapsed) {
    std::vector<float> heights = {
        total_height * 0.56f,
        total_height * 0.22f,
        total_height * 0.22f,
    };

    heights[0] = knowledge_collapsed ? collapsed_height : heights[0];
    heights[1] = chemistry_collapsed ? collapsed_height : heights[1];
    heights[2] = diagnostics_collapsed ? collapsed_height : heights[2];

    float remaining = total_height - gap * 2.0f - heights[0] - heights[1] - heights[2];
    remaining = std::max(0.0f, remaining);

    if (!chemistry_collapsed) {
        heights[1] += remaining;
    } else if (!diagnostics_collapsed) {
        heights[2] += remaining;
    } else if (!knowledge_collapsed) {
        heights[0] += remaining;
    }

    return heights;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        LaunchOptions launch = parse_args(argc, argv);
        if (launch.test_mode) {
            return run_test_mode(launch);
        }
        AppConfig config = launch.config;
        neuro::AppController app(config);
        app.load_state();

        neuro::GuiState gui;
        gui.topic_input = config.topic;
        gui.knowledge_modes = config.knowledge_node_modes;
        gui.runtime_tuning = config.runtime_tuning;
        if (auto run = app.maybe_process_initial_knowledge()) {
            gui.status_message = run->summary;
        }
        gui.cached_state = app.snapshot();
        gui.topic_input = gui.cached_state.topic;
        gui.runtime_tuning = app.runtime_tuning();
        neuro::AsyncStepRunner runner;

        if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) != 0) {
            throw std::runtime_error(std::string("SDL_Init failed: ") + SDL_GetError());
        }

#if defined(__APPLE__)
        const char * glsl_version = "#version 150";
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
#else
        const char * glsl_version = "#version 130";
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
#endif
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
        SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
        SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);

        SDL_Window * window = SDL_CreateWindow(
            "NeuroAi ImGui",
            SDL_WINDOWPOS_CENTERED,
            SDL_WINDOWPOS_CENTERED,
            1440,
            900,
            SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
        if (window == nullptr) {
            throw std::runtime_error(std::string("SDL_CreateWindow failed: ") + SDL_GetError());
        }

        SDL_GLContext gl_context = SDL_GL_CreateContext(window);
        if (gl_context == nullptr) {
            throw std::runtime_error(std::string("SDL_GL_CreateContext failed: ") + SDL_GetError());
        }

        SDL_GL_MakeCurrent(window, gl_context);
        SDL_GL_SetSwapInterval(1);

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO & io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

        apply_dashboard_style();
        ImGui_ImplSDL2_InitForOpenGL(window, gl_context);
        ImGui_ImplOpenGL3_Init(glsl_version);

        bool done = false;
        while (!done) {
            SDL_Event event;
            while (SDL_PollEvent(&event)) {
                ImGui_ImplSDL2_ProcessEvent(&event);
                if (event.type == SDL_QUIT) {
                    done = true;
                }
                if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE &&
                    event.window.windowID == SDL_GetWindowID(window)) {
                    done = true;
                }
            }

            runner.update();
            if (auto error = runner.take_error()) {
                gui.error_message = *error;
                gui.auto_run = false;
            }
            if (auto decision = runner.take_completed_decision()) {
                gui.status_message = decision_summary(*decision);
                gui.cached_state = app.snapshot();
                gui.error_message.clear();
            }
            if (!runner.busy()) {
                gui.cached_state = app.snapshot();
                if (gui.auto_run) {
                    if (runner.start(app, gui.event_input)) {
                        gui.status_message = "Auto-run step started.";
                        gui.error_message.clear();
                    } else {
                        gui.auto_run = false;
                    }
                }
            }

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplSDL2_NewFrame();
            ImGui::NewFrame();

            ImGuiViewport * viewport = ImGui::GetMainViewport();
            const bool busy = runner.busy();
            const float gap = 8.0f;
            const float collapsed_h = std::max(28.0f, ImGui::GetFontSize() + ImGui::GetStyle().FramePadding.y * 2.0f + 6.0f);
            const float top_h = gui.cached_state.runtime_log.empty() ? 172.0f : 172.0f;
            const float left_w = viewport->WorkSize.x * 0.24f;
            const float right_w = viewport->WorkSize.x * 0.28f;
            const float center_w = viewport->WorkSize.x - left_w - right_w - gap * 2.0f;
            const float top_target_h = 172.0f;
            const float overview_h = ImGui::GetStateStorage()->GetBool(ImGui::GetID("overview_collapsed"), false) ? collapsed_h : top_target_h;
            const float body_y = viewport->WorkPos.y + overview_h + gap;
            const float body_h = viewport->WorkSize.y - overview_h - gap;

            const bool session_collapsed = ImGui::GetStateStorage()->GetBool(ImGui::GetID("session_collapsed"), false);
            const bool controls_collapsed = ImGui::GetStateStorage()->GetBool(ImGui::GetID("controls_collapsed"), false);
            const std::vector<float> left_heights = allocate_panel_heights(
                body_h,
                gap,
                collapsed_h,
                {270.0f, std::max(160.0f, body_h - 270.0f - gap)},
                {session_collapsed, controls_collapsed});

            const bool knowledge_collapsed = ImGui::GetStateStorage()->GetBool(ImGui::GetID("knowledge_collapsed"), false);
            const bool chemistry_collapsed = ImGui::GetStateStorage()->GetBool(ImGui::GetID("chemistry_collapsed"), false);
            const bool diagnostics_collapsed = ImGui::GetStateStorage()->GetBool(ImGui::GetID("diagnostics_collapsed"), false);
            const std::vector<float> right_heights = allocate_right_panel_heights(
                body_h,
                gap,
                collapsed_h,
                knowledge_collapsed,
                chemistry_collapsed,
                diagnostics_collapsed);

            ImGui::SetNextWindowPos(viewport->WorkPos, ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(viewport->WorkSize.x, top_target_h), ImGuiCond_Always);
            const bool overview_collapsed = render_overview_panel(gui.cached_state, gui, busy);
            ImGui::GetStateStorage()->SetBool(ImGui::GetID("overview_collapsed"), overview_collapsed);

            ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x, body_y), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(left_w, left_heights[0]), ImGuiCond_Always);
            const bool session_now_collapsed = render_session_panel(app, gui.cached_state, gui, busy);
            ImGui::GetStateStorage()->SetBool(ImGui::GetID("session_collapsed"), session_now_collapsed);

            ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x, body_y + left_heights[0] + gap), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(left_w, left_heights[1]), ImGuiCond_Always);
            const bool controls_now_collapsed = render_controls_panel(app, runner, gui);
            ImGui::GetStateStorage()->SetBool(ImGui::GetID("controls_collapsed"), controls_now_collapsed);

            ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + left_w + gap, body_y), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(center_w, body_h), ImGuiCond_Always);
            render_graph_panel(gui.cached_state, gui);

            ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + left_w + gap + center_w + gap, body_y), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(right_w, right_heights[0]), ImGuiCond_Always);
            const bool knowledge_now_collapsed = render_knowledge_panel(app, gui.cached_state, gui, busy);
            ImGui::GetStateStorage()->SetBool(ImGui::GetID("knowledge_collapsed"), knowledge_now_collapsed);

            ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + left_w + gap + center_w + gap, body_y + right_heights[0] + gap), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(right_w, right_heights[1]), ImGuiCond_Always);
            const bool chemistry_now_collapsed = render_chemistry_panel(gui.cached_state);
            ImGui::GetStateStorage()->SetBool(ImGui::GetID("chemistry_collapsed"), chemistry_now_collapsed);

            ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + left_w + gap + center_w + gap, body_y + right_heights[0] + gap + right_heights[1] + gap), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(right_w, right_heights[2]), ImGuiCond_Always);
            const bool diagnostics_now_collapsed = render_diagnostics_panel(gui.cached_state, gui, runner);
            ImGui::GetStateStorage()->SetBool(ImGui::GetID("diagnostics_collapsed"), diagnostics_now_collapsed);

            ImGui::Render();
            int drawable_w = 0;
            int drawable_h = 0;
            SDL_GL_GetDrawableSize(window, &drawable_w, &drawable_h);
            glViewport(0, 0, drawable_w, drawable_h);
            glClearColor(0.05f, 0.07f, 0.09f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            SDL_GL_SwapWindow(window);
        }

        runner.wait();
        app.save_state();
        app.save_markdown();

        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplSDL2_Shutdown();
        ImGui::DestroyContext();
        SDL_GL_DeleteContext(gl_context);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
