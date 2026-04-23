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
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

using neuro::AppConfig;

void print_usage(const char * argv0) {
    std::cerr
        << "Usage: " << argv0 << " -m /path/to/model.gguf [--topic \"topic\"] [--n-predict 192] [--n-ctx 2048] [--ngl 99]\n"
        << "       [--state-file neuro_state.bin] [--knowledge-node-process] [--knowledge-node-modes all|mode,mode]\n"
        << "       [--no-post-knowledge-node-process]\n";
}

AppConfig parse_args(int argc, char ** argv) {
    AppConfig args;
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
    return args;
}

std::string decision_summary(const neuro::LlmDecision & decision) {
    std::ostringstream out;
    out << "state " << decision.current_state << " -> " << decision.next_state
        << " | mode=" << decision.authoring_mode
        << " | title=" << decision.update_title;
    return out.str();
}

} // namespace

int main(int argc, char ** argv) {
    try {
        AppConfig config = parse_args(argc, argv);
        neuro::AppController app(config);
        app.load_state();

        neuro::GuiState gui;
        gui.knowledge_modes = config.knowledge_node_modes;
        if (auto run = app.maybe_process_initial_knowledge()) {
            gui.status_message = run->summary;
        }
        gui.cached_state = app.snapshot();
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

        ImGui::StyleColorsDark();
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
            }
            if (auto decision = runner.take_completed_decision()) {
                gui.status_message = decision_summary(*decision);
                gui.cached_state = app.snapshot();
                gui.error_message.clear();
            }
            if (!runner.busy()) {
                gui.cached_state = app.snapshot();
            }

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplSDL2_NewFrame();
            ImGui::NewFrame();

            ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f), ImGuiCond_Always);
            ImGui::SetNextWindowSize(io.DisplaySize, ImGuiCond_Always);
            ImGui::Begin("NeuroAi Workspace", nullptr,
                ImGuiWindowFlags_NoDecoration |
                ImGuiWindowFlags_NoMove |
                ImGuiWindowFlags_NoSavedSettings);

            render_session_panel(app, gui.cached_state, gui);
            render_controls_panel(app, runner, gui);

            ImGui::Columns(2, "main_columns", true);
            render_chemistry_panel(gui.cached_state);
            render_graph_panel(gui.cached_state, gui);
            ImGui::NextColumn();
            render_knowledge_panel(app, gui.cached_state, gui);
            render_diagnostics_panel(gui.cached_state, gui, runner);
            ImGui::Columns(1);

            ImGui::End();

            ImGui::Render();
            glViewport(0, 0, static_cast<int>(io.DisplaySize.x), static_cast<int>(io.DisplaySize.y));
            glClearColor(0.08f, 0.08f, 0.10f, 1.0f);
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
