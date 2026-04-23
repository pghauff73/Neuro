CXX ?= g++
GUI_TARGET ?= neuro_llama_gui
TEST_TARGET ?= neuro_llama_test

CORE_SRC := src/core.cpp src/llama_runtime.cpp src/app.cpp
GUI_SRC := src/gui_main.cpp src/async_step.cpp src/ui_session.cpp src/ui_controls.cpp src/ui_chemistry.cpp src/ui_graph.cpp src/ui_knowledge.cpp src/ui_diagnostics.cpp
TEST_SRC := src/main.cpp
IMGUI_SRC := imgui-master/imgui.cpp imgui-master/imgui_draw.cpp imgui-master/imgui_tables.cpp imgui-master/imgui_widgets.cpp imgui-master/misc/cpp/imgui_stdlib.cpp
IMGUI_BACKEND_SRC := imgui-master/backends/imgui_impl_sdl2.cpp imgui-master/backends/imgui_impl_opengl3.cpp

# Prefer an installed llama package if available.
PKG_CFLAGS := $(shell pkg-config --cflags llama 2>/dev/null)
PKG_LIBS   := $(shell pkg-config --libs llama 2>/dev/null)
SDL_CFLAGS := $(shell pkg-config --cflags sdl2 gl 2>/dev/null)
SDL_LIBS   := $(shell pkg-config --libs sdl2 gl 2>/dev/null)

# Fallback for a local llama.cpp checkout/build.
LLAMA_CPP_DIR ?= ./llama.cpp
LOCAL_CFLAGS := -I$(LLAMA_CPP_DIR)/include -I$(LLAMA_CPP_DIR)/ggml/include -I$(LLAMA_CPP_DIR)/vendor
LOCAL_LIBDIRS := -L$(LLAMA_CPP_DIR)/build/bin -Wl,-rpath,$(abspath $(LLAMA_CPP_DIR)/build/bin)
LOCAL_LLAMA_SO := $(firstword $(wildcard $(LLAMA_CPP_DIR)/build/bin/libllama.so*))
LOCAL_GGML_SO := $(firstword $(wildcard $(LLAMA_CPP_DIR)/build/bin/libggml.so*))
LOCAL_GGML_BASE_SO := $(firstword $(wildcard $(LLAMA_CPP_DIR)/build/bin/libggml-base.so*))
LOCAL_GGML_CPU_SO := $(firstword $(wildcard $(LLAMA_CPP_DIR)/build/bin/libggml-cpu.so*))
LOCAL_LIBS := $(LOCAL_LLAMA_SO) $(LOCAL_GGML_SO) $(LOCAL_GGML_BASE_SO) $(LOCAL_GGML_CPU_SO)

CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -pedantic
CPPFLAGS += -Iinclude -Iimgui-master -Iimgui-master/backends -Iimgui-master/misc/cpp $(if $(PKG_CFLAGS),$(PKG_CFLAGS),$(LOCAL_CFLAGS)) $(SDL_CFLAGS)
LDFLAGS  += $(if $(PKG_LIBS),,$(LOCAL_LIBDIRS))
LDLIBS   += $(if $(PKG_LIBS),$(PKG_LIBS),$(LOCAL_LIBS)) -pthread -ldl -lm
GUI_LDLIBS := $(LDLIBS) $(SDL_LIBS)

all: $(GUI_TARGET)

$(GUI_TARGET): $(CORE_SRC) $(GUI_SRC) $(IMGUI_SRC) $(IMGUI_BACKEND_SRC)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) $(GUI_LDLIBS)

$(TEST_TARGET): $(CORE_SRC) $(TEST_SRC)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(GUI_TARGET) $(TEST_TARGET)

run-gui: $(GUI_TARGET)
	./$(GUI_TARGET)

run-test: $(TEST_TARGET)
	./$(TEST_TARGET)

.PHONY: all clean run-gui run-test
