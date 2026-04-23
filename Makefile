CXX ?= g++
TARGET ?= neuro_llama
SRC := src/main.cpp

# Prefer an installed llama package if available.
PKG_CFLAGS := $(shell pkg-config --cflags llama 2>/dev/null)
PKG_LIBS   := $(shell pkg-config --libs llama 2>/dev/null)

# Fallback for a local llama.cpp checkout/build.
LLAMA_CPP_DIR ?= ./llama.cpp
LOCAL_CFLAGS := -I$(LLAMA_CPP_DIR)/include -I$(LLAMA_CPP_DIR)/ggml/include
LOCAL_LIBDIRS := -L$(LLAMA_CPP_DIR)/build/src -L$(LLAMA_CPP_DIR)/build/ggml/src
LOCAL_LIBS := -lllama -lggml -lggml-base -lggml-cpu

CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -pedantic
CPPFLAGS += -Iinclude $(if $(PKG_CFLAGS),$(PKG_CFLAGS),$(LOCAL_CFLAGS))
LDFLAGS  += $(if $(PKG_LIBS),,$(LOCAL_LIBDIRS))
LDLIBS   += $(if $(PKG_LIBS),$(PKG_LIBS),$(LOCAL_LIBS)) -pthread -ldl -lm

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
