# NeuroAi C++ Runtime

This project is a C++ llama.cpp runtime for the NeuroAI body-of-work engine.
The previous Python prompt runner has been removed; runtime state, knowledge
entries, KnowledgeNode processing, chemical updates, and prompt construction
now live in the C++ process.

## Build

Build the binary with:

```sh
make
```

The Makefile first tries `pkg-config --libs llama`. If no installed llama
package is available, it links against a local `./llama.cpp` build:

```sh
cmake -S llama.cpp -B llama.cpp/build
cmake --build llama.cpp/build -j
make
```

## Run

```sh
./neuro_llama_gui -m ./qwen2.5-1.5b-instruct-q4_k_m.gguf --topic "Your topic"
```

Known local model targets:
- `qwen2.5-1.5b-instruct-q4_k_m.gguf`
- `LFM2.5-350M-Q4_0.gguf`
- `tinyllama-1.1b-chat-v0.3.Q2_K.gguf`

The runtime loads `neuro_state.bin` on startup when it exists and writes the
current in-memory state back to that binary file on exit. Use `--state-file` to
choose another path:

```sh
./neuro_llama_gui -m ./qwen2.5-1.5b-instruct-q4_k_m.gguf --state-file ./my-session.bin
```

The ImGui application exposes session save/load, markdown export, chemistry
ticks, async LLM steps, knowledge entry management, graph inspection, and
diagnostics panels. Full runtime persistence uses the binary state file.

## Self-Test Mode

Run a headless runtime self-test without opening the GUI:

```sh
./neuro_llama_gui -m ./qwen2.5-1.5b-instruct-q4_k_m.gguf --test-mode
```

Optional controls:

```sh
./neuro_llama_gui -m ./qwen2.5-1.5b-instruct-q4_k_m.gguf --test-mode --test-iterations 2 --test-event "Produce a short grounded confirmation."
```

The self-test checks:
- direct JSON input/output against the LLM runtime
- end-to-end plan/write stage operation
- stage timing against the current runtime tuning budget
- decision quality signals such as confidence and insufficient-context flags
