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
./neuro_llama -m ./LFM2.5-350M-Q4_0.gguf --topic "Your topic"
```

The runtime loads `neuro_state.bin` on startup when it exists and writes the
current in-memory state back to that binary file on exit. Use `--state-file` to
choose another path:

```sh
./neuro_llama -m ./LFM2.5-350M-Q4_0.gguf --state-file ./my-session.bin
```

Interactive commands:

```text
/step <text>                         add an event and run one LLM iteration
/tick <secs>                         advance chemistry without an LLM call
/knowledge-add <title>|<url>|<text>  add an in-memory knowledge entry
/knowledge [modes]                   process in-memory KnowledgeNode modes
/show                                print current engine state
/save                                write body-of-work.md
/quit                                exit
```

Knowledge entries are intentionally in-memory only. `/save` exports the current
body graph as Markdown; it does not write JSON state. Full runtime persistence
uses the binary state file.
# Neuro
# Neuro
