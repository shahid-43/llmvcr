# llmvcr

> Record and replay LLM API responses for fast, free, deterministic tests.

[![PyPI version](https://img.shields.io/pypi/v/llmvcr.svg)](https://pypi.org/project/llmvcr/)
[![Python](https://img.shields.io/pypi/pyversions/llmvcr.svg)](https://pypi.org/project/llmvcr/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-81%20passed-brightgreen.svg)]()

---

## The Problem

Testing code that calls an LLM API is painful:

| Problem | Impact |
|---|---|
| 🐢 **Slow** | Each test waits 2–10 seconds for a real API response |
| 💸 **Expensive** | Every test run costs real money |
| 🎲 **Flaky** | Same prompt, different response — random test failures |
| 📡 **Needs internet** | Tests break offline or when the API is down |
| 🔒 **Secrets in CI** | You must inject API keys into every environment |

**llmvcr solves all five** by recording real LLM responses once and replaying them instantly — for free, offline, with 100% identical results every time.

---

## How It Works

Just like **VCR.py** records and replays HTTP requests, `llmvcr` records and replays LLM API responses. But unlike VCR.py, it works at the **SDK layer** — not the raw HTTP layer — so it handles streaming, never touches your API key, and produces clean human-readable YAML cassettes.

```
First run   →  hits real API  →  saves response to cassette YAML
Every run after  →  reads from cassette  →  instant, free, offline
```

---

## Supported Providers

| Provider | SDK | Models | Install |
|---|---|---|---|
| **OpenAI** | `openai` | GPT-4, GPT-4o, GPT-3.5 | `pip install llmvcr[openai]` |
| **Anthropic** | `anthropic` | Claude 3.5, Claude 3 | `pip install llmvcr[anthropic]` |
| **Google Gemini** | `google-genai` | Gemini 2.0, 1.5 | `pip install llmvcr[gemini]` |
| **Llama (local)** | `ollama` | Llama 3.1, 3.2, Mistral, Qwen | `pip install llmvcr[ollama]` |
| **Llama (cloud)** | `groq` | Llama 3.3 70B, 3.1 8B | `pip install llmvcr[groq]` |

---

## Installation

```bash
# Core + one provider
pip install llmvcr[openai]
pip install llmvcr[anthropic]
pip install llmvcr[gemini]
pip install llmvcr[ollama]
pip install llmvcr[groq]

# Everything at once
pip install llmvcr[all]
```

> **Ollama note:** Also requires the Ollama app running locally.
> Download from [ollama.com](https://ollama.com), then `ollama pull llama3.1`.

---

## Quick Start

### 1. The `@cassette` decorator (recommended for tests)

Add one decorator to your test. First run records, every run after replays.

```python
import llmvcr
import openai

@llmvcr.cassette("cassettes/summarize.yaml")
def test_summarize():
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Summarize AI in one sentence."}]
    )
    assert "intelligence" in response.choices[0].message.content

# First run:  hits OpenAI API → saves to cassettes/summarize.yaml
# Every run:  reads from cassette → instant, free, offline
```

### 2. Context managers (for scripts and notebooks)

```python
import llmvcr
import openai

# Record
with llmvcr.record("cassettes/demo.yaml"):
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}]
    )

# Replay (never hits the API)
with llmvcr.playback("cassettes/demo.yaml"):
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)  # instant
```

### 3. Explicit modes

```python
# auto (default) — record if cassette missing, replay if it exists
@llmvcr.cassette("cassettes/test.yaml")

# record — always re-record, overwrite cassette
@llmvcr.cassette("cassettes/test.yaml", mode="record")

# playback — always replay, raise error if cassette missing
@llmvcr.cassette("cassettes/test.yaml", mode="playback")
```

---

## Provider Examples

### OpenAI

```python
import llmvcr
import openai

@llmvcr.cassette("cassettes/openai_test.yaml", provider="openai")
def test_openai():
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "What is Python?"}]
    )
    assert "programming" in response.choices[0].message.content
```

### Anthropic

```python
import llmvcr
import anthropic

@llmvcr.cassette("cassettes/claude_test.yaml", provider="anthropic")
def test_anthropic():
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
        messages=[{"role": "user", "content": "What is Python?"}]
    )
    assert "programming" in response.content[0].text
```

### Google Gemini

```python
import llmvcr
from google import genai
from google.genai import types

@llmvcr.cassette("cassettes/gemini_test.yaml", provider="gemini")
def test_gemini():
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="What is Python?",
        config=types.GenerateContentConfig(
            system_instruction="You are a concise assistant."
        )
    )
    assert "programming" in response.text
```

### Llama via Ollama (local)

```python
import llmvcr
import ollama

@llmvcr.cassette("cassettes/llama_local_test.yaml", provider="ollama")
def test_ollama():
    response = ollama.chat(
        model="llama3.1",
        messages=[{"role": "user", "content": "What is Python?"}]
    )
    assert "programming" in response.message.content
    # also works: response["message"]["content"]
```

### Llama via Groq (cloud)

```python
import llmvcr
from groq import Groq

@llmvcr.cassette("cassettes/llama_groq_test.yaml", provider="groq")
def test_groq():
    client = Groq()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "What is Python?"}]
    )
    assert "programming" in response.choices[0].message.content
```

---

## What a Cassette Looks Like

After recording, a clean human-readable YAML file is saved — no binary blobs, no API keys, safe to commit to git:

```yaml
llmvcr_version: '0.1'
provider: openai
recorded_at: '2025-03-03T10:24:00Z'
interactions:
  - request:
      model: gpt-4
      messages:
        - role: user
          content: Summarize AI in one sentence.
    response:
      id: chatcmpl-abc123
      choices:
        - message:
            role: assistant
            content: Artificial intelligence is the simulation of human-like reasoning
              and decision-making by computer systems.
      usage:
        prompt_tokens: 14
        completion_tokens: 22
        total_tokens: 36
```

You can review it, edit it, and diff it in pull requests just like any other file.

---

## CLI

```bash
# Inspect a cassette — see stored interactions, token counts, models
llmvcr info cassettes/demo.yaml

# Record a script to a cassette
llmvcr record --output cassettes/demo.yaml python my_script.py

# Replay a script using a cassette (no API call made)
llmvcr playback --cassette cassettes/demo.yaml python my_script.py
```

Example `llmvcr info` output:

```
Cassette:      cassettes/demo.yaml
Provider:      openai
Recorded at:   2025-03-03T10:24:00Z
Version:       0.1
Interactions:  2

  [1] model=gpt-4
      prompt:   'Summarize AI in one sentence.'
      response: 'Artificial intelligence is the simulation of...'
      tokens:   36

  [2] model=gpt-4
      prompt:   'What is Python?'
      response: 'Python is a high-level programming language...'
      tokens:   38
```

---

## Why Not VCR.py or pytest-recording?

Those tools were built for generic HTTP APIs. LLMs break them in several ways:

| Issue | VCR.py / pytest-recording | llmvcr |
|---|---|---|
| Streaming responses | ❌ Crashes with chunked read errors | ✅ Records and replays natively |
| API key in cassette | ❌ Saved in plaintext by default | ✅ Never sees credentials |
| Adding `temperature=0.7` | ❌ Breaks cassette match | ✅ Ignored, match still works |
| Gzip-encoded responses | ❌ Saves as unreadable binary | ✅ Always clean YAML |
| Zero-config setup | ❌ Requires conftest.py boilerplate | ✅ Works out of the box |

The root cause: VCR.py intercepts raw HTTP bytes. llmvcr intercepts at the SDK layer, where responses are already decoded Python objects — no gzip, no auth headers, no socket state.

---

## Running the Tests

### Install dev dependencies

```bash
# Clone the repo
git clone https://github.com/your-username/llmvcr
cd llmvcr

# Install in editable mode with dev dependencies
pip install -e ".[all]"
pip install pytest
```

### Run all tests

```bash
pytest
```

### Run with verbose output (see each test name)

```bash
pytest -v
```

### Run a specific test file

```bash
pytest tests/test_matching.py
pytest tests/test_storage.py
pytest tests/test_recorder.py
pytest tests/test_cassettes.py
pytest tests/test_new_providers.py
```

### Run a specific test by name

```bash
pytest -v -k "test_playback_ignores_temperature"
```

### Run with coverage report

```bash
pip install pytest-cov
pytest --cov=llmvcr --cov-report=term-missing
```

### Expected output

```
============================= test session starts ==============================
collected 81 items

tests/test_cassettes.py ....................                             [ 25%]
tests/test_matching.py .........                                        [ 36%]
tests/test_new_providers.py ...................................           [ 79%]
tests/test_recorder.py .........                                        [ 90%]
tests/test_storage.py .......                                           [100%]

============================== 81 passed in 0.70s ==============================
```

> All 81 tests run **without any API keys** — they use pre-recorded cassettes in `tests/cassettes/`.

---

## Project Structure

```
llmvcr/
├── pyproject.toml                  # Package metadata and build config
├── requirements.txt                # All dependencies listed
├── README.md
├── LICENSE
│
├── src/
│   └── llmvcr/
│       ├── __init__.py             # Public API: cassette, record, playback
│       ├── cassette.py             # @llmvcr.cassette() decorator
│       ├── context.py              # with llmvcr.record() / playback()
│       ├── recorder.py             # Core engine: record/playback state
│       ├── matching.py             # Semantic request matching
│       ├── storage.py              # Read/write cassette YAML files
│       ├── errors.py               # Custom exceptions
│       ├── _patch.py               # Provider routing
│       ├── cli.py                  # CLI commands
│       │
│       └── providers/
│           ├── openai_provider.py
│           ├── anthropic_provider.py
│           ├── gemini_provider.py
│           ├── ollama_provider.py
│           └── groq_provider.py
│
└── tests/
    ├── cassettes/
    │   ├── sample_openai.yaml
    │   ├── sample_anthropic.yaml
    │   ├── sample_gemini.yaml
    │   ├── sample_ollama.yaml
    │   └── sample_groq.yaml
    ├── test_matching.py
    ├── test_storage.py
    ├── test_recorder.py
    ├── test_cassettes.py
    └── test_new_providers.py
```

---

## Error Reference

| Error | Cause | Fix |
|---|---|---|
| `CassetteNotFoundError` | Used `mode="playback"` but no cassette file exists | Run once with `mode="record"` first |
| `NoMatchFoundError` | Request doesn't match any interaction in the cassette | Re-record the cassette or add the interaction |
| `ProviderNotSupportedError` | Unknown provider name passed | Use one of: `openai`, `anthropic`, `gemini`, `ollama`, `groq` |

---

## Contributing

1. Fork the repo
2. Create a branch: `git checkout -b my-feature`
3. Make your changes and add tests
4. Run the test suite: `pytest`
5. Open a pull request

When adding a new provider, follow the pattern in `src/llmvcr/providers/openai_provider.py` — implement `serialize_request`, `serialize_response`, `deserialize_response`, and `make_patch`, then register the provider in `_patch.py`.

---

## License

MIT — see [LICENSE](LICENSE).