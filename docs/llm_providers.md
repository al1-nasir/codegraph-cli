# Multi-LLM Provider Configuration Guide

CodeGraph CLI now supports multiple LLM providers! You can use local Ollama or cloud APIs like Groq, OpenAI, and Anthropic.

---

## Supported Providers

1. **Ollama** (default) - Fully local, no API key needed
2. **Groq** - Fast cloud inference with generous free tier
3. **OpenAI** - GPT-4 and GPT-3.5-turbo
4. **Anthropic** - Claude 3.5 Sonnet and Opus

---

## Configuration Methods

### Method 1: Environment Variables (Recommended)

Set these environment variables to configure your preferred provider:

```bash
# Choose provider (ollama, groq, openai, anthropic)
export CODEGRAPH_LLM_PROVIDER="groq"

# Set API key for cloud providers
export CODEGRAPH_LLM_API_KEY="your-api-key-here"

# Optional: Set specific model
export CODEGRAPH_LLM_MODEL="llama-3.3-70b-versatile"
```

Then use CodeGraph normally:

```bash
cg index /path/to/project
cg impact my_function
```

### Method 2: CLI Flags

Pass provider settings directly in commands:

```bash
# Using Groq
cg index /path/to/project \
  --llm-provider groq \
  --llm-api-key "gsk_your_groq_api_key" \
  --llm-model "llama-3.3-70b-versatile"

# Using OpenAI
cg index /path/to/project \
  --llm-provider openai \
  --llm-api-key "sk-your_openai_api_key" \
  --llm-model "gpt-4"

# Using Anthropic
cg index /path/to/project \
  --llm-provider anthropic \
  --llm-api-key "sk-ant-your_anthropic_api_key" \
  --llm-model "claude-3-5-sonnet-20241022"
```

---

## Provider-Specific Setup

### Groq (Recommended for Cloud)

**Why Groq?**
- ⚡ Extremely fast inference
- 🆓 Generous free tier
- 🎯 Great for code analysis

**Setup:**

1. Get API key from [console.groq.com](https://console.groq.com)
2. Set environment variable:
   ```bash
   export CODEGRAPH_LLM_PROVIDER="groq"
   export CODEGRAPH_LLM_API_KEY="gsk_..."
   ```

**Recommended Models:**
- `llama-3.3-70b-versatile` (default, best balance)
- `mixtral-8x7b-32768` (longer context)
- `llama-3.1-70b-versatile` (alternative)

**Example:**
```bash
export CODEGRAPH_LLM_PROVIDER="groq"
export CODEGRAPH_LLM_API_KEY="gsk_abc123..."
export CODEGRAPH_LLM_MODEL="llama-3.3-70b-versatile"

cg index ~/my-project
cg impact process_payment
```

---

### Ollama (Default - Local)

**Why Ollama?**
- 🔒 Fully local, no API calls
- 🆓 Completely free
- 🔐 Maximum privacy

**Setup:**

1. Install Ollama:
   ```bash
   # Linux
   curl https://ollama.ai/install.sh | sh
   
   # macOS
   brew install ollama
   ```

2. Start Ollama:
   ```bash
   ollama serve
   ```

3. Pull a model:
   ```bash
   ollama pull qwen2.5-coder:7b
   ```

**Recommended Models:**
- `qwen2.5-coder:7b` (default, best for code)
- `codellama:7b` (alternative)
- `qwen2.5-coder:14b` (better quality, needs more RAM)

**Example:**
```bash
# No API key needed!
cg index ~/my-project --llm-provider ollama
```

---

### OpenAI

**Why OpenAI?**
- 🧠 Most capable models (GPT-4)
- 📚 Best for complex reasoning
- 💰 Pay-per-use pricing

**Setup:**

1. Get API key from [platform.openai.com](https://platform.openai.com)
2. Set environment variable:
   ```bash
   export CODEGRAPH_LLM_PROVIDER="openai"
   export CODEGRAPH_LLM_API_KEY="sk-..."
   ```

**Recommended Models:**
- `gpt-4` (default, best quality)
- `gpt-3.5-turbo` (faster, cheaper)
- `gpt-4-turbo` (latest)

**Example:**
```bash
export CODEGRAPH_LLM_PROVIDER="openai"
export CODEGRAPH_LLM_API_KEY="sk-proj-abc123..."
export CODEGRAPH_LLM_MODEL="gpt-4"

cg index ~/my-project
```

---

### Anthropic (Claude)

**Why Anthropic?**
- 🎯 Excellent at code analysis
- 📖 Very long context windows
- 🔍 Great reasoning capabilities

**Setup:**

1. Get API key from [console.anthropic.com](https://console.anthropic.com)
2. Set environment variable:
   ```bash
   export CODEGRAPH_LLM_PROVIDER="anthropic"
   export CODEGRAPH_LLM_API_KEY="sk-ant-..."
   ```

**Recommended Models:**
- `claude-3-5-sonnet-20241022` (default, best balance)
- `claude-3-opus-20240229` (highest quality)
- `claude-3-haiku-20240307` (fastest, cheapest)

**Example:**
```bash
export CODEGRAPH_LLM_PROVIDER="anthropic"
export CODEGRAPH_LLM_API_KEY="sk-ant-api03-abc123..."
export CODEGRAPH_LLM_MODEL="claude-3-5-sonnet-20241022"

cg index ~/my-project
```

---

## Quick Comparison

| Provider   | Cost      | Speed    | Privacy | Setup Difficulty | Best For                |
|------------|-----------|----------|---------|------------------|-------------------------|
| Ollama     | Free      | Medium   | ⭐⭐⭐⭐⭐ | Medium           | Local-first, privacy    |
| Groq       | Free tier | ⭐⭐⭐⭐⭐ | ⭐⭐⭐   | Easy             | Fast cloud inference    |
| OpenAI     | Pay       | Fast     | ⭐⭐⭐   | Easy             | Best quality            |
| Anthropic  | Pay       | Fast     | ⭐⭐⭐   | Easy             | Code analysis, reasoning|

---

## Fallback Behavior

If the LLM provider is unavailable (no API key, network error, etc.), CodeGraph automatically falls back to deterministic explanations:

```
LLM provider 'groq' was unavailable; returning a deterministic fallback summary.
Context excerpt:
[Code context...]

Recommendation:
- Inspect the listed call/dependency chain
- Run unit tests around impacted functions
- Validate side effects at integration boundaries
```

This ensures CodeGraph **always works**, even without LLM access.

---

## Complete Example Workflows

### Workflow 1: Start with Ollama, Switch to Groq

```bash
# 1. Start with local Ollama
ollama serve
ollama pull qwen2.5-coder:7b
cg index ~/my-project

# 2. Later, switch to Groq for faster analysis
export CODEGRAPH_LLM_PROVIDER="groq"
export CODEGRAPH_LLM_API_KEY="gsk_..."
cg impact my_function  # Uses Groq now
```

### Workflow 2: Use Different Providers for Different Projects

```bash
# Project 1: Use local Ollama (private code)
cg index ~/private-project --llm-provider ollama

# Project 2: Use Groq (open source, need speed)
cg index ~/open-source-project \
  --llm-provider groq \
  --llm-api-key "gsk_..."
```

### Workflow 3: Compare Provider Outputs

```bash
# Analyze with Ollama
export CODEGRAPH_LLM_PROVIDER="ollama"
cg impact process_payment > ollama_analysis.txt

# Analyze with Groq
export CODEGRAPH_LLM_PROVIDER="groq"
export CODEGRAPH_LLM_API_KEY="gsk_..."
cg impact process_payment > groq_analysis.txt

# Compare results
diff ollama_analysis.txt groq_analysis.txt
```

---

## Troubleshooting

### "LLM provider unavailable" message

**For Ollama:**
- Check if Ollama is running: `curl http://127.0.0.1:11434/api/tags`
- Start Ollama: `ollama serve`
- Verify model is pulled: `ollama list`

**For Cloud Providers:**
- Verify API key is set: `echo $CODEGRAPH_LLM_API_KEY`
- Check API key is valid (test with curl)
- Ensure you have credits/quota remaining

### Wrong model being used

Check your environment variables:
```bash
echo $CODEGRAPH_LLM_PROVIDER
echo $CODEGRAPH_LLM_MODEL
```

Override with CLI flags:
```bash
cg index ~/project --llm-provider groq --llm-model "llama-3.3-70b-versatile"
```

### API rate limits

**Groq free tier limits:**
- 30 requests/minute
- 6,000 requests/day

**Solutions:**
- Use Ollama for unlimited local inference
- Upgrade to paid tier
- Add delays between commands

---

## Best Practices

### 1. Use Environment Variables

Store configuration in your shell profile:

```bash
# ~/.bashrc or ~/.zshrc
export CODEGRAPH_LLM_PROVIDER="groq"
export CODEGRAPH_LLM_API_KEY="gsk_..."
export CODEGRAPH_LLM_MODEL="llama-3.3-70b-versatile"
```

### 2. Keep API Keys Secure

```bash
# Use a secrets manager
export CODEGRAPH_LLM_API_KEY=$(cat ~/.secrets/groq_api_key)

# Or use direnv for per-project configuration
# .envrc
export CODEGRAPH_LLM_PROVIDER="groq"
export CODEGRAPH_LLM_API_KEY="gsk_..."
```

### 3. Choose Provider Based on Use Case

- **Private code**: Use Ollama (local)
- **Fast iteration**: Use Groq (fastest)
- **Best quality**: Use OpenAI GPT-4
- **Long context**: Use Anthropic Claude

### 4. Test Fallback Behavior

Intentionally trigger fallback to ensure it works:

```bash
# Set invalid API key
export CODEGRAPH_LLM_API_KEY="invalid"
cg impact my_function
# Should see fallback explanation
```

---

## See Also

- [Setup Guide](setup.md) - Installation and configuration
- [Command Reference](commands.md) - CLI command documentation
- [Architecture](architecture.md) - How LLM integration works
