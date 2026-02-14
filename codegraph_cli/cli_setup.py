"""Interactive setup wizard for CodeGraph CLI."""

from __future__ import annotations

import sys
from typing import Optional

import typer

from . import config_manager
from .embeddings import EMBEDDING_MODELS

app = typer.Typer(help="Setup wizard for LLM provider configuration")


# Provider model options
PROVIDER_MODELS = {
    "groq": [
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ],
    "openai": [
        "gpt-4",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
    ],
    "anthropic": [
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
    ],
    "gemini": [
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-1.0-pro",
    ],
    "openrouter": [
        "google/gemini-2.0-flash-exp:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "deepseek/deepseek-chat-v3-0324:free",
        "qwen/qwen3-235b-a22b:free",
        "stepfun/step-3.5-flash:free",
        "anthropic/claude-sonnet-4",
        "openai/gpt-4o",
    ],
}


def print_header():
    """Print setup wizard header."""
    typer.echo("")
    typer.echo(typer.style("╭──────────────────────────────────────────────╮", fg=typer.colors.CYAN))
    typer.echo(typer.style("│", fg=typer.colors.CYAN) + typer.style("   🔧 CodeGraph LLM Setup Wizard              ", bold=True) + typer.style("│", fg=typer.colors.CYAN))
    typer.echo(typer.style("╰──────────────────────────────────────────────╯", fg=typer.colors.CYAN))


def print_success(message: str):
    """Print success message."""
    typer.echo(typer.style(f"✅ {message}", fg=typer.colors.GREEN))


def print_error(message: str):
    """Print error message."""
    typer.echo(typer.style(f"❌ {message}", fg=typer.colors.RED), err=True)


def print_info(message: str):
    """Print info message."""
    typer.echo(typer.style(f"ℹ️  {message}", fg=typer.colors.BLUE))


# All supported providers for quick lookup
ALL_PROVIDERS = ["ollama", "groq", "openai", "anthropic", "gemini", "openrouter"]


def select_provider() -> str:
    """Interactive provider selection.
    
    Returns:
        Selected provider name
    """
    typer.echo("\nChoose your LLM provider:")
    typer.echo("  1) Ollama      (local, free)")
    typer.echo("  2) Groq        (cloud, fast, free tier)")
    typer.echo("  3) OpenAI      (cloud, paid)")
    typer.echo("  4) Anthropic   (cloud, paid)")
    typer.echo("  5) Gemini      (cloud, free tier available)")
    typer.echo("  6) OpenRouter  (cloud, multi-model, free tier available)")
    
    provider_map = {
        "1": "ollama",
        "2": "groq",
        "3": "openai",
        "4": "anthropic",
        "5": "gemini",
        "6": "openrouter",
    }
    
    while True:
        choice = typer.prompt("\nEnter choice [1-6]", type=str)
        if choice in provider_map:
            return provider_map[choice]
        print_error("Invalid choice. Please enter 1-6.")


def setup_ollama() -> tuple[str, str, str]:
    """Setup Ollama provider.
    
    Returns:
        Tuple of (provider, model, endpoint)
    """
    typer.echo("\n" + typer.style("Setting up Ollama", bold=True))
    typer.echo("━" * 50)
    
    # Check if Ollama is running
    endpoint = typer.prompt("Ollama endpoint", default="http://127.0.0.1:11434")
    
    typer.echo("\n⏳ Checking Ollama connection...")
    if not config_manager.validate_ollama_connection(endpoint):
        print_error("Cannot connect to Ollama!")
        print_info("Make sure Ollama is running: https://ollama.ai")
        print_info("Start Ollama and run this setup again.")
        raise typer.Exit(code=1)
    
    print_success("Connected to Ollama")
    
    # Fetch available models
    typer.echo("\n⏳ Fetching available models...")
    models = config_manager.get_ollama_models(endpoint)
    
    if not models:
        print_error("No models found!")
        print_info("Pull a model first: ollama pull qwen2.5-coder:7b")
        raise typer.Exit(code=1)
    
    # Display models
    typer.echo("\nAvailable models:")
    for i, model in enumerate(models, 1):
        typer.echo(f"  {i}) {model}")
    
    # Select model
    while True:
        choice = typer.prompt(f"\nSelect model [1-{len(models)}]", type=int)
        if 1 <= choice <= len(models):
            selected_model = models[choice - 1]
            break
        print_error(f"Invalid choice. Please enter a number between 1 and {len(models)}.")
    
    return "ollama", selected_model, endpoint


def setup_cloud_provider(provider: str) -> tuple[str, str, str]:
    """Setup cloud provider (Groq, OpenAI, Anthropic, Gemini, OpenRouter).
    
    Args:
        provider: Provider name
    
    Returns:
        Tuple of (provider, model, api_key)
    """
    provider_display = {
        "openrouter": "OpenRouter",
        "openai": "OpenAI",
    }.get(provider, provider.title())
    
    typer.echo(f"\n" + typer.style(f"Setting up {provider_display}", bold=True))
    typer.echo("━" * 50)
    
    # Provider-specific hints
    if provider == "gemini":
        print_info("Get your Gemini API key at: https://aistudio.google.com/apikey")
    elif provider == "openrouter":
        print_info("Get your OpenRouter API key at: https://openrouter.ai/keys")
        print_info("Many free models available! Look for models ending with ':free'")
    
    # Get API key
    api_key = typer.prompt(f"\nEnter your {provider_display} API key", hide_input=True)
    
    if not api_key.strip():
        print_error("API key cannot be empty!")
        raise typer.Exit(code=1)
    
    # Display available models
    models = PROVIDER_MODELS.get(provider, [])
    typer.echo("\nAvailable models:")
    for i, model in enumerate(models, 1):
        typer.echo(f"  {i}) {model}")
    
    # Select model
    while True:
        choice = typer.prompt(f"\nSelect model [1-{len(models)}] or enter custom model name", type=str)
        
        # Check if it's a number
        try:
            idx = int(choice)
            if 1 <= idx <= len(models):
                selected_model = models[idx - 1]
                break
            print_error(f"Invalid choice. Please enter a number between 1 and {len(models)}.")
        except ValueError:
            # Custom model name
            selected_model = choice.strip()
            if selected_model:
                break
            print_error("Model name cannot be empty!")
    
    # Validate API key
    typer.echo("\n⏳ Validating API key...")
    is_valid, error_msg = config_manager.validate_api_key(provider, api_key, selected_model)
    
    if not is_valid:
        print_error(f"API key validation failed: {error_msg}")
        print_info("Please check your API key and try again.")
        raise typer.Exit(code=1)
    
    print_success("API key validated successfully")
    
    return provider, selected_model, api_key


def display_summary(provider: str, model: str, api_key: str = "", endpoint: str = ""):
    """Display configuration summary.
    
    Args:
        provider: Provider name
        model: Model name
        api_key: API key (masked for display)
        endpoint: Endpoint URL
    """
    typer.echo("\n" + typer.style("✅ Configuration Summary", bold=True, fg=typer.colors.GREEN))
    typer.echo("━" * 50)
    typer.echo(f"Provider: {typer.style(provider, fg=typer.colors.CYAN)}")
    typer.echo(f"Model: {typer.style(model, fg=typer.colors.CYAN)}")
    
    if api_key:
        masked_key = api_key[:8] + "*" * (len(api_key) - 8)
        typer.echo(f"API Key: {masked_key}")
    
    if endpoint:
        typer.echo(f"Endpoint: {endpoint}")
    
    typer.echo("")


def setup():
    """Interactive setup wizard for LLM provider configuration."""
    print_header()
    
    # Select provider
    provider = select_provider()
    
    # Provider-specific setup
    api_key = ""
    endpoint = ""
    
    if provider == "ollama":
        provider, model, endpoint = setup_ollama()
    elif provider == "openrouter":
        provider, model, api_key = setup_cloud_provider(provider)
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
    else:
        provider, model, api_key = setup_cloud_provider(provider)
    
    # Display summary
    display_summary(provider, model, api_key, endpoint)
    
    # Confirm save
    save = typer.confirm(f"Save to {config_manager.CONFIG_FILE}?", default=True)
    
    if not save:
        print_info("Configuration not saved.")
        raise typer.Exit(code=0)
    
    # Save configuration
    success = config_manager.save_config(provider, model, api_key, endpoint)
    
    if success:
        print_success(f"Configuration saved to {config_manager.CONFIG_FILE}")
        print_info(f"You can now use 'cg' commands without specifying provider options!")
        typer.echo("\nExample commands:")
        typer.echo("  cg index ./my-project")
        typer.echo("  cg search 'authentication logic'")
        typer.echo("  cg impact main")
    else:
        print_error("Failed to save configuration!")
        raise typer.Exit(code=1)

    # Offer embedding setup
    typer.echo("")
    setup_emb = typer.confirm("Configure embedding model for semantic search?", default=True)
    if setup_emb:
        _interactive_embedding_setup()


def set_llm(
    provider: str = typer.Argument(..., help="LLM provider: ollama, groq, openai, anthropic, gemini, openrouter"),
    model: str = typer.Option(None, "--model", "-m", help="Model name (uses provider default if not set)."),
    api_key: str = typer.Option(None, "--api-key", "-k", help="API key for cloud providers."),
    endpoint: str = typer.Option(None, "--endpoint", "-e", help="Custom endpoint URL."),
    no_validate: bool = typer.Option(False, "--no-validate", help="Skip API key validation."),
):
    """Quickly switch LLM provider without full setup wizard.
    
    Examples:
        cg set-llm groq -k YOUR_API_KEY
        cg set-llm gemini -k YOUR_API_KEY -m gemini-2.0-flash
        cg set-llm openrouter -k YOUR_API_KEY -m google/gemini-2.0-flash-exp:free
        cg set-llm ollama -m qwen2.5-coder:7b
    """
    provider = provider.lower().strip()
    
    if provider not in ALL_PROVIDERS:
        print_error(f"Unknown provider '{provider}'. Choose from: {', '.join(ALL_PROVIDERS)}")
        raise typer.Exit(code=1)
    
    # Load current config as base
    current = config_manager.load_config()
    
    # Get defaults for the chosen provider
    defaults = config_manager.get_provider_config(provider)
    
    # Resolve model
    resolved_model = model or defaults.get("model", "")
    
    # Resolve endpoint
    if provider == "openrouter" and not endpoint:
        resolved_endpoint = "https://openrouter.ai/api/v1/chat/completions"
    elif provider == "ollama" and not endpoint:
        resolved_endpoint = defaults.get("endpoint", "http://127.0.0.1:11434/api/generate")
    else:
        resolved_endpoint = endpoint or ""
    
    # Resolve API key
    resolved_api_key = api_key or ""
    
    # Cloud providers need an API key
    if provider not in ("ollama",) and not resolved_api_key:
        # Check if there's one in current config for same provider
        if current.get("provider") == provider and current.get("api_key"):
            resolved_api_key = current["api_key"]
            print_info(f"Reusing existing API key for {provider}")
        else:
            resolved_api_key = typer.prompt(f"Enter your {provider} API key", hide_input=True)
    
    # Validate if needed
    if not no_validate and provider not in ("ollama",) and resolved_api_key:
        typer.echo("⏳ Validating API key...")
        is_valid, error_msg = config_manager.validate_api_key(provider, resolved_api_key, resolved_model)
        if not is_valid:
            print_error(f"Validation failed: {error_msg}")
            force = typer.confirm("Save anyway?", default=False)
            if not force:
                raise typer.Exit(code=1)
    elif provider == "ollama":
        typer.echo("⏳ Checking Ollama connection...")
        base_ep = resolved_endpoint.replace("/api/generate", "")
        if not config_manager.validate_ollama_connection(base_ep):
            print_error("Cannot connect to Ollama!")
            force = typer.confirm("Save anyway?", default=False)
            if not force:
                raise typer.Exit(code=1)
    
    # Save
    success = config_manager.save_config(provider, resolved_model, resolved_api_key, resolved_endpoint)
    
    if success:
        print_success(f"LLM provider set to: {provider}")
        typer.echo(f"  Provider: {typer.style(provider, fg=typer.colors.CYAN)}")
        typer.echo(f"  Model:    {typer.style(resolved_model, fg=typer.colors.CYAN)}")
        if resolved_endpoint:
            typer.echo(f"  Endpoint: {resolved_endpoint}")
    else:
        print_error("Failed to save configuration!")
        raise typer.Exit(code=1)


def unset_llm():
    """Reset LLM configuration to defaults (removes API keys and provider settings)."""
    typer.echo("\n🔧 " + typer.style("Unset LLM Configuration", bold=True, fg=typer.colors.CYAN))
    typer.echo("━" * 50)
    
    if not config_manager.CONFIG_FILE.exists():
        print_info("No LLM configuration found. Nothing to unset.")
        raise typer.Exit(code=0)
    
    # Show current config
    current = config_manager.load_config()
    typer.echo(f"\nCurrent provider: {typer.style(current.get('provider', 'none'), fg=typer.colors.YELLOW)}")
    typer.echo(f"Current model:    {typer.style(current.get('model', 'none'), fg=typer.colors.YELLOW)}")
    if current.get('api_key'):
        masked = current['api_key'][:8] + '****'
        typer.echo(f"API key:          {masked}")
    
    typer.echo("")
    typer.echo("Choose what to do:")
    typer.echo("  1) Reset to Ollama defaults (remove cloud keys)")
    typer.echo("  2) Delete entire config file")
    typer.echo("  3) Cancel")
    
    choice = typer.prompt("\nEnter choice [1-3]", type=str)
    
    if choice == "1":
        success = config_manager.save_config("ollama", "qwen2.5-coder:7b", "", "http://127.0.0.1:11434/api/generate")
        if success:
            print_success("Configuration reset to Ollama defaults.")
            print_info("API keys have been removed.")
        else:
            print_error("Failed to reset configuration!")
            raise typer.Exit(code=1)
    
    elif choice == "2":
        confirm = typer.confirm("Are you sure you want to delete the config file?", default=False)
        if confirm:
            try:
                config_manager.CONFIG_FILE.unlink()
                print_success(f"Deleted {config_manager.CONFIG_FILE}")
                print_info("Will use Ollama defaults on next run.")
            except OSError as e:
                print_error(f"Failed to delete config: {e}")
                raise typer.Exit(code=1)
        else:
            print_info("Cancelled.")
    
    else:
        print_info("Cancelled.")


def show_llm():
    """Show current LLM provider configuration."""
    typer.echo("")
    typer.echo(typer.style("╭──────────────────────────────────────────────╮", fg=typer.colors.CYAN))
    typer.echo(typer.style("│", fg=typer.colors.CYAN) + typer.style("   🔍 LLM Configuration                       ", bold=True) + typer.style("│", fg=typer.colors.CYAN))
    typer.echo(typer.style("╰──────────────────────────────────────────────╯", fg=typer.colors.CYAN))
    
    exists = config_manager.CONFIG_FILE.exists()
    cfg = config_manager.load_config()
    
    provider = cfg.get("provider", "ollama")
    model = cfg.get("model", "qwen2.5-coder:7b")
    endpoint = cfg.get("endpoint", "")
    api_key = cfg.get("api_key", "")
    
    # Provider badge colors
    provider_color = {
        "ollama": typer.colors.GREEN,
        "groq": typer.colors.YELLOW,
        "openai": typer.colors.CYAN,
        "anthropic": typer.colors.MAGENTA,
        "gemini": typer.colors.BLUE,
        "openrouter": typer.colors.BRIGHT_CYAN,
    }.get(provider, typer.colors.WHITE)
    
    typer.echo(f"  Provider  {typer.style(f' {provider.upper()} ', bg=provider_color, fg=typer.colors.WHITE, bold=True)}")
    typer.echo(f"  Model     {typer.style(model, fg=typer.colors.WHITE, bold=True)}")
    if endpoint:
        typer.echo(f"  Endpoint  {typer.style(endpoint, dim=True)}")
    if api_key:
        masked = api_key[:8] + '•' * min(len(api_key) - 8, 16)
        typer.echo(f"  API Key   {masked}")
    else:
        typer.echo(f"  API Key   {typer.style('(not set)', dim=True)}")
    typer.echo(f"  Config    {typer.style(str(config_manager.CONFIG_FILE), dim=True)}")
    
    typer.echo("")
    typer.echo(typer.style("  Quick Commands", bold=True))
    typer.echo(typer.style("  ─────────────────────────────────────────", dim=True))
    typer.echo(f"  {typer.style('cg setup', fg=typer.colors.YELLOW)}              Full interactive wizard")
    typer.echo(f"  {typer.style('cg set-llm <name>', fg=typer.colors.YELLOW)}     Quick switch provider")
    typer.echo(f"  {typer.style('cg unset-llm', fg=typer.colors.YELLOW)}          Reset / clear config")
    typer.echo("")


# ===================================================================
# Embedding model commands
# ===================================================================

def _interactive_embedding_setup():
    """Interactive embedding model picker (called from setup wizard)."""
    typer.echo("")
    typer.echo(typer.style("╭──────────────────────────────────────────────╮", fg=typer.colors.CYAN))
    typer.echo(typer.style("│", fg=typer.colors.CYAN) + typer.style("   Embedding Model Setup                       ", bold=True) + typer.style("│", fg=typer.colors.CYAN))
    typer.echo(typer.style("╰──────────────────────────────────────────────╯", fg=typer.colors.CYAN))
    typer.echo("")
    typer.echo("Choose an embedding model for semantic code search:")
    typer.echo("Larger models give better results but need more disk/RAM.\n")

    # List models with numbers
    model_keys = list(EMBEDDING_MODELS.keys())
    for i, key in enumerate(model_keys, 1):
        spec = EMBEDDING_MODELS[key]
        name_col = f"{key}".ljust(12)
        size_col = f"({spec['size']})".ljust(14)
        desc = spec["description"]
        typer.echo(f"  {i}) {name_col} {size_col} {desc}")

    typer.echo("")

    while True:
        choice = typer.prompt(f"Enter choice [1-{len(model_keys)}]", type=str)
        try:
            idx = int(choice)
            if 1 <= idx <= len(model_keys):
                selected = model_keys[idx - 1]
                break
        except ValueError:
            # Accept model key directly
            if choice.strip() in model_keys:
                selected = choice.strip()
                break
        print_error(f"Invalid choice. Enter 1-{len(model_keys)} or a model key.")

    spec = EMBEDDING_MODELS[selected]

    if selected != "hash":
        typer.echo(f"\n  Model:    {typer.style(spec['name'], fg=typer.colors.CYAN)}")
        typer.echo(f"  Download: {typer.style(spec['size'], fg=typer.colors.YELLOW)}")
        typer.echo(f"  Dim:      {spec['dim']}")
        print_info("Requires: pip install codegraph-cli[embeddings]")
    else:
        typer.echo(f"\n  Model: {typer.style('Hash Embedding (zero-dependency)', fg=typer.colors.CYAN)}")
        print_info("No download needed, but no semantic understanding.")

    success = config_manager.save_embedding_config(selected)
    if success:
        print_success(f"Embedding model set to: {selected}")
        if selected != "hash":
            print_info(f"Model will be downloaded on first use (~{spec['size']}).")
            print_info("Re-index your project after changing embeddings: cg index <path>")
    else:
        print_error("Failed to save embedding config!")


def set_embedding(
    model: str = typer.Argument(
        ...,
        help="Embedding model key: qodo-1.5b, jina-code, bge-base, minilm, hash",
    ),
):
    """Set the embedding model for semantic code search.

    Available models (smallest to largest):

        hash        0 bytes    No download, keyword-level only
        minilm      ~80 MB     Tiny, fast, decent quality
        bge-base    ~440 MB    Solid general-purpose
        jina-code   ~550 MB    Code-aware, good quality
        qodo-1.5b   ~6.2 GB   Best quality, code-optimized

    Examples:
        cg set-embedding minilm
        cg set-embedding jina-code
        cg set-embedding hash
    """
    model = model.lower().strip()

    if model not in EMBEDDING_MODELS:
        print_error(
            f"Unknown model '{model}'. "
            f"Choose from: {', '.join(EMBEDDING_MODELS.keys())}"
        )
        raise typer.Exit(code=1)

    spec = EMBEDDING_MODELS[model]
    success = config_manager.save_embedding_config(model)

    if success:
        print_success(f"Embedding model set to: {model}")
        typer.echo(f"  Name: {typer.style(spec['name'], fg=typer.colors.CYAN)}")
        typer.echo(f"  Dim:  {spec['dim']}")
        if model != "hash":
            typer.echo(f"  Size: {spec['size']} (downloaded on first use)")
            print_info("Re-index your project after changing: cg index <path>")
    else:
        print_error("Failed to save configuration!")
        raise typer.Exit(code=1)


def unset_embedding():
    """Reset embedding model to default (hash — no download)."""
    success = config_manager.clear_embedding_config()
    if success:
        print_success("Embedding model reset to default (hash).")
        print_info("No neural model will be used. Re-index to apply.")
    else:
        print_error("Failed to reset embedding config!")
        raise typer.Exit(code=1)


def show_embedding():
    """Show current embedding model configuration."""
    typer.echo("")
    typer.echo(typer.style("╭──────────────────────────────────────────────╮", fg=typer.colors.CYAN))
    typer.echo(typer.style("│", fg=typer.colors.CYAN) + typer.style("   Embedding Configuration                     ", bold=True) + typer.style("│", fg=typer.colors.CYAN))
    typer.echo(typer.style("╰──────────────────────────────────────────────╯", fg=typer.colors.CYAN))

    emb_cfg = config_manager.load_embedding_config()
    current_key = emb_cfg.get("model", "hash")
    spec = EMBEDDING_MODELS.get(current_key)

    if spec is None:
        typer.echo(f"  Model   {typer.style(current_key, fg=typer.colors.RED)} (unknown)")
    else:
        typer.echo(f"  Model   {typer.style(f' {current_key} ', bg=typer.colors.CYAN, fg=typer.colors.WHITE, bold=True)}")
        typer.echo(f"  Name    {typer.style(spec['name'], bold=True)}")
        typer.echo(f"  Dim     {spec['dim']}")
        typer.echo(f"  Size    {spec['size']}")
        typer.echo(f"  Desc    {spec['description']}")

    typer.echo("")
    typer.echo(typer.style("  Available Models", bold=True))
    typer.echo(typer.style("  ─────────────────────────────────────────", dim=True))
    for key, s in EMBEDDING_MODELS.items():
        marker = typer.style(" *", fg=typer.colors.GREEN) if key == current_key else "  "
        typer.echo(f"  {marker} {key.ljust(12)} {s['size'].ljust(12)} {s['description']}")

    typer.echo("")
    typer.echo(typer.style("  Quick Commands", bold=True))
    typer.echo(typer.style("  ─────────────────────────────────────────", dim=True))
    typer.echo(f"  {typer.style('cg set-embedding <model>', fg=typer.colors.YELLOW)}   Switch model")
    typer.echo(f"  {typer.style('cg unset-embedding', fg=typer.colors.YELLOW)}         Reset to hash")
    typer.echo("")


if __name__ == "__main__":
    app()
