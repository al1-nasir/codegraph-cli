# CodeGraph CLI — Cheat Sheet

## Quick Start
```bash
cg quickstart              # Setup and index in 30 seconds
cg init                    # Alias for quickstart
```

## Most Used Commands
```bash
cg search "query"          # Find code semantically
cg chat start              # Interactive Q&A about code
cg v2 generate "desc"      # Generate new code
cg v2 review <file>        # AI code review
```

## Aliases (shortcuts)
```bash
cg find "query"            # Same as cg search
cg ask                     # Same as cg chat start
cg gen "description"       # Same as cg v2 generate
cg fix <path>              # Same as cg v2 diagnose fix
```

## Project Management
```bash
cg index <path>            # Index a project
cg list-projects           # List all indexed projects
cg load-project <name>     # Switch active project
cg delete-project <name>   # Delete a project
cg tree                    # Show project structure
cg tree --full             # Show with functions/classes
```

## Code Analysis
```bash
cg search "query"          # Semantic code search
cg impact <symbol>         # Impact analysis
cg graph <symbol>          # Show dependency graph
cg rag-context "query"     # View RAG context
cg health dashboard        # Project health report
```

## Code Improvement
```bash
cg v2 review <file>              # AI code review
cg v2 review <file> --llm        # With LLM deep analysis
cg v2 refactor rename <old> <new>  # Rename symbol safely
cg v2 refactor extract-function <file> <start> <end> <name>
cg v2 diagnose check <path>      # Scan for errors
cg v2 diagnose fix <path>        # Auto-fix errors
cg v2 test unit <symbol>         # Generate unit tests
cg v2 test integration "flow"    # Generate integration tests
```

## Code Generation
```bash
cg v2 generate "description"            # Generate code
cg v2 generate "desc" --file ctx.py     # With context file
cg v2 generate "desc" --output out/     # Output to directory
cg v2 generate "desc" --preview         # Preview only
cg v2 generate "desc" --auto-apply      # Skip confirmation
```

## Workflows
```bash
cg review-and-fix <file>          # Review → diagnose → fix
cg review-and-fix <file> --apply  # Auto-apply fixes
cg full-analysis <symbol>         # Impact + graph + RAG
cg full-analysis <symbol> --export  # With HTML export
```

## Chat Mode
```bash
cg chat start              # Start chat session
cg chat start --crew       # Multi-agent mode
cg chat start --new        # Force new session
cg chat list               # List sessions
cg chat delete <id>        # Delete a session
cg chat export <id>        # Export conversation
```

### Chat Commands (inside chat)
- `/exit`     — Exit and save session
- `/clear`    — Clear conversation history
- `/new`      — Start fresh session
- `/help`     — Show available commands
- `/apply`    — Apply pending code proposal
- `/preview`  — Preview pending changes
- `/rollback` — Rollback a file (crew mode)
- `/backups`  — List file backups (crew mode)

## Watch Mode
```bash
cg watch                   # Watch current dir for changes
cg watch ./src             # Watch specific directory
cg watch --interval 5      # Set debounce interval
cg watch --full            # Full re-index on changes
```

## History & Undo
```bash
cg undo                    # Undo last change
cg undo --steps 3          # Undo multiple steps
cg redo                    # Redo undone change
cg history show            # Show change history
cg history clear           # Clear history
```

## Configuration
```bash
cg config setup                   # Interactive setup wizard
cg config set-llm <provider>      # Quick switch LLM
cg config set-llm groq -k KEY     # Set Groq with API key
cg config show-llm                # Show current LLM config
cg config unset-llm               # Reset LLM config
cg config set-embedding <model>   # Set embedding model
cg config show-embedding          # Show embedding config
cg config unset-embedding         # Reset to hash embeddings
```

## Debug Tools
```bash
cg debug-embed "text"      # Debug embedding output
cg debug-rag "query"       # Debug RAG retrieval
cg debug-context "query"   # Show LLM context assembly
```

## Graph Export
```bash
cg export-graph                    # Export full graph as HTML
cg export-graph <symbol>           # Focus on symbol
cg export-graph --format dot       # Export as Graphviz DOT
cg export-graph -o graph.html      # Specify output file
```

## Tips
- Use `cg <command> --help` for detailed options
- Watch mode: `cg watch` for auto-reindexing
- Health dashboard: `cg health dashboard` for quick overview
- Support NO_COLOR env var for plain output
