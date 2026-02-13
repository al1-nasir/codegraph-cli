# Command Reference

Complete reference for all CodeGraph CLI commands.

## Global Options

```bash
cg --help              # Show help message
cg --version           # Show version (coming soon)
```

---

## `cg index`

Index a project folder into semantic memory.

### Syntax

```bash
cg index PROJECT_PATH [OPTIONS]
```

### Arguments

- `PROJECT_PATH` - Path to the project directory to index (required)

### Options

- `--name, -n TEXT` - Custom name for the project memory (default: directory name)
- `--llm-model TEXT` - LLM model name for reasoning (default: `qwen2.5-coder:7b`)

### Examples

```bash
# Index current directory
cg index .

# Index with custom name
cg index /path/to/myproject --name MyApp

# Index with specific LLM model
cg index ~/projects/api --name API --llm-model codellama:7b
```

### Output

```
Indexed '/path/to/project' as project 'MyApp'.
Nodes: 245 | Edges: 512
```

---

## `cg list-projects`

List all indexed project memories.

### Syntax

```bash
cg list-projects
```

### Examples

```bash
cg list-projects
```

### Output

```
* MyApp          # * indicates currently loaded project
  OtherProject
  TestProject
```

---

## `cg load-project`

Switch to a different project memory.

### Syntax

```bash
cg load-project PROJECT_NAME
```

### Arguments

- `PROJECT_NAME` - Name of the project to load (required)

### Examples

```bash
cg load-project MyApp
```

### Output

```
Loaded project 'MyApp'.
```

---

## `cg current-project`

Show the currently loaded project.

### Syntax

```bash
cg current-project
```

### Examples

```bash
cg current-project
```

### Output

```
MyApp
```

Or if no project is loaded:

```
No project loaded
```

---

## `cg search`

Semantic search across the loaded project.

### Syntax

```bash
cg search QUERY [OPTIONS]
```

### Arguments

- `QUERY` - Search query (required)

### Options

- `--top-k INTEGER` - Maximum number of results (default: 5, range: 1-30)

### Examples

```bash
# Find database-related code
cg search "database connection"

# Find authentication functions
cg search "user authentication" --top-k 10

# Find error handling
cg search "exception handling"
```

### Output

```
[function] db.connect  score=0.842
  db/connection.py:15-32
  def connect(host: str, port: int) -> Connection:

[class] DatabaseManager  score=0.756
  db/manager.py:8-45
  class DatabaseManager:

[function] db.execute_query  score=0.689
  db/query.py:22-38
  def execute_query(sql: str, params: dict) -> List[dict]:
```

---

## `cg impact`

Analyze the impact of changing a symbol (function/class/module).

### Syntax

```bash
cg impact SYMBOL [OPTIONS]
```

### Arguments

- `SYMBOL` - Function, class, or module name to analyze (required)

### Options

- `--hops INTEGER` - Dependency traversal depth (default: 2, range: 1-6)
- `--show-graph / --no-graph` - Include ASCII graph (default: show)

### Examples

```bash
# Analyze impact of changing a function
cg impact process_payment

# Deep analysis with 3 hops
cg impact UserService --hops 3

# Without ASCII graph
cg impact calculate_total --no-graph
```

### Output

```
Root: processor.OrderProcessor.create_order
Impacted symbols:
- models.Order.total
- models.Order
- processor.UserProcessor.get_user

ASCII graph:
processor.OrderProcessor.create_order
  |- calls -> processor.UserProcessor.get_user
  |- calls -> models.Order
  processor.UserProcessor.get_user
  models.Order
    |- contains -> models.Order.total
    models.Order.total

Explanation:
Main risks:
- Changing create_order() may affect order validation logic
- Dependencies on User model could break if signature changes
- Order creation flow impacts downstream payment processing

Most likely breakpoints:
- UserProcessor.get_user() if user lookup logic changes
- Order model if constructor signature changes

Test recommendations:
- Add unit tests for order creation edge cases
- Test user validation scenarios
- Verify payment integration still works
```

---

## `cg graph`

Show ASCII dependency graph around a symbol.

### Syntax

```bash
cg graph SYMBOL [OPTIONS]
```

### Arguments

- `SYMBOL` - Function, class, or module to visualize (required)

### Options

- `--depth INTEGER` - Traversal depth (default: 2, range: 1-6)

### Examples

```bash
# Show dependencies of a class
cg graph UserProcessor

# Deep graph with 3 levels
cg graph PaymentService --depth 3
```

### Output

```
processor.UserProcessor (class)
  |-contains-> processor.UserProcessor.__init__
  |-contains-> processor.UserProcessor.create_user
  |-contains-> processor.UserProcessor.get_user
    |-calls-> utils.validate_email
    |-calls-> models.User
```

---

## `cg export-graph`

Export dependency graph to HTML or Graphviz DOT format.

### Syntax

```bash
cg export-graph [SYMBOL] [OPTIONS]
```

### Arguments

- `SYMBOL` - Optional symbol to focus on (exports full graph if omitted)

### Options

- `--format, -f TEXT` - Export format: `html` or `dot` (default: `html`)
- `--output, -o PATH` - Output file path (default: `{project}_graph.{format}`)

### Examples

```bash
# Export full project graph as HTML
cg export-graph --format html

# Export subgraph around a symbol
cg export-graph PaymentService --format html --output payment.html

# Export as Graphviz DOT for custom rendering
cg export-graph --format dot --output graph.dot
```

### Output

```
Exported graph to /path/to/output.html
```

You can then open the HTML file in a browser or process the DOT file with Graphviz:

```bash
# Render DOT file to PNG
dot -Tpng graph.dot -o graph.png
```

---

## `cg rag-context`

Retrieve semantic code context without analysis (useful for debugging RAG).

### Syntax

```bash
cg rag-context QUERY [OPTIONS]
```

### Arguments

- `QUERY` - Query to retrieve context for (required)

### Options

- `--top-k INTEGER` - Number of snippets to retrieve (default: 6, range: 1-30)

### Examples

```bash
# Get context about payment processing
cg rag-context "payment processing logic"

# Get more context snippets
cg rag-context "database queries" --top-k 10
```

### Output

```
[function] process_payment (payment/processor.py:45)
Score: 0.823
```python
def process_payment(order_id: int, amount: float) -> bool:
    """Process payment for an order."""
    # ... code snippet ...
```

[class] PaymentGateway (payment/gateway.py:12)
Score: 0.756
```python
class PaymentGateway:
    """Interface to payment provider."""
    # ... code snippet ...
```
```

---

## `cg unload-project`

Unload the currently active project (without deleting data).

### Syntax

```bash
cg unload-project
```

### Examples

```bash
cg unload-project
```

### Output

```
Unloaded active project.
```

---

## `cg delete-project`

Permanently delete a project memory.

### Syntax

```bash
cg delete-project PROJECT_NAME
```

### Arguments

- `PROJECT_NAME` - Name of project to delete (required)

### Examples

```bash
cg delete-project OldProject
```

### Output

```
Deleted project 'OldProject'.
```

**Warning**: This permanently deletes all indexed data for the project.

---

## `cg merge-projects`

Merge one project memory into another.

### Syntax

```bash
cg merge-projects SOURCE_PROJECT TARGET_PROJECT
```

### Arguments

- `SOURCE_PROJECT` - Project to merge from (required)
- `TARGET_PROJECT` - Project to merge into (required)

### Examples

```bash
# Merge Frontend into FullStack project
cg merge-projects Frontend FullStack
```

### Output

```
Merged 'Frontend' into 'FullStack'.
```

**Note**: The source project is not deleted after merging.

---

## Tips & Tricks

### Chaining Commands

```bash
# Index and immediately search
cg index ~/myproject --name MyProj && cg search "authentication"

# Switch project and analyze
cg load-project API && cg impact handle_request
```

### Using with Other Tools

```bash
# Export graph and render with Graphviz
cg export-graph --format dot --output graph.dot
dot -Tpng graph.dot -o graph.png

# Search and pipe to grep
cg search "database" | grep "connection"
```

### Project Naming Conventions

- Use descriptive names: `MyApp-Backend` instead of `proj1`
- Include version if tracking multiple: `API-v2`, `API-v3`
- Use consistent naming: `ProjectName-Component`

---

## See Also

- [Setup Guide](setup.md) - Installation and configuration
- [Workflows](workflows.md) - Common use cases and examples
- [Architecture](architecture.md) - How CodeGraph works internally
