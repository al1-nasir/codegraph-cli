# Common Workflows

Real-world examples of using CodeGraph CLI for common development tasks.

---

## Workflow 1: Exploring a New Codebase

**Scenario**: You've just joined a team and need to understand an unfamiliar codebase.

### Steps

```bash
# 1. Index the project
cg index /path/to/new/project --name NewProject

# 2. Get an overview - search for main entry points
cg search "main function"
cg search "application startup"

# 3. Find key components
cg search "database"
cg search "API endpoints"
cg search "authentication"

# 4. Understand a specific module
cg graph DatabaseManager --depth 3

# 5. See what depends on critical functions
cg impact authenticate_user --hops 3
```

### Expected Outcome

- Understand project structure
- Identify key components and their relationships
- Know which parts of the code are most interconnected

---

## Workflow 2: Impact Analysis Before Refactoring

**Scenario**: You need to refactor a function but want to understand what might break.

### Steps

```bash
# 1. Load the project
cg load-project MyApp

# 2. Analyze impact of the function you want to change
cg impact calculate_discount --hops 2

# 3. Review the ASCII graph to see dependencies
# (automatically shown with impact command)

# 4. Search for similar patterns that might also need updating
cg search "discount calculation"

# 5. Export a visual graph for documentation
cg export-graph calculate_discount --format html --output discount_impact.html
```

### Expected Outcome

- Know exactly which functions/classes depend on your target
- Understand the ripple effects of your changes
- Have a visual reference for code review discussions

---

## Workflow 3: Finding Code Duplication

**Scenario**: You suspect there's duplicated logic across the codebase.

### Steps

```bash
# 1. Search for the functionality
cg search "email validation"

# 2. Review all matches - look for similar implementations
# The search results show file paths and code snippets

# 3. For each match, see its dependencies
cg graph validate_email --depth 1
cg graph check_email_format --depth 1

# 4. Decide which implementation to keep and analyze impact
cg impact validate_email --hops 2
```

### Expected Outcome

- Find all implementations of similar functionality
- Understand which one is most widely used
- Plan consolidation safely

---

## Workflow 4: Onboarding Documentation

**Scenario**: Create visual documentation for new team members.

### Steps

```bash
# 1. Export the full project graph
cg export-graph --format html --output docs/full_architecture.html

# 2. Export key subsystems
cg export-graph AuthenticationService --format html --output docs/auth_system.html
cg export-graph PaymentProcessor --format html --output docs/payment_system.html

# 3. Create a DOT file for custom styling
cg export-graph --format dot --output docs/architecture.dot

# 4. Render with Graphviz
dot -Tpng docs/architecture.dot -o docs/architecture.png
```

### Expected Outcome

- Interactive HTML graphs for exploration
- PNG diagrams for documentation
- Visual aids for onboarding presentations

---

## Workflow 5: Debugging a Bug

**Scenario**: A bug was reported in `process_order` and you need to trace the issue.

### Steps

```bash
# 1. Find the function
cg search "process_order"

# 2. See what it calls (potential bug sources)
cg graph process_order --depth 2

# 3. Check what calls it (to understand the context)
# Look at the ASCII graph output - shows both directions

# 4. Get related code context for analysis
cg rag-context "order processing validation"

# 5. Analyze impact if you need to fix it
cg impact process_order --hops 2
```

### Expected Outcome

- Understand the function's dependencies
- See the full call chain
- Know what to test after fixing

---

## Workflow 6: Multi-Project Comparison

**Scenario**: You have a monorepo with multiple services and want to compare them.

### Steps

```bash
# 1. Index each service separately
cg index services/auth --name AuthService
cg index services/payment --name PaymentService
cg index services/notification --name NotificationService

# 2. Switch between projects to compare
cg load-project AuthService
cg search "database connection"

cg load-project PaymentService
cg search "database connection"

# 3. Optionally merge related services
cg merge-projects AuthService FullBackend
cg merge-projects PaymentService FullBackend

# 4. Analyze the combined system
cg load-project FullBackend
cg impact DatabaseConnection --hops 3
```

### Expected Outcome

- Compare implementations across services
- Find inconsistencies or duplication
- Understand cross-service dependencies

---

## Workflow 7: Code Review Preparation

**Scenario**: You're about to submit a PR that changes several functions.

### Steps

```bash
# 1. For each changed function, analyze impact
cg impact update_user_profile --hops 2
cg impact validate_input --hops 2

# 2. Export graphs to include in PR description
cg export-graph update_user_profile --format html --output pr_impact.html

# 3. Search for related code that reviewers should check
cg search "user profile"

# 4. Get context for the PR description
cg rag-context "user profile management"
```

### Expected Outcome

- Comprehensive impact analysis for reviewers
- Visual aids showing affected code
- Context for better code review discussions

---

## Workflow 8: Dependency Audit

**Scenario**: You need to understand what depends on a third-party library you want to upgrade.

### Steps

```bash
# 1. Search for imports of the library
cg search "import requests"
cg search "from requests"

# 2. For each usage, check the impact
cg impact make_api_call --hops 3

# 3. Export a full graph to see all connections
cg export-graph --format dot --output dependency_audit.dot

# 4. Render and review
dot -Tpng dependency_audit.dot -o dependency_audit.png
```

### Expected Outcome

- Complete list of files using the library
- Understanding of how deeply integrated it is
- Risk assessment for the upgrade

---

## Workflow 9: Incremental Learning

**Scenario**: Learn the codebase one module at a time.

### Steps

```bash
# Day 1: Authentication
cg search "authentication"
cg graph AuthService --depth 2
cg impact authenticate_user --hops 2

# Day 2: Database Layer
cg search "database"
cg graph DatabaseManager --depth 3
cg impact execute_query --hops 2

# Day 3: API Layer
cg search "API endpoint"
cg graph APIRouter --depth 2
cg impact handle_request --hops 3

# Create a learning journal
cg export-graph AuthService --format html --output learning/day1_auth.html
cg export-graph DatabaseManager --format html --output learning/day2_db.html
```

### Expected Outcome

- Structured learning path
- Visual references for each module
- Growing understanding of the system

---

## Workflow 10: Refactoring Planning

**Scenario**: Plan a major refactoring to extract a service.

### Steps

```bash
# 1. Identify all functions related to the service
cg search "email sending"
cg search "notification"

# 2. For each function, analyze dependencies
cg impact send_email --hops 3
cg impact queue_notification --hops 3

# 3. Export the subgraph
cg export-graph send_email --format html --output refactor_plan.html

# 4. Document what needs to move
# Review the HTML graph and mark functions to extract

# 5. After refactoring, index the new service
cg index services/email --name EmailService

# 6. Compare before and after
cg load-project EmailService
cg graph EmailSender --depth 2
```

### Expected Outcome

- Clear understanding of what to extract
- Dependency map for the refactoring
- Validation that the new service is properly isolated

---

## Tips for Effective Workflows

### Use Descriptive Project Names

```bash
# Good
cg index ~/projects/ecommerce --name ECommerce-Backend-v2

# Less helpful
cg index ~/projects/ecommerce --name proj1
```

### Combine with Git Workflows

```bash
# Before making changes
git checkout -b feature/refactor-auth
cg impact AuthService --hops 3 > impact_analysis.txt
git add impact_analysis.txt
git commit -m "Document impact analysis for auth refactoring"

# Make changes...

# After changes
cg index . --name ECommerce-Backend-v2-updated
# Compare old vs new if needed
```

### Regular Re-indexing

```bash
# After pulling latest changes
git pull origin main
cg index . --name MyProject  # Re-index to update graph

# Or set up a git hook
# .git/hooks/post-merge:
#!/bin/bash
cg index . --name MyProject
```

### Export for Documentation

```bash
# Generate architecture docs automatically
cg export-graph --format html --output docs/architecture.html
cg export-graph AuthService --format html --output docs/auth.html
cg export-graph PaymentService --format html --output docs/payment.html

# Commit to repo
git add docs/*.html
git commit -m "Update architecture documentation"
```

---

## See Also

- [Command Reference](commands.md) - Detailed command documentation
- [Setup Guide](setup.md) - Installation and configuration
- [Architecture](architecture.md) - How CodeGraph works
