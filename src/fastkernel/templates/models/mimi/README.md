# Mimi campaign

```bash
fast-kernel probe        # GPU + backends -> capabilities.json
fast-kernel baseline     # experiment #0 (stock transformers MimiModel, fp32) + noise floor + PLAN.md
fast-kernel dashboard    # live graph at http://127.0.0.1:8765
```

Then, in Claude Code from the repository root: type "Optimize the Mimi codec model." and the agent runs the endless experiment loop. See RECIPES.md and PLAN.md.
