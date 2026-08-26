Optimize the Mimi codec campaign in this directory with fast-kernel. Read AGENTS.md, GOAL.md, PLAN.md,
RECIPES.md and KNOWLEDGE.md, then run the experiment loop indefinitely: profile, pick the highest-Amdahl
untried idea, edit only candidate/, `fast-kernel eval`, learn, repeat. Never stop to ask; the harness
decides keep/revert. Start with CUDA graphs on the stock path, then the fused RVQ search, then the
transformer blocks, then the SEANet convolutions.
