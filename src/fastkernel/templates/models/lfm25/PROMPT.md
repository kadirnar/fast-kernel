Optimize the LFM2.5 campaign in this directory with fast-kernel: read AGENTS.md, GOAL.md, PLAN.md and
KNOWLEDGE.md, then run the endless experiment loop (profile -> highest-Amdahl untried idea -> edit
candidate/ only -> `fast-kernel eval` -> learn -> repeat). Decode is launch bound: start with a static
KV cache + CUDA-graph decode step, then fused norms/MLP epilogues, then a fused ShortConv kernel.
