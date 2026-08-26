Optimize the LFM2-Audio campaign in this directory with fast-kernel: read AGENTS.md, GOAL.md, PLAN.md and
KNOWLEDGE.md, then run the endless experiment loop. Generation is launch bound (backbone step + 8
depthformer steps per frame): start with CUDA graphs on the per-step forwards, then fused norms/MLP/
ShortConv kernels, then the Mimi decoder and FastConformer encoder.
