Optimize the YOLO campaign in this directory with fast-kernel: read AGENTS.md, GOAL.md, PLAN.md and
KNOWLEDGE.md, then run the endless experiment loop. Batch 1 is launch bound: CUDA-graph the forward
first, then channels-last/cuDNN tuning, then fused Conv+SiLU epilogues and copy elimination.
