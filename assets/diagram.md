# DR-QFormer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DR-QFormer Pipeline                      │
└─────────────────────────────────────────────────────────────┘

Input Query
    │
    ├──────────────────┐
    │                  │
    ▼                  ▼
┌─────────┐      ┌─────────────┐
│ Frozen  │      │   Frozen    │
│Retriever│      │  LLM (for   │
│         │      │ generation) │
└─────────┘      └─────────────┘
    │                  ▲
    │ P_embeds         │ Z (condensed)
    │ [k, d]           │
    ▼                  │
┌─────────────────────┴───────┐
│      DR-QFormer (TRAINABLE) │
│                              │
│  ┌────────────────────────┐ │
│  │ Learnable Query Tokens│ │
│  │    (LQs) [N, d]       │ │
│  └────────────────────────┘ │
│            │                 │
│            ▼                 │
│  ┌────────────────────────┐ │
│  │ Self-Attention (SA)    │ │
│  │ over [LQs, q/a text]  │ │
│  └────────────────────────┘ │
│            │                 │
│            ▼                 │
│  ┌────────────────────────┐ │
│  │ Cross-Attention (CA)   │ │
│  │ over P_embeds          │ │
│  └────────────────────────┘ │
│            │                 │
│            ▼                 │
│  ┌────────────────────────┐ │
│  │  Output: Z [N, d]      │ │
│  └────────────────────────┘ │
└──────────────┬───────────────┘
               │
               ├────────────┬──────────────┬──────────────┐
               ▼            ▼              ▼              ▼
         ┌──────────┐ ┌──────────┐  ┌──────────┐  ┌──────────┐
         │Entailment│ │ Sorting  │  │Condense  │  │   ...    │
         │   Head   │ │   Head   │  │   Head   │  │  (more)  │
         │(TRAINABLE)│(TRAINABLE)│  │(TRAINABLE)│  │          │
         └──────────┘ └──────────┘  └──────────┘  └──────────┘
               │            │              │
               ▼            ▼              ▼
         [k] labels   [k] scores    LLM prefix
```

## Key Components

### Frozen Components
- **Retriever**: Dense retriever (Contriever, DPR, etc.)
  - Maps query → top-k fragments
  - Returns fragment embeddings P_embeds [k, d]
  - All parameters frozen

- **LLM**: Causal language model (LLaMA, Mistral, Phi, etc.)
  - Generates text conditioned on Z
  - All parameters frozen

### Trainable Components
- **Q-Former**: BLIP-2-style transformer
  - N learnable query tokens (LQs)
  - Self-attention over [LQs, query/answer text]
  - Cross-attention over retriever embeddings
  - Outputs contextualized representations Z [N, d]

- **Task Heads**: Lightweight projection layers
  - EntailmentHead: Binary classification for fragment relevance
  - SortingHead: Ranking scores for fragments
  - CondenseHead: Projects Z to LLM input space

## Attention Policy

**Self-Attention (SA)**:
- LQs attend to: [LQs, query tokens, answer tokens]
- Bidirectional attention
- Learns to aggregate query/answer information

**Cross-Attention (CA)**:
- LQs attend to: P_embeds (retriever fragment embeddings)
- Bidirectional attention
- Learns to extract relevant information from retrieved fragments

## Training Tasks

### Task E: Entailment Tagging
- Input: Query + k retrieved fragments
- Output: Binary labels [k] (relevant/irrelevant)
- Loss: Binary cross-entropy

### Task S: Fragment Sorting
- Input: Query + k retrieved fragments
- Output: Relevance scores [k]
- Loss: Ranking loss (ListMLE, pairwise, etc.)

### Task C: Condensing-Generation
- Input: Query + k retrieved fragments + reference answer
- Output: Condensed representation Z → LLM generation
- Loss: Reward margin (ROUGE/BLEU of generated vs. reference)

## Parameter Efficiency

**Total trainable parameters**: ~50-100M
- Q-Former: ~40-80M (depending on depth/width)
- Task heads: ~1-10M each

**Total frozen parameters**: ~1-10B
- Retriever: ~100-400M
- LLM: ~1-10B

**Efficiency ratio**: ~1-2% trainable parameters
