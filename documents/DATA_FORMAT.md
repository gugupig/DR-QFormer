# Data Format Examples

This document describes the expected data formats for DR-QFormer training and evaluation.

## General Structure

All datasets should be in JSONL (JSON Lines) format, where each line is a valid JSON object representing one example.

## Task E: Fragment-Level Entailment Tagging (蕴含-标注)

### Purpose
Learn which fragments entail/support the query (Primal) or answer (Dual).

### Required Fields
- `query`: Input query string (for Primal mode)
- `answer`: Ground-truth answer string (for Dual mode, optional for Primal)
- `fragments`: List of k retrieved text fragments
  - `text`: Fragment text content
  - `score`: Retrieval score (float, from retriever)
  - `doc_id`: Source document ID
  - `entailment_label`: Binary label (0 or 1) indicating if fragment is golden evidence
- `gt_k`: **[NEW]** Pre-computed binary vector [k] for efficient training
- `example_id`: Unique identifier

### Supervision: gt_k Vector
**gt_k** is a k-dimensional binary vector marking which fragments are golden evidence:
- `1`: Fragment entails/supports the query (Primal) or answer (Dual)
- `0`: Fragment does not entail/is irrelevant

**Shape**: `[k]` where k = number of fragments (e.g., k=10)
**Example**: `[0, 1, 0, 1, 0, 0, 1, 0, 0, 0]` means fragments #1, #3, #6 are golden

### Generation Methods for gt_k

#### Method 1: Answer Span Matching (for Primal/QA)
```python
def generate_gt_k_primal(query, answer, fragments):
    """Check if fragment contains answer span."""
    gt_k = []
    for frag in fragments:
        # Exact match or fuzzy match
        has_answer = answer.lower() in frag.text.lower()
        gt_k.append(1 if has_answer else 0)
    return gt_k
```

#### Method 2: NLI Model (Natural Language Inference)
```python
def generate_gt_k_nli(query, fragments):
    """Use pre-trained NLI model."""
    nli_model = load_nli_model("roberta-large-mnli")
    gt_k = []
    for frag in fragments:
        # Query as premise, fragment as hypothesis (or vice versa)
        score = nli_model.predict_entailment(query, frag.text)
        gt_k.append(1 if score > threshold else 0)
    return gt_k
```

#### Method 3: Human Annotation
Label fragments manually as relevant (1) or irrelevant (0) to the query/answer.

### Example (Primal Mode - QA)
```json
{
  "example_id": "nq_train_0001",
  "query": "What is the capital of France?",
  "answer": "Paris",
  "fragments": [
    {
      "text": "Paris is the capital and most populous city of France.",
      "score": 0.95,
      "doc_id": "wiki_france_001",
      "entailment_label": 1
    },
    {
      "text": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.",
      "score": 0.82,
      "doc_id": "wiki_eiffel_001",
      "entailment_label": 1
    },
    {
      "text": "Lyon is the third-largest city in France.",
      "score": 0.65,
      "doc_id": "wiki_france_005",
      "entailment_label": 0
    }
  ],
  "gt_k": [1, 1, 0]
}
```

### Example (Dual Mode - QG)
```json
{
  "example_id": "nq_train_0001_dual",
  "answer": "Paris",
  "query": "What is the capital of France?",
  "fragments": [
    {
      "text": "Paris is the capital and most populous city of France.",
      "score": 0.95,
      "doc_id": "wiki_france_001",
      "entailment_label": 1
    },
    {
      "text": "Many tourists visit the Louvre Museum in Paris every year.",
      "score": 0.75,
      "doc_id": "wiki_paris_tourism",
      "entailment_label": 0
    }
  ],
  "gt_k": [1, 0]
}
```

### Loss Computation
```python
# EntailmentHead output: logits [batch, k]
# Ground truth: gt_k [batch, k]
loss = F.binary_cross_entropy_with_logits(logits, gt_k.float())
```

## Task S: Fragment-Level Sorting Supervision (排序-监督)

### Purpose
Train CA layer attention weights to match target ranking distribution over fragments.

### Required Fields
- `query`: Input query string (for Primal mode)
- `answer`: Ground-truth answer string (for Dual mode, optional for Primal)
- `fragments`: List of k retrieved text fragments
  - `text`: Fragment text content
  - `score`: Retrieval score (float, from retriever)
  - `doc_id`: Source document ID
  - `relevance_score`: Ground-truth relevance score (float, higher = more relevant)
- `gt_soft_weights`: **[NEW]** Pre-computed probability distribution [k] for efficient training
- `example_id`: Unique identifier

### Supervision: gt_soft_weights Distribution
**gt_soft_weights** is a k-dimensional probability distribution reflecting fragment relative importance:
- **Shape**: `[k]` where k = number of fragments (e.g., k=10)
- **Constraint**: Elements sum to 1.0 (valid probability distribution)
- **Semantics**: Higher value = more important/relevant fragment

**Example**: `[0.4, 0.3, 0.2, 0.1, 0.0, ...]` means:
- Fragment #0 is most important (40%)
- Fragment #1 is second (30%)
- Fragment #2 is third (20%)
- etc.

### Generation Methods for gt_soft_weights

#### Method 1: Softmax of Relevance Scores (from offline retriever/reranker)
```python
def generate_gt_soft_weights_from_scores(fragments, temperature=1.0):
    """Convert relevance scores to probability distribution."""
    import numpy as np
    
    scores = np.array([f.relevance_score for f in fragments])
    
    # Apply softmax
    exp_scores = np.exp(scores / temperature)
    gt_soft_weights = exp_scores / exp_scores.sum()
    
    return gt_soft_weights.tolist()

# Example:
# relevance_scores = [3.0, 2.5, 1.0, 0.5]
# gt_soft_weights = [0.52, 0.32, 0.12, 0.04]  (after softmax)
```

#### Method 2: Normalized Rankings
```python
def generate_gt_soft_weights_from_ranks(fragments):
    """Use position-based weights (e.g., 1/rank)."""
    k = len(fragments)
    weights = np.array([1.0 / (i + 1) for i in range(k)])  # 1, 1/2, 1/3, ...
    gt_soft_weights = weights / weights.sum()  # Normalize
    return gt_soft_weights.tolist()

# Example for k=4:
# weights = [1.0, 0.5, 0.33, 0.25]
# gt_soft_weights = [0.48, 0.24, 0.16, 0.12]  (normalized)
```

#### Method 3: Human Annotations → Softmax
```python
def generate_gt_soft_weights_from_human(annotations):
    """Convert human relevance judgments (0-3 scale) to distribution."""
    # annotations = [3, 2, 1, 0, 1, ...]  (per fragment)
    scores = np.array(annotations).astype(float)
    exp_scores = np.exp(scores)
    gt_soft_weights = exp_scores / exp_scores.sum()
    return gt_soft_weights.tolist()
```

#### Method 4: BM25/Dense Retriever Scores
```python
def generate_gt_soft_weights_from_retriever(retriever, query, fragments):
    """Use offline retriever/reranker scores."""
    scores = []
    for frag in fragments:
        score = retriever.score(query, frag.text)
        scores.append(score)
    
    # Normalize to distribution
    scores = np.array(scores)
    exp_scores = np.exp(scores)
    gt_soft_weights = exp_scores / exp_scores.sum()
    return gt_soft_weights.tolist()
```

### Example (Primal Mode - QA)
```json
{
  "example_id": "msmarco_train_0001",
  "query": "how does photosynthesis work",
  "answer": "Plants use sunlight to convert CO2 and water into glucose and oxygen",
  "fragments": [
    {
      "text": "Photosynthesis is the process by which plants use sunlight, water and carbon dioxide to create oxygen and energy in the form of sugar.",
      "score": 0.88,
      "doc_id": "bio_photo_001",
      "relevance_score": 3.0
    },
    {
      "text": "Chloroplasts are organelles found in plant cells that conduct photosynthesis.",
      "score": 0.75,
      "doc_id": "bio_photo_002",
      "relevance_score": 2.5
    },
    {
      "text": "Plants need sunlight to grow and produce food.",
      "score": 0.60,
      "doc_id": "bio_plant_001",
      "relevance_score": 1.0
    },
    {
      "text": "The sun is a star at the center of our solar system.",
      "score": 0.45,
      "doc_id": "astro_sun_001",
      "relevance_score": 0.5
    }
  ],
  "gt_soft_weights": [0.52, 0.32, 0.12, 0.04]
}
```

### Example (Dual Mode - QG)
```json
{
  "example_id": "msmarco_train_0001_dual",
  "answer": "Plants use sunlight to convert CO2 and water into glucose and oxygen",
  "query": "how does photosynthesis work",
  "fragments": [
    {
      "text": "Photosynthesis is the process by which plants use sunlight, water and carbon dioxide to create oxygen and energy in the form of sugar.",
      "score": 0.88,
      "doc_id": "bio_photo_001",
      "relevance_score": 3.5
    },
    {
      "text": "Chlorophyll is the green pigment in plants that absorbs light energy.",
      "score": 0.70,
      "doc_id": "bio_chloro_001",
      "relevance_score": 2.0
    }
  ],
  "gt_soft_weights": [0.67, 0.33]
}
```

### Loss Computation
```python
# SortingHead output: predicted_weights [batch, k] (softmax probabilities)
# Ground truth: gt_soft_weights [batch, k] (target distribution)
loss = F.kl_div(
    F.log_softmax(predicted_logits, dim=-1),  # Log probabilities
    gt_soft_weights,                           # Target distribution
    reduction='batchmean'
)
```

### Notes
- **Temperature**: Adjust softmax temperature to control distribution sharpness
  - High temp (>1): Flatter distribution (more uncertain)
  - Low temp (<1): Sharper distribution (more confident)
- **Cold Start**: If no offline scores available, use uniform distribution initially
- **Quality**: Better offline ranker → better gt_soft_weights → better Task S training

## Task C: Condensing-Generation

Each example should have:
- `query`: Input query string
- `answer`: Ground-truth answer for reward computation
- `fragments`: List of retrieved fragments
  - `text`: Fragment text content
  - `score`: Retrieval score (float)
  - `doc_id`: Source document ID
- `example_id`: Unique identifier

```json
{
  "example_id": "eli5_train_0001",
  "query": "Why is the sky blue?",
  "answer": "The sky appears blue because molecules in the air scatter blue light from the sun more than they scatter red light. This is called Rayleigh scattering.",
  "fragments": [
    {
      "text": "Rayleigh scattering is the scattering of electromagnetic radiation by particles much smaller than the wavelength of the radiation.",
      "score": 0.85,
      "doc_id": "physics_scatter_001"
    },
    {
      "text": "Blue light is scattered more than other colors because it travels as shorter, smaller waves.",
      "score": 0.80,
      "doc_id": "physics_light_001"
    },
    {
      "text": "The atmosphere is made up of nitrogen, oxygen, and other gases.",
      "score": 0.65,
      "doc_id": "earth_atmos_001"
    }
  ]
}
```

## Question Answering (General)

For general QA tasks, include both query and answer:

```json
{
  "example_id": "qa_0001",
  "query": "Who wrote Romeo and Juliet?",
  "answer": "William Shakespeare",
  "fragments": [
    {
      "text": "Romeo and Juliet is a tragedy written by William Shakespeare early in his career about two young Italian star-crossed lovers.",
      "score": 0.95,
      "doc_id": "lit_shakespeare_001"
    }
  ],
  "metadata": {
    "dataset": "natural_questions",
    "split": "train",
    "difficulty": "easy"
  }
}
```

## Question Generation

For QG tasks:

```json
{
  "example_id": "qg_0001",
  "context": "Paris is the capital and most populous city of France. It has an area of 105 square kilometres and a population of 2.1 million.",
  "query": "What is the population of Paris?",
  "answer": "2.1 million",
  "fragments": [
    {
      "text": "The population of Paris is approximately 2.1 million residents within the city limits.",
      "score": 0.90,
      "doc_id": "wiki_paris_demographics"
    }
  ]
}
```

## Minimal Example

Absolute minimum required fields:

```json
{
  "query": "example query",
  "fragments": [
    {"text": "fragment 1"},
    {"text": "fragment 2"}
  ]
}
```

## Preprocessing Notes

1. **Tokenization**: Not included in data files; handled by model tokenizers
2. **Embeddings**: P_embeds computed by retriever at runtime
3. **Fragment Limits**: Truncate or pad to `k_fragments` during loading
4. **Text Length**: Long texts should be truncated by tokenizer max_length

## Creating Your Own Dataset

To create a dataset:

1. **Retrieve fragments** for each query using your retriever
2. **Add labels/scores** depending on task:
   - Entailment: Binary relevance labels
   - Sorting: Relevance scores (can be from human annotations or heuristics)
   - Generation: Reference answers
3. **Format as JSONL** with one example per line
4. **Split into train/dev/test** sets

Example script structure:

```python
import json

examples = []
for query, answer in your_data:
    fragments = retriever.retrieve(query, k=10)
    
    example = {
        "example_id": generate_id(),
        "query": query,
        "answer": answer,
        "fragments": [
            {
                "text": frag.text,
                "score": frag.score,
                "doc_id": frag.doc_id,
                # Add task-specific fields
            }
            for frag in fragments
        ]
    }
    examples.append(example)

# Save as JSONL
with open("train.jsonl", "w") as f:
    for example in examples:
        f.write(json.dumps(example) + "\n")
```

## Data Loading

The `dr_qformer.data.interfaces.load_dataset()` function will:
1. Parse JSONL file
2. Create `Example` objects with `Fragment` instances
3. Return a `DRQFormerDataset` instance

See `dr_qformer/data/interfaces.py` for implementation details (TODO).
