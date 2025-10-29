# Data Format Examples

This document describes the expected data formats for DR-QFormer training and evaluation.

## General Structure

All datasets should be in JSONL (JSON Lines) format, where each line is a valid JSON object representing one example.

## Task E: Entailment Tagging

Each example should have:
- `query`: Input query string
- `fragments`: List of retrieved fragments
  - `text`: Fragment text content
  - `score`: Retrieval score (float)
  - `doc_id`: Source document ID
  - `entailment_label`: Binary label (0 or 1) indicating relevance
- `example_id`: Unique identifier

```json
{
  "example_id": "nq_train_0001",
  "query": "What is the capital of France?",
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
  ]
}
```

## Task S: Fragment Sorting

Each example should have:
- `query`: Input query string
- `fragments`: List of retrieved fragments
  - `text`: Fragment text content
  - `score`: Retrieval score (float)
  - `doc_id`: Source document ID
  - `relevance_score`: Ground-truth relevance score (float, higher = more relevant)
- `example_id`: Unique identifier

```json
{
  "example_id": "msmarco_train_0001",
  "query": "how does photosynthesis work",
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
    }
  ]
}
```

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
