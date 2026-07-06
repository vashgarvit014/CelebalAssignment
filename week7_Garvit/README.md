# Document Question Answering System (RAG) — Version 2

Second version of the RAG mini project, built to actually compare two
different approaches to the same problem rather than just reskinning
version 1. Same underlying task, several deliberately different choices.

## What's different from version 1

| | Version 1 | Version 2 |
|---|---|---|
| Dataset | DBMS notes | Computer networking notes |
| Chunking | Fixed-length (400 chars + overlap) | Paragraph-based, merged up to a target size |
| Vector store | FAISS (`IndexFlatIP`) | ChromaDB |
| Code structure | Standalone functions | `RAGSystem` class |
| UI | Streamlit | Gradio |
| Generation model | flan-t5-small | flan-t5-small (kept the same on purpose) |

Kept the generation model identical in both so that if the answers come out
different, it's actually because of the retrieval/chunking/structure
changes and not just random model variance.

## What's in here

- `rag_system.py` - the `RAGSystem` class: load, chunk, embed, index, retrieve, generate
- `gradio_app.py` - Gradio UI on top of `RAGSystem`
- `network_notes.txt` - sample dataset (computer networking basics)
- `requirements.txt`

## How to run this

Same deal as version 1 - runs locally, not in Colab, since it's a
persistent local web server.

```
pip install -r requirements.txt
python gradio_app.py
```

Gradio will print a local URL (usually `http://127.0.0.1:7860`) - open that
in a browser. First run downloads the embedding + generation models, so
give it a minute.

## Things I noticed comparing the two versions

- **Paragraph chunking is genuinely better** than fixed-length for this kind
  of document - stopped the "cuts a word in half at the boundary" problem
  from version 1 completely. Downside is chunk sizes are less predictable
  since paragraphs vary in length.
- **ChromaDB vs FAISS** - Chroma's API is friendlier (`add()` + `query()`,
  no manual index math), and it can persist to disk by itself. FAISS felt
  more "raw" but also more transparent about what's actually happening.
  Chroma's default distance metric is Euclidean, not cosine - had to set
  `metadata={"hnsw:space": "cosine"}` explicitly when creating the
  collection, otherwise the similarity scores come out wrong (found this by
  actually testing it - got a negative similarity score before adding that
  line).
- **Class vs functions** - the class version keeps related state (chunks,
  models, the collection) together on `self` instead of passing five
  arguments into every function. Feels cleaner once the pipeline has this
  many moving parts, though it's a bit more to set up initially (`__init__`,
  private helper methods) compared to just writing plain functions.
- **Same fallback edge case as version 1** - here it's the BGP question that
  the TF-IDF fallback struggles with (low retrieval score, since "BGP" and
  "autonomous system" barely overlap with how the notes phrase things),
  mirroring the BCNF acronym issue from version 1's DBMS notes. Same root
  cause both times - short acronym-style queries don't share enough words
  with the source text for a word-overlap-based fallback to catch them.

## If a model fails to load (no internet)

Same fallback design as version 1 - embeddings fall back to TF-IDF vectors
(explicitly passed into Chroma rather than relying on Chroma's own default
embedder, which also needs an internet download the first time it's used).
Generation falls back to picking the most relevant sentences straight out
of the retrieved chunks. Both print a warning to the terminal when this
happens.
