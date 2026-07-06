"""
gradio_app.py

UI for version 2 of the RAG project. Using Gradio instead of Streamlit this
time mainly to actually compare the two - Gradio works pretty differently
under the hood, you wire up specific functions to specific button clicks
instead of the whole script re-running top to bottom on every interaction
like Streamlit does. Took a bit to wrap my head around that difference.

Run this with:
    python gradio_app.py
"""

import random

import gradio as gr
import pandas as pd

from rag_system import RAGSystem

DEFAULT_DOC = "network_notes.txt"

print("Setting up RAG system, this loads the models once at startup...")
rag = RAGSystem()

# gradio doesn't give each session its own python object automatically the
# way streamlit's script-rerun model kind of does, so keeping track of the
# currently loaded chunks in a plain dict here instead. fine for a single
# person using this locally, would need a different approach (like actual
# gr.State) if this needed to support multiple users at once
app_state = {"chunks": [], "doc_name": None}


def get_filepath(file):
    """
    gradio's file upload gives back a plain string path in newer versions,
    but handling the older .name attribute style too just in case someone
    runs this with an older gradio install
    """
    if file is None:
        return None
    if isinstance(file, str):
        return file
    return file.name


def load_and_index(file, chunk_size):
    filepath = get_filepath(file)

    if filepath is not None:
        raw_text = rag.load_document(file_path=filepath)
        doc_name = filepath.replace("\\", "/").split("/")[-1]
    else:
        raw_text = rag.load_document(file_path=DEFAULT_DOC)
        doc_name = f"{DEFAULT_DOC} (bundled sample)"

    chunks = rag.chunk_by_paragraph(raw_text, target_size=int(chunk_size))
    rag.build_index(chunks)

    app_state["chunks"] = chunks
    app_state["doc_name"] = doc_name

    return f"Loaded **{doc_name}** — {len(raw_text)} characters, split into **{len(chunks)}** chunks."


def ask_question(query, top_k):
    if not app_state["chunks"]:
        return "Load a document first using the button above.", ""

    if not query or not query.strip():
        return "Type a question first.", ""

    answer, retrieved = rag.answer(query, top_k=int(top_k))

    chunks_display = ""
    for r in retrieved:
        chunks_display += f"**similarity score: {r['score']}**\n\n{r['chunk']}\n\n---\n\n"

    return answer, chunks_display


# question bank for the validation check - written specifically against the
# bundled network_notes.txt, same idea as version 1's validation section
question_bank = [
    ("What is the OSI model?", ["osi", "layers"]),
    ("What does TCP stand for?", ["transmission control protocol", "tcp"]),
    ("What is subnetting?", ["subnet"]),
    ("What is a firewall?", ["firewall"]),
    ("What is packet switching?", ["packet switching"]),
    ("What is DNS used for?", ["domain name system", "dns"]),
    ("What is congestion control?", ["congestion"]),
    ("What is a VPN?", ["vpn"]),
    ("What does BGP do?", ["bgp", "autonomous system"]),
    ("What is the difference between a LAN and a WAN?", ["lan", "wan"]),
]


def run_validation():
    if not app_state["chunks"]:
        return "Load a document first using the button above.", None

    sample = random.sample(question_bank, 5)
    rows = []
    for q, keywords in sample:
        retrieved = rag.retrieve(q, top_k=1)
        top_chunk_lower = retrieved[0]["chunk"].lower()
        hit = any(kw in top_chunk_lower for kw in keywords)
        rows.append({
            "query": q,
            "retrieval_score": retrieved[0]["score"],
            "keyword_found": hit,
        })

    df = pd.DataFrame(rows)
    accuracy = df["keyword_found"].mean() * 100
    return f"**Retrieval accuracy on this sample: {accuracy:.1f}%**", df


def get_metrics_report():
    if not app_state["chunks"]:
        return None

    report = {
        "document": app_state["doc_name"],
        "chunking_strategy": "paragraph-based with merging (not fixed-length)",
        "num_chunks": len(app_state["chunks"]),
        "embedding_model": "all-MiniLM-L6-v2" if rag.embed_model is not None else "TF-IDF (fallback, no internet)",
        "vector_store": "ChromaDB (cosine distance)",
        "generation_model": "google/flan-t5-small" if rag.generator is not None else "extractive fallback (no model, no internet)",
    }
    return pd.DataFrame(list(report.items()), columns=["metric", "value"])


with gr.Blocks(title="Document QA System (RAG) - v2") as demo:
    gr.Markdown("# Document Question Answering System — v2")
    gr.Markdown(
        "Second version of the RAG mini project. Different dataset (networking notes instead of "
        "DBMS notes), class-based pipeline instead of standalone functions, paragraph-based "
        "chunking instead of fixed-length, and ChromaDB instead of FAISS."
    )

    gr.Markdown("## 1. Document")
    with gr.Row():
        file_input = gr.File(label="Upload a .txt or .pdf (leave empty to use the bundled networking notes)")
        chunk_size_slider = gr.Slider(200, 1000, value=500, step=50, label="Target chunk size (characters)")

    load_button = gr.Button("Load document", variant="secondary")
    load_status = gr.Markdown()

    load_button.click(load_and_index, inputs=[file_input, chunk_size_slider], outputs=load_status)

    gr.Markdown("## 2. Ask a question")
    with gr.Row():
        question_input = gr.Textbox(label="Your question", scale=3)
        top_k_slider = gr.Slider(1, 5, value=3, step=1, label="Chunks to retrieve", scale=1)

    ask_button = gr.Button("Get answer", variant="primary")
    answer_output = gr.Textbox(label="Answer", lines=3)
    chunks_output = gr.Markdown(label="Retrieved chunks (what the answer is based on)")

    ask_button.click(ask_question, inputs=[question_input, top_k_slider], outputs=[answer_output, chunks_output])

    gr.Markdown("## 3. Validation check")
    gr.Markdown("Samples 5 random questions from a small test bank and checks whether the right keywords show up in the retrieved chunk.")
    validate_button = gr.Button("Run validation check")
    validation_status = gr.Markdown()
    validation_table = gr.Dataframe()

    validate_button.click(run_validation, outputs=[validation_status, validation_table])

    gr.Markdown("## 4. System metrics report")
    metrics_button = gr.Button("Show metrics report")
    metrics_table = gr.Dataframe()

    metrics_button.click(get_metrics_report, outputs=metrics_table)

    # auto-load the bundled document as soon as the app starts, so there's
    # something to query immediately without clicking "Load document" first
    demo.load(lambda: load_and_index(None, 500), outputs=load_status)


if __name__ == "__main__":
    demo.launch()
