"""
rag_system.py

Version 2 of the RAG pipeline - built as a class instead of standalone
functions like the first version, mainly to compare the two styles and see
which one actually feels cleaner to work with once the project grows.

Other things changed on purpose vs version 1:
- chunking splits on paragraphs first instead of just cutting every N
  characters, so it stops slicing words in half at chunk boundaries
- uses ChromaDB as the vector store instead of FAISS, since chroma persists
  to disk automatically and has a slightly friendlier api for a beginner
  (add documents + query, no manual index building)

Generation model (flan-t5-small) kept the same as version 1 on purpose, so
if answers come out different between the two versions, it's actually
because of the retrieval/structure changes and not just model randomness.
"""

import re


class RAGSystem:
    """
    wraps the whole pipeline - load, chunk, embed, store, retrieve, generate -
    as one object instead of a bunch of separate functions. state (chunks,
    the chroma collection, the loaded models) lives on self instead of
    getting passed around as arguments everywhere.
    """

    def __init__(self, collection_name="notes"):
        self.chunks = []
        self.embed_model = None
        self.generator = None
        self.collection = None
        self.collection_name = collection_name

        self._load_models()
        self._setup_vector_store()

    # -----------------------------------------------------------------
    # setup
    # -----------------------------------------------------------------

    def _load_models(self):
        """
        tries to load the real embedding + generation models. if either
        fails (no internet), falls back to None and the rest of the class
        checks for that and adjusts - same fallback idea as version 1, just
        living inside the class now instead of floating as loose variables

        note - originally let chroma fall back to its own default embedding
        function when sentence-transformers failed to load, but that default
        also has to download its own small model the first time it's used,
        so on a genuinely offline machine that fails too. switched the
        fallback to tf-idf instead (via self.tfidf_vectorizer below), since
        that never needs to touch the network at all.
        """
        self.tfidf_vectorizer = None

        try:
            from sentence_transformers import SentenceTransformer
            self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print("embedding model failed to load:", e)
            self.embed_model = None

        try:
            from transformers import pipeline
            self.generator = pipeline("text2text-generation", model="google/flan-t5-small")
        except Exception as e:
            print("generation model failed to load:", e)
            self.generator = None

    def _setup_vector_store(self):
        import chromadb

        # in-memory client is enough for this project, chromadb.PersistentClient(path=...)
        # would be the swap if we wanted this to survive between runs
        client = chromadb.Client()

        # fresh collection each time so re-running doesn't duplicate old chunks
        try:
            client.delete_collection(self.collection_name)
        except Exception:
            pass  # collection didn't exist yet, nothing to delete

        # chroma defaults to euclidean distance, but the retrieve() method
        # below converts distance into a similarity score assuming cosine
        # distance (similarity = 1 - distance only makes sense for cosine),
        # so need to explicitly ask for cosine space here or the scores
        # come out wrong (found this the hard way - got a negative score
        # when testing before adding this line)
        self.collection = client.create_collection(
            self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    # -----------------------------------------------------------------
    # document loading + chunking
    # -----------------------------------------------------------------

    def load_document(self, file_path=None, file_obj=None):
        if file_obj is not None:
            name = file_obj.name.lower()
            if name.endswith(".pdf"):
                from pypdf import PdfReader
                reader = PdfReader(file_obj)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
            return file_obj.read().decode("utf-8")

        if file_path.lower().endswith(".pdf"):
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text

        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def chunk_by_paragraph(self, text, target_size=500):
        """
        splits on blank lines (paragraph breaks) first, so we never cut a
        sentence or word in half like the fixed-length approach in version 1
        did. if a single paragraph is longer than target_size, it gets kept
        as its own chunk anyway rather than force-splitting it - better to
        have one slightly oversized chunk than to chop a paragraph
        mid-sentence.

        smaller paragraphs get merged together up to roughly target_size so
        we don't end up with a ton of tiny, low-context chunks either.
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) <= target_size:
                current = (current + "\n\n" + para).strip()
            else:
                if current:
                    chunks.append(current)
                current = para

        if current:
            chunks.append(current)

        return chunks

    # -----------------------------------------------------------------
    # indexing
    # -----------------------------------------------------------------

    def build_index(self, chunks):
        self.chunks = chunks

        if self.embed_model is not None:
            embeddings = self.embed_model.encode(chunks).tolist()
        else:
            # tf-idf fallback - fits a vectorizer on the chunks and uses
            # those as the vectors instead. keeping the fitted vectorizer
            # around on self since the query later needs to be transformed
            # with the exact same vocabulary, not a freshly fitted one
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tfidf_vectorizer = TfidfVectorizer(stop_words="english")
            embeddings = self.tfidf_vectorizer.fit_transform(chunks).toarray().tolist()

        self.collection.add(
            ids=[str(i) for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings,
        )

    # -----------------------------------------------------------------
    # retrieval
    # -----------------------------------------------------------------

    def retrieve(self, query, top_k=3):
        if self.embed_model is not None:
            query_embedding = self.embed_model.encode([query]).tolist()
        else:
            # same vectorizer that was fit on the chunks in build_index,
            # NOT a new one - has to share the same vocabulary or the
            # vectors would be meaningless to compare
            query_embedding = self.tfidf_vectorizer.transform([query]).toarray().tolist()

        results = self.collection.query(query_embeddings=query_embedding, n_results=top_k)

        retrieved = []
        docs = results["documents"][0]
        distances = results["distances"][0]

        for doc, dist in zip(docs, distances):
            # chroma returns distance (lower = more similar), flipping it to
            # a similarity-style score just so it reads the same direction
            # as version 1's cosine similarity scores (higher = better match)
            similarity = 1 - dist
            retrieved.append({"chunk": doc, "score": round(float(similarity), 4)})

        return retrieved

    # -----------------------------------------------------------------
    # generation
    # -----------------------------------------------------------------

    def _generate_with_llm(self, query, retrieved):
        context = " ".join([r["chunk"] for r in retrieved])[:800]
        prompt = f"Answer the question using only the given context.\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
        output = self.generator(prompt, max_new_tokens=80)
        return output[0]["generated_text"].strip()

    def _generate_extractive(self, query, retrieved):
        full_context = " ".join([r["chunk"] for r in retrieved])
        sentences = re.split(r'(?<=[.!?])\s+', full_context)

        q_words = set(re.findall(r"\w+", query.lower()))
        stopwords = {"what", "is", "are", "the", "a", "an", "of", "in", "to",
                     "how", "does", "do", "explain", "define"}
        q_words = q_words - stopwords

        scored = []
        for sent in sentences:
            sent_words = set(re.findall(r"\w+", sent.lower()))
            overlap = len(q_words & sent_words)
            if overlap > 0:
                scored.append((overlap, sent.strip()))

        scored.sort(key=lambda x: x[0], reverse=True)

        if not scored:
            return "Sorry, I couldn't find a relevant answer for this in the document."

        return " ".join([s for _, s in scored[:3]])

    def answer(self, query, top_k=3):
        retrieved = self.retrieve(query, top_k=top_k)

        if self.generator is not None:
            try:
                response = self._generate_with_llm(query, retrieved)
            except Exception as e:
                print("generation failed, falling back to extractive method:", e)
                response = self._generate_extractive(query, retrieved)
        else:
            response = self._generate_extractive(query, retrieved)

        return response, retrieved
