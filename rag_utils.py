import os
import pickle
import faiss
import numpy as np
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from cross_encoder import CrossEncoder

class RAGDatabase:
    def __init__(self, faiss_path='rag_index.faiss', meta_path='rag_meta.pkl'):
        self.faiss_path = faiss_path
        self.meta_path = meta_path
        self.bi = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        self.cross = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
        self.tokenizer = AutoTokenizer.from_pretrained('sberbank-ai/rugpt3large_based_on_gpt2')
        self.gen = AutoModelForCausalLM.from_pretrained('sberbank-ai/rugpt3large_based_on_gpt2')
        self.reset()

    def reset(self):
        self.chunks = []
        self.meta = []
        self.index = None

    def add_document(self, text, filename):
        entries = []
        for match in re.finditer(r"\[(SPEAKER_\d+) ([0-9.]+)-([0-9.]+)\] (.+)", text):
            spk, st, en, txt = match.groups()
            entries.append(txt)
        self.chunks.extend(entries)
        self.meta.extend([(filename, i) for i in range(len(entries))])

    def build(self):
        embs = normalize(self.bi.encode(self.chunks))
        dim = embs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embs)
        faiss.write_index(self.index, self.faiss_path)
        with open(self.meta_path, 'wb') as f:
            pickle.dump(self.meta, f)

    def query(self, q, topk=3):
        emb = normalize(self.bi.encode([q]))
        D, I = self.index.search(emb, k=20)
        scores = self.cross.predict([(q, self.chunks[i]) for i in I[0]])
        best = np.argsort(scores)[-topk:]
        return [self.chunks[int(I[0][i])] for i in best]

    def answer(self, q):
        ctx = self.query(q)
        prompt = 'Контекст: ' + '\n'.join(ctx) + '\nВопрос: ' + q + '\nОтвет:'
        inputs = self.tokenizer(prompt, return_tensors='pt')
        out = self.gen.generate(**inputs, max_new_tokens=150)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
