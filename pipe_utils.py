import os
import re
import tempfile
import numpy as np
from datetime import datetime
from pyannote.audio import Pipeline, Model, Inference
import whisper
import faiss
from pydub import AudioSegment

# Embedding inference client
def get_embedding_inference():
    model = Model.from_pretrained("pyannote/embedding")
    return Inference(model, window="whole")

# Register voice into FAISS index and metadata TSV
def register_voice(fio: str, audio_path: str, index_path="voice_index.faiss", meta_path="voice_meta.tsv"):
    inf = get_embedding_inference()
    embedding = inf(audio_path).reshape(1, -1)
    # load or init index
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        with open(meta_path, 'a') as meta:
            idx = index.ntotal
            index.add(embedding)
            meta.write(f"{idx}\t{fio}\n")
    else:
        dim = embedding.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embedding)
        with open(meta_path, 'w') as meta:
            meta.write("id\tfio\n0\t" + fio + "\n")
    faiss.write_index(index, index_path)

# Diarize and transcribe
def transcribe_and_diarize(audio_path, output_txt_path, hf_token, whisper_model="base"):
    diarizer = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)
    diarization = diarizer(audio_path)
    model = whisper.load_model(whisper_model)
    with open(output_txt_path, 'w') as out:
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start, end = turn.start, turn.end
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            AudioSegment.from_file(audio_path)[start*1000:end*1000].export(tmp.name, format="wav")
            txt = model.transcribe(tmp.name)['text']
            out.write(f"[{speaker} {start:.2f}-{end:.2f}] {txt}\n")
            os.unlink(tmp.name)

# Identify speakers by comparing to registered voices
def identify_speakers(transcript_path, index_path="voice_index.faiss", meta_path="voice_meta.tsv"):
    index = faiss.read_index(index_path)
    id2fio = dict(line.strip().split("\t") for line in open(meta_path) if '\t' in line)
    segments = []
    # parse
    pattern = re.compile(r"\[(SPEAKER_\d+) ([0-9.]+)-([0-9.]+)\]")
    for line in open(transcript_path):
        m = pattern.match(line)
        if m:
            segments.append((m.group(1), float(m.group(2)), float(m.group(3)), line))
    mapping = {}
    inf = get_embedding_inference()
    for spk, segs in itertools.groupby(segments, key=lambda x: x[0]):
        combined = AudioSegment.empty()
        for _, st, en, txt in segs:
            combined += AudioSegment.from_file(transcript_path)[int(st*1000):int(en*1000)]
        emb = inf(combined).reshape(1,-1)
        _, I = index.search(emb, 1)
        mapping[spk] = id2fio[str(I[0][0])]
    return mapping

# Replace SPEAKER_X with real names
def replace_speaker_ids(transcript_path, mapping, output_path):
    text = open(transcript_path).read()
    for k, v in mapping.items():
        text = re.sub(fr"\[{k}", f"[{v}", text)
    with open(output_path, 'w') as f: f.write(text)
