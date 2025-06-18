import streamlit as st
from pipe_utils import register_voice, transcribe_and_diarize, identify_speakers, replace_speaker_ids
from rag_utils import RAGDatabase
import tempfile
import os

st.set_page_config(page_title="Voice+RAG System")
mode = st.sidebar.radio("Экран", ["Материалы", "Аудио анализ", "RAG Чат"])

# INIT RAG DB
rag_db = RAGDatabase()

if mode == "Материалы":
    st.header("Добавление материалов")
    # Voice reg
    fio = st.text_input("ФИО пользователя")
    audio = st.file_uploader("Аудио для регистрации голоса", type=['wav','mp3'])
    if st.button("Зарегистрировать голос") and fio and audio:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        tmp.write(audio.read()); tmp.flush()
        register_voice(fio, tmp.name)
        st.success("Голос зарегистрирован")
    # RAG txt
    txt = st.file_uploader("Загрузить TXT для RAG", type=['txt'])
    if st.button("Добавить в RAG") and txt:
        data = txt.read().decode('utf-8')
        rag_db.add_document(data, txt.name)
        rag_db.build()
        st.success("Документ добавлен и индекс построен")

elif mode == "Аудио анализ":
    st.header("Анализ аудио")
    audio = st.file_uploader("Загрузить аудио", type=['wav','mp3'])
    hf = st.text_input("HF токен для диаризации")
    if st.button("Начать анализ") and audio and hf:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        tmp.write(audio.read()); tmp.flush()
        out_txt = tmp.name + '.txt'
        transcribe_and_diarize(tmp.name, out_txt, hf)
        # identify
        mapping = identify_speakers(out_txt)
        final = tmp.name + '.final.txt'
        replace_speaker_ids(out_txt, mapping, final)
        with open(final) as f: text = f.read()
        st.download_button("Скачать результат", data=text, file_name='transcript.txt')
        st.text_area("Транскрипт", text, height=300)

else:  # RAG Chat
    st.header("RAG Чат")
    query = st.text_input("Введите вопрос")
    if st.button("Спросить") and query:
        answer = rag_db.answer(query)
        st.write(answer)
