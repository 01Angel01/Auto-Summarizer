import streamlit as st
import requests
import os

API_URL = "http://127.0.0.1:8000"

st.title("Auto Summarizer")

source = st.radio("Select source for summarization:", ("Text", "PDF", "Audio"))

# Summarizing Text
if source == "Text":
    text = st.text_area("Paste your text here:")
    if st.button("Summarize Text"):
        if text:
            response = requests.post(f"{API_URL}/summarize_text/", json={"text": text})
            data = response.json()
            if "summary" in data:
                st.write("Summary:")
                st.write(data["summary"])
            else:
                st.error(data.get("error", "Unknown error"))

# Summarizing PDF
elif source == "PDF":
    pdf_file = st.file_uploader("Upload a PDF file", type="pdf")
    if pdf_file and st.button("Summarize PDF"):
        files = {"file": pdf_file.getvalue()}
        response = requests.post(f"{API_URL}/summarize_pdf/", files=files)
        data = response.json()
        if "summary" in data:
            st.write("Summary:")
            st.write(data["summary"])
        else:
            st.error(data.get("error", "Unknown error"))

# Summarizing Audio
elif source == "Audio":
    audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])
    if audio_file and st.button("Process Audio"):
        # Pass audio file correctly as a file object
        files = {"file": audio_file}
        response = requests.post(
            f"{API_URL}/process_audio/",
            files=files,
            data={"user_prompt": "Summarize this audio."},
        )
        data = response.json()
        if "processed_text" in data:
            st.text_area("Processed Output", data["processed_text"], height=300)
        else:
            st.error(data.get("error", "Unknown error"))
