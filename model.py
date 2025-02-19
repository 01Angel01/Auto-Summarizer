from transformers import pipeline


# Fungsi untuk memuat model Hugging Face untuk summarization
def load_model():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    # summarizer = pipeline("summarization", model="facebook/mbart-large-50-many-to-one-mmt")
    return summarizer
