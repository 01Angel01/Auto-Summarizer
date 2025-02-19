from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import whisper
import fitz  # PyMuPDF untuk ekstrak teks dari PDF
from model import load_model
import google.generativeai as genai
import os
from dotenv import load_dotenv
from pydub import AudioSegment
import traceback
from pydub.utils import which

# Ensure ffmpeg is correctly detected
AudioSegment.ffmpeg = which("ffmpeg")

load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Load Summarization model
summarizer = load_model()

# Load Whisper model
whisper_model = whisper.load_model("base")

# Configure Google API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Ensure the temp directory exists
os.makedirs("temp", exist_ok=True)


class SummarizationRequest(BaseModel):
    text: str = None
    user_prompt: str = None


@app.post("/summarize_text/")
async def summarize_text(request: SummarizationRequest):
    """Summarize input text."""
    if request.text:
        summary = summarizer(
            request.text, max_length=150, min_length=30, do_sample=False
        )
        return {"summary": summary[0]["summary_text"]}
    return {"error": "No text provided for summarization"}


@app.post("/summarize_pdf/")
async def summarize_pdf(file: UploadFile = File(...)):
    """Extract text from PDF and summarize it."""
    try:
        pdf_document = fitz.open(stream=file.file.read(), filetype="pdf")
        text = "\n".join([page.get_text() for page in pdf_document])

        if not text.strip():
            return {"error": "Failed to extract text from PDF"}

        summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
        return {"summary": summary[0]["summary_text"]}
    except Exception as e:
        return {"error": f"Error processing PDF: {str(e)}"}


@app.post("/process_audio/")
async def process_audio(
    file: UploadFile = File(...), user_prompt: str = "Summarize this audio."
):
    """Process uploaded audio and generate a summary."""
    print("Processing audio...")  # This confirms that the route is hit
    try:
        # Ensure temp directory exists
        os.makedirs("temp", exist_ok=True)

        # Save the uploaded file
        file_location = os.path.abspath(os.path.join("temp", file.filename))
        with open(file_location, "wb") as f:
            f.write(file.file.read())

        print(f"File saved at: {file_location}")
        print(f"Current working directory: {os.getcwd()}")  # Debugging line

        # Check if the file exists
        if not os.path.isfile(file_location):
            print(f"File not found at: {file_location}")
            return {"error": "File not found."}

        # Convert MP3 to WAV if necessary
        if file_location.endswith(".mp3"):
            wav_file_location = file_location.replace(".mp3", ".wav")
            print(f"Converting MP3 to WAV: {wav_file_location}")
            audio = AudioSegment.from_mp3(file_location)
            audio.export(wav_file_location, format="wav")
            print(f"MP3 converted to WAV at: {wav_file_location}")
            file_location = wav_file_location  # Use the WAV file for transcription

        # Check if the file exists after conversion
        if not os.path.isfile(file_location):
            print(f"Converted file not found at: {file_location}")
            return {"error": "Converted file not found."}

        print(f"Transcribing file: {file_location}")

        # Transcribe audio using Whisper
        result = whisper_model.transcribe(file_location)
        transcription = result.get("text", "")
        print(f"Transcription result: {transcription}")

        # Generate summary using Google's Generative AI
        model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
        response = model.generate_content([user_prompt, transcription])

        # Clean up the temporary file
        os.remove(file_location)

        return {"processed_text": response.text}

    except Exception as e:
        print(f"Error: {str(e)}")
        print("Stack trace:", traceback.format_exc())  # Print the full stack trace
        return {"error": f"Error processing audio: {str(e)}"}
