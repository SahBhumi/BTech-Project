import os
from flask import Flask, render_template, request
import speech_recognition as sr
from pydub import AudioSegment
from transformers import pipeline
import spacy

# Initialize the Flask app
app = Flask(__name__)

# Setup Hugging Face transformer model for summarization
summarizer = pipeline("summarization", model="t5-small")

# Load spaCy model for keyword extraction
nlp = spacy.load("en_core_web_sm")

# Ensure 'uploads' folder exists
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# Route to upload and process audio file
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the uploaded audio file
        audio_file = request.files["audio"]
        if audio_file:
            # Save the file locally
            audio_path = os.path.join("uploads", audio_file.filename)
            audio_file.save(audio_path)

            # Convert audio file to .wav format
            converted_audio_path = convert_audio(audio_path)

            # Convert audio to text
            text = audio_to_text(converted_audio_path)

            # Generate summary
            summary = generate_summary(text)

            # Extract keywords
            keywords = extract_keywords(text)

            return render_template("index.html", summary=summary, text=text, keywords=keywords)

    return render_template("index.html", summary=None, keywords=None)

# Function to convert audio to .wav format
def convert_audio(audio_path):
    try:
        audio = AudioSegment.from_file(audio_path)
        new_audio_path = audio_path.split('.')[0] + ".wav"
        audio.export(new_audio_path, format="wav")
        return new_audio_path
    except Exception as e:
        return f"Error converting audio: {str(e)}"

# Function to convert audio to text
def audio_to_text(audio_path):
    recognizer = sr.Recognizer()

    # Load audio file using SpeechRecognition
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)

    # Recognize speech and return text
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Sorry, could not understand the audio."
    except sr.RequestError as e:
        return f"Error with the speech recognition service: {e}"

# Function to generate summary using Hugging Face summarizer
def generate_summary(text):
    try:
        if len(text.split()) > 150:
            max_length = int(len(text.split()) * 0.3)
            min_length = int(len(text.split()) * 0.1)
        else:
            max_length = 50
            min_length = 10
        
        summaries = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        summary = summaries[0]['summary_text']
        
        if len(summary) >= len(text):
            summary = summarizer(text, max_length=int(len(text.split()) * 0.2), min_length=10, do_sample=False)[0]['summary_text']
        
        return summary
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# Function to extract keywords using spaCy
def extract_keywords(text):
    doc = nlp(text)
    keywords = [chunk.text for chunk in doc.noun_chunks if chunk.root.pos_ in ('NOUN', 'PROPN')]
    return ", ".join(keywords)

# Start Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
