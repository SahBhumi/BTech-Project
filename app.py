import os
from flask import Flask, render_template, request
import speech_recognition as sr
from pydub import AudioSegment
from transformers import pipeline

# Initialize the Flask app
app = Flask(__name__)

# Setup Hugging Face transformer model for summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


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

            return render_template("index.html", summary=summary, text=text)

    return render_template("index.html", summary=None)

# Function to convert audio to .wav format
def convert_audio(audio_path):
    audio = AudioSegment.from_file(audio_path)
    new_audio_path = audio_path.split('.')[0] + ".wav"
    audio.export(new_audio_path, format="wav")
    return new_audio_path

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
        summary = summarizer(text, max_length=1000, min_length=100, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error generating summary: {str(e)}"
    
    
# Start Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
