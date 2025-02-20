import os
import shutil
import whisper
import yt_dlp
import torch
import tkinter as tk
from tkinter import messagebox
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Check if FFmpeg is installed
if not shutil.which("ffmpeg"):
    messagebox.showerror("Error", "FFmpeg not found! Please install and add it to PATH.")
    exit()

# Load AI models
whisper_model = whisper.load_model("large")  
summarization_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50")

def download_audio(url):
    """Downloads only audio from YouTube and returns the file path."""
    output_file = "downloaded_audio.m4a"

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_file,
        "quiet": True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return output_file
    except Exception as e:
        messagebox.showerror("Error", f"Failed to download audio: {str(e)}")
        return None

def transcribe_audio(audio_path):
    """Transcribes the downloaded audio and detects language."""
    try:
        result = whisper_model.transcribe(audio_path, language=None)  # Auto-detect language
        transcript = result["text"]
        detected_lang = result["language"]
        return transcript, detected_lang
    except Exception as e:
        return f"Error in transcription: {str(e)}", None

def summarize_text(text, lang_code):
    """Summarizes transcribed text in its detected language."""
    if len(text) < 50:
        return "Transcript too short to summarize."

    try:
        tokenizer.src_lang = lang_code  # Set detected language
        inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)

        summary_ids = summarization_model.generate(inputs["input_ids"], max_length=150, min_length=50, length_penalty=2.0)
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error in summarization: {str(e)}"

def process_audio():
    """Handles YouTube URL input, downloads audio, transcribes, and summarizes it."""
    url = entry_url.get().strip()
    
    if not url:
        messagebox.showerror("Error", "Please enter a valid YouTube URL.")
        return

    try:
        status_label.config(text="Downloading audio...", fg="blue")
        app.update_idletasks()
        audio_path = download_audio(url)

        if not audio_path:
            status_label.config(text="Download failed!", fg="red")
            return

        status_label.config(text="Transcribing audio...", fg="blue")
        app.update_idletasks()
        transcript, detected_lang = transcribe_audio(audio_path)

        if detected_lang:
            status_label.config(text=f"Detected Language: {detected_lang.upper()}", fg="blue")
        else:
            detected_lang = "en"  # Default to English

        status_label.config(text="Generating summary...", fg="blue")
        app.update_idletasks()
        summary = summarize_text(transcript, detected_lang)

        status_label.config(text="Summary generated!", fg="green")
        
        # Display results in GUI
        text_transcript.delete("1.0", tk.END)
        text_transcript.insert(tk.END, transcript)

        text_summary.delete("1.0", tk.END)
        text_summary.insert(tk.END, summary)

        text_language.delete("1.0", tk.END)
        text_language.insert(tk.END, detected_lang.upper())

        # Save to file
        with open("audio_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"Detected Language: {detected_lang.upper()}\n\nTranscript:\n{transcript}\n\nSummary:\n{summary}")

        messagebox.showinfo("Success", "Summary saved to 'audio_summary.txt'!")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to process audio: {str(e)}")

# GUI Setup
app = tk.Tk()
app.title("AI-Powered YouTube Audio Summarizer")
app.geometry("700x650")

tk.Label(app, text="Enter YouTube Video URL:", font=("Arial", 12)).pack(pady=5)
entry_url = tk.Entry(app, width=50, font=("Arial", 12))
entry_url.pack(pady=5)

btn_summarize = tk.Button(app, text="Summarize", command=process_audio, font=("Arial", 12), bg="blue", fg="white")
btn_summarize.pack(pady=10)

status_label = tk.Label(app, text="", font=("Arial", 10))
status_label.pack()

# Detected Language Section
tk.Label(app, text="Detected Language:", font=("Arial", 12, "bold")).pack(pady=5)
text_language = tk.Text(app, height=1, width=20, font=("Arial", 10))
text_language.pack(pady=5)

# Transcript Section
tk.Label(app, text="Transcript:", font=("Arial", 12, "bold")).pack(pady=5)
text_transcript = tk.Text(app, height=10, width=80, font=("Arial", 10), wrap="word")
text_transcript.pack(pady=5)

# Summary Section
tk.Label(app, text="Summary:", font=("Arial", 12, "bold")).pack(pady=5)
text_summary = tk.Text(app, height=5, width=80, font=("Arial", 10), wrap="word")
text_summary.pack(pady=5)

app.mainloop()
