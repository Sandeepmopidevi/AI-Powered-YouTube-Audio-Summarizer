import os
import shutil
import whisper
import yt_dlp
import threading
from transformers import pipeline
import tkinter as tk
from tkinter import messagebox

# Check if FFmpeg is installed
if not shutil.which("ffmpeg"):
    messagebox.showerror("Error", "FFmpeg not found! Please install and add it to PATH.")
    exit()

# Load AI models
whisper_model = whisper.load_model("base")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def download_audio(url):
    """Downloads only audio from YouTube using yt-dlp and returns the file path."""
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
    """Transcribes the downloaded audio using Whisper AI."""
    try:
        result = whisper_model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        return f"Error in transcription: {str(e)}"

def summarize_text(text):
    """Summarizes the transcribed text using a transformer model."""
    if len(text) < 50:
        return "Transcript too short to summarize."

    try:
        summary = summarizer(text, max_length=100, min_length=50, do_sample=False)[0]["summary_text"]
        return summary
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
        transcript = transcribe_audio(audio_path)

        status_label.config(text="Generating summary...", fg="blue")
        app.update_idletasks()
        summary = summarize_text(transcript)

        status_label.config(text="Summary generated!", fg="green")
        text_summary.delete("1.0", tk.END)
        text_summary.insert(tk.END, f"Transcript:\n{transcript}\n\nSummary:\n{summary}")

        # Save to file
        with open("audio_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"Transcript:\n{transcript}\n\nSummary:\n{summary}")

        messagebox.showinfo("Success", "Summary saved to 'audio_summary.txt'!")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to process audio: {str(e)}")

# GUI Setup
app = tk.Tk()
app.title("AI Audio Summarizer")
app.geometry("600x500")

tk.Label(app, text="Enter YouTube Video URL:", font=("Arial", 12)).pack(pady=5)
entry_url = tk.Entry(app, width=50, font=("Arial", 12))
entry_url.pack(pady=5)

btn_summarize = tk.Button(app, text="Summarize", command=process_audio, font=("Arial", 12), bg="blue", fg="white")
btn_summarize.pack(pady=10)

status_label = tk.Label(app, text="", font=("Arial", 10))
status_label.pack()

text_summary = tk.Text(app, height=15, width=70, font=("Arial", 10), wrap="word")
text_summary.pack(pady=10)

app.mainloop()
