# 🎙️ AI-Powered YouTube Audio Summarizer  

🚀 **Extracts and summarizes YouTube video content by processing only the audio.**  

## 📌 Overview  
This AI-powered tool **downloads only the audio** from a YouTube video, transcribes it using **Whisper AI**, and generates a **concise summary** using **AI-based text summarization**.  

## 🔥 Features  
✅ **Downloads Only Audio** – No need to download the full video  
✅ **Speech-to-Text Transcription** – Converts spoken words into text using Whisper AI  
✅ **AI-Powered Summarization** – Generates a short and meaningful summary  
✅ **Separate Sections for Transcript & Summary** – Clear and structured output in GUI  
✅ **Saves Summary to a Text File** – For easy reference  
✅ **User-Friendly GUI** – Simple interface with buttons for summarization  

## 📂 Installation  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/Sandeepmopidevi/AI-Powered-YouTube-Audio-Summarizer.git
cd AI-Powered-YouTube-Audio-Summarizer
```

### 2️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
```

### 3️⃣ Ensure FFmpeg is Installed  
- Download and install **FFmpeg** from [FFmpeg.org](https://ffmpeg.org/download.html)  
- Add FFmpeg to your system's **PATH**  

## 🚀 How to Use  

1️⃣ **Run the Application:**  
```bash
python app.py
```

2️⃣ **Enter the YouTube Video URL** in the GUI  

3️⃣ **Click "Summarize"** – The tool will:  
   - Download only the audio  
   - Transcribe the speech  
   - Generate a summarized version  

4️⃣ **View Transcript & Summary** in separate sections  

5️⃣ **Save Summary to File** automatically  

## 🛠️ Requirements  
- Python 3.8+  
- FFmpeg  
- Whisper AI  
- Transformers  
- yt-dlp  
- MoviePy  
- OpenAI CLIP  
- Tkinter (for GUI)  

## 📜 License  
This project is licensed under the **MIT License**.  

## 🤝 Contributing  
Pull requests are welcome! If you have suggestions or improvements, feel free to contribute.