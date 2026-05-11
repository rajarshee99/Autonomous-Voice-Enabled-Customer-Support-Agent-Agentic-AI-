import requests

API_KEY = ""  # <-- put your new key here
URL = "https://api.groq.com/openai/v1/audio/transcriptions"


def _transcribe_with_files(files_payload):
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }

    try:
        response = requests.post(URL, headers=headers, files=files_payload)
        response.raise_for_status()
        result = response.json()
        return result["text"]
    except Exception as e:
        print("Error:", e)
        return None

def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        files = {
            "file": (audio_file.name, audio_file, "application/octet-stream"),
            "model": (None, "whisper-large-v3")
        }
        return _transcribe_with_files(files)


def transcribe_audio_file(uploaded_file):
    uploaded_file.stream.seek(0)
    filename = uploaded_file.filename or "voice-input.webm"
    content_type = uploaded_file.mimetype or "application/octet-stream"

    files = {
        "file": (filename, uploaded_file.stream, content_type),
        "model": (None, "whisper-large-v3")
    }
    text = _transcribe_with_files(files)
    uploaded_file.stream.seek(0)
    return text


if __name__ == "__main__":
    audio_file = "sample.wav"  # <-- your audio file
    text = transcribe_audio(audio_file)

    if text:
        print("Transcribed Text:")
        print(text)

