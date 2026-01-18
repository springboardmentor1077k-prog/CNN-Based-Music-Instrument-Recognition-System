from backend.inference import load_audio, predict

def run_inference(audio_path: str):
    audio = load_audio(audio_path)
    results = predict(audio)
    return results
