import requests

API_URL = "http://127.0.0.1:8000/analyze"


def analyze_audio(audio_path, segment_length, aggregation):
    """
    Sends audio file to FastAPI backend and returns JSON result
    """

    with open(audio_path, "rb") as f:
        files = {
            "file": f
        }

        data = {
            "segment_length": str(segment_length),
            "aggregation": aggregation
        }

        response = requests.post(API_URL, files=files, data=data)

    if response.status_code != 200:
        raise RuntimeError(
            f"Backend error {response.status_code}: {response.text}"
        )

    return response.json()
