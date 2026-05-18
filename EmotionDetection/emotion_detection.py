# emotion_detection.py
import requests

_EMOTION_URL = "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
_HEADERS = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}

def emotion_detector(text_to_analyze: str):
    """
    Send text_to_analyze to the Watson EmotionPredict endpoint and
    return the 'text' attribute from the response object.
    """
    if not isinstance(text_to_analyze, str):
        raise TypeError("text_to_analyze must be a string")

    payload = {"raw_document": {"text": text_to_analyze}}
    resp = requests.post(_EMOTION_URL, json=payload, headers=_HEADERS, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    # Return the span text if present
    try:
        return data["emotionPredictions"][0]["emotionMentions"][0]["span"]["text"]
    except Exception:
        # Fallback: return raw_document.text if present
        if isinstance(data.get("raw_document"), dict) and "text" in data["raw_document"]:
            return data["raw_document"]["text"]
        # Last resort: return the whole JSON as a string
        return str(data)
