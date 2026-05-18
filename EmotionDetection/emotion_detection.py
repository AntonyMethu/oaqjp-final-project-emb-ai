import requests

_EMOTION_URL = "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
_HEADERS = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}

def _extract_emotions_from_response(data):
    try:
        emo = data["emotionPredictions"][0].get("emotion")
        if isinstance(emo, dict):
            return {k: float(emo.get(k, 0.0)) for k in ("anger", "disgust", "fear", "joy", "sadness")}
    except Exception:
        pass

    try:
        mentions = data["emotionPredictions"][0].get("emotionMentions", [])
        if mentions and isinstance(mentions, list):
            for m in mentions:
                if "emotion" in m and isinstance(m["emotion"], dict):
                    return {k: float(m["emotion"].get(k, 0.0)) for k in ("anger", "disgust", "fear", "joy", "sadness")}
    except Exception:
        pass

    return {k: 0.0 for k in ("anger", "disgust", "fear", "joy", "sadness")}

def emotion_detector(text_to_analyze: str):
    if not isinstance(text_to_analyze, str):
        raise TypeError("text_to_analyze must be a string")

    payload = {"raw_document": {"text": text_to_analyze}}
    resp = requests.post(_EMOTION_URL, json=payload, headers=_HEADERS, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    emotions = _extract_emotions_from_response(data)
    dominant = max(emotions.items(), key=lambda kv: kv[1])[0]

    return {
        "anger": emotions["anger"],
        "disgust": emotions["disgust"],
        "fear": emotions["fear"],
        "joy": emotions["joy"],
        "sadness": emotions["sadness"],
        "dominant_emotion": dominant
    }
