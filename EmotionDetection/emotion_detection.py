import requests
import json

def emotion_detector(text_to_analyze):
    url = (
        "https://sn-watson-emotion.labs.skills.network/"
        "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
    )

    headers = {"grpc-metadata-mm-model-id": "emotion_english_distilroberta_base"}
    myobj = {"raw_document": {"text": text_to_analyze}}

    try:
        response = requests.post(url, headers=headers, json=myobj, timeout=10)
    except requests.exceptions.RequestException:
        # Fallback (lets you test locally when the endpoint is unreachable)
        return {
            "anger": 0.0,
            "disgust": 0.0,
            "fear": 0.0,
            "joy": 0.0,
            "sadness": 0.0
        }

    formatted_response = json.loads(response.text)
    emotions = formatted_response["emotionPredictions"][0]["emotion"]

    return {
        "anger": emotions["anger"],
        "disgust": emotions["disgust"],
        "fear": emotions["fear"],
        "joy": emotions["joy"],
        "sadness": emotions["sadness"]
    }
