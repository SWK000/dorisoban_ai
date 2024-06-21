from flask import Flask, request, jsonify
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, KeywordsOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import os

app = Flask(__name__)

# 환경 변수에서 IBM Watson API 키와 서비스 URL 가져오기
IBM_API_KEY = 'GFqSHawAvnoLmgTyS4YG7geo8Ob3h8KLBZLRFtPKTKj7'
IBM_SERVICE_URL = 'https://api.au-syd.natural-language-understanding.watson.cloud.ibm.com/instances/b4542792-15e1-4cee-8e98-0ed255cd2a74'

authenticator = IAMAuthenticator(IBM_API_KEY)
nlu = NaturalLanguageUnderstandingV1(
    version='2021-08-01',
    authenticator=authenticator
)
nlu.set_service_url(IBM_SERVICE_URL)

@app.route('/api/webhook', methods=['POST'])
def webhook():
    data = request.get_json()
    text = data['userRequest']['utterance']

    response = nlu.analyze(
        text=text,
        features=Features(keywords=KeywordsOptions())
    ).get_result()

    keywords = [keyword['text'] for keyword in response.get('keywords', [])]
    reply = f"You mentioned: {', '.join(keywords)}"

    return jsonify({
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": reply,
                    },
                },
            ],
        },
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
