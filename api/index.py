from flask import Flask, request, jsonify
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, KeywordsOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

app = Flask(__name__)

authenticator = IAMAuthenticator('GFqSHawAvnoLmgTyS4YG7geo8Ob3h8KLBZLRFtPKTKj7')
nlu = NaturalLanguageUnderstandingV1(
    version='2021-08-01',
    authenticator=authenticator
)
nlu.set_service_url('https://api.au-syd.natural-language-understanding.watson.cloud.ibm.com/instances/b4542792-15e1-4cee-8e98-0ed255cd2a74')

@app.route('/api/webhook', methods=['POST'])
def webhook():
    data = request.get_json()
    text = data['userRequest']['utterance']

    response = nlu.analyze(
        text=text,
        features=Features(keywords=KeywordsOptions())
    ).get_result()

    keywords = [keyword['text'] for keyword in response['keywords']]
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
    app.run(host='0.0.0.0', port=8000, debug=True)

