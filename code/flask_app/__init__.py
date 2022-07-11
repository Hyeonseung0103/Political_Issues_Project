from time import time
import time
from flask import Flask, jsonify, request
import torch
import json
from transformers import PreTrainedTokenizerFast
from transformers.models.bart import BartForConditionalGeneration

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False # 한글깨지는거

# Fine tuning한 KoBART 모델 불러오기
model = BartForConditionalGeneration.from_pretrained('./flask_app/kobart_summary')

#hugging face에 있는 토크나이저 불러오기
tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')

# 사용자에게서 문서를 입력 받으면 KoBART 모델이 요약한 요약본을 리턴해주는 함수
@app.route("/summary", methods = ['GET', 'POST'])
def return_summary():
#    sentence = request.args.get('data') 
    sentence = request.get_json()['body']['data'][:500]

    s = time.time()

    input_ids = tokenizer.encode(sentence)
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.unsqueeze(0)
    output = model.generate(input_ids, eos_token_id=1, max_length=64, num_beams=2)
    output = tokenizer.decode(output[0], skip_special_tokens=True)

    print(time.time() - s)
    return jsonify({'data': output})
    
