from flask import Flask, request, render_template, send_file, jsonify
from rule_base import risk_rule_base,checkbox_rule_base
from predictor import set_init,inference
import base64
import json

template_dir = '../html/templates'
static_dir = '../html/static'
app = Flask(__name__, template_folder=template_dir, static_url_path='/static', static_folder=static_dir)

tokenizer,estimator = set_init()

@app.route('/ZhongYin_api', methods=['POST'])
def run_ZhongYin_api():
    json_list = []
    if request.method == 'POST':
        if 'filename' not in request.files:
            json_list = [{'image': 'there is no filename in form!'}]            
        jsonfile = json.loads(request.form.get('data'))
        predict_data = jsonfile["ad"]
        state,risk_rate = risk_rule_base(predict_data)
        print('risk_rate',risk_rate)
        if state =="Illegal":
            prob,keywords = inference(predict_data,tokenizer,estimator) # 單測試句子 
            prob = 100.0
        elif risk_rate > 0: #高風險加權
            prob,keywords = inference(predict_data,tokenizer,estimator) # 單測試句子 
            if prob < 50:
                prob = (prob+50)*0.5
        elif risk_rate < 0: #低風險加權
            prob,keywords = inference(predict_data,tokenizer,estimator) # 單測試句子  
            if prob > 50:
                prob = (prob-50)*0.5 
        else:
            prob,keywords = inference(predict_data,tokenizer,estimator) # 單測試句子    
        rulebase = "欲刊登廣告內容，違反藥事法第69條之機率 : "+str(prob)+'%'
        # rulebase = checkbox_rule_base(predict_data,prob) # checkbox rule base system
    json_list = [{"result":rulebase,"keywords":keywords}]
    print(json_list)
    return jsonify(json_list)

@app.route('/', methods=['GET'])
def run_app():
    return render_template('index.html')

if __name__ == "__main__": 
    #app.run(debug=True, host='0.0.0.0', port=5555, ssl_context='adhoc')
    app.run(debug=True)
