# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 16:51:37 2021

@author: Alan Lin
"""

def in_sentence(given, sub):
    return sub in given

def risk_rule_base(sentence):
    risk_word = "./data/risk.txt"
    h_risk_word = "./data/high_risk_words.txt"
    l_risk_word = "./data/low_risk_words.txt"
    state,s_H,s_L = False,False,False
    class_label = {True:'Illegal',False:'legal'}
    class_H_risk = {True:1,False:0}
    class_L_risk = {True:-1,False:0}
    with open(risk_word, "r", encoding = "utf-8") as Risk,\
            open(h_risk_word, "r", encoding = "utf-8") as H_R,\
                open(l_risk_word, "r", encoding = "utf-8") as L_R:
        risk = []
        high_risk = []
        low_risk = []
        for r in Risk.readlines():
            r = r.replace('\n', '')
            risk.append(r)
        for h in H_R.readlines():
            h = h.replace('\n', '')
            high_risk.append(h)
        for l in L_R.readlines():
            l = l.replace('\n', '')
            low_risk.append(l)      
    # Absolutely illegal
    for i in risk:
        if not state:
            state = state or in_sentence(sentence,i)
        else:
            break # state = True
    # High risk Evaluate
    for i in high_risk:
        if not s_H:
            s_H = s_H or in_sentence(sentence,i)
        else:
            break # s_H = True
    # Low risk Evaluate
    for i in low_risk:
        if not s_L:
            s_L = s_L or in_sentence(sentence,i)
        else:
            break # s_L = True
    rate = class_H_risk[s_H]+class_L_risk[s_L]  # {rate｜ < 0 : low risk, = 0 : nothing, > 0 : high risk}
    return class_label[state] , rate

def checkbox_rule_base(jsonData,prob): 
    cb1 = jsonData['checkbox1']
    cb2 = jsonData['checkbox2']
    cb3 = jsonData['checkbox3']
    it1 = jsonData['inputtext1']
    it2 = jsonData['inputtext2']
    result = ["欲刊登廣告內容，違反藥事法第69條之機率 : ",prob,'%']
    if cb1 == 'Yes':
        pass
    else:
        result.append('請注意! 你有可能已經違返藥事法第65條，未依法申請販賣藥商許可證，不得為藥物廣告行為。')
    if cb2 == 'Yes':
        pass
    else:
        result.append('請注意! 你有可能已經違返藥事法第66-1條，未經事先核准即擅自刊登廣。')
    if cb3 == 'Yes':
        pass
    else:
        result.append('請注意! 需於有效期限前30日內申請展延，以免違反藥事法第66-1條。')
    if it1 == it2:
        pass
    else:
        result.append('請注意!若擅自更改原審內容或超出規定範圍時，可能違反藥事法第66-2條廣告與表定內容不符')
    return result

# test risk_rule_base
# text = '我們的產品根治你的毛病'
# ans,evl = risk_rule_base(text)
# print("ANS : ",ans,evl)