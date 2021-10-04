# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 19:16:45 2021

@author: Alan Lin
"""

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import os
import jieba
import jieba.analyse
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from scipy.special import softmax


OUTPUT_DIR = './weight' #model file path
label_list = [0, 1]
MAX_SEQ_LENGTH = 256 #128
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_chinese_L-12_H-768_A-12/1"
PREPROCESS_MODEL = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
class_weight = np.array([2, 2])/2

# set GPU memory
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement = True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.5)
config.gpu_options.allow_growth = True
sess0 = tf.InteractiveSession(config = config)

def create_tokenizer_from_hub_module():
    with tf.Graph().as_default():
      os.environ['TFHUB_CACHE_DIR'] = './tf_cache' # set Cache dir
      bert_module = hub.Module(BERT_MODEL_HUB)
      tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
      with tf.Session() as sess:
        vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],tokenization_info["do_lower_case"]])
    return bert.tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

def set_init():
    tokenizer = create_tokenizer_from_hub_module()
    # load BERT HUB MODEL
    estimator = create_estimator()
    return tokenizer,estimator
    
def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,num_labels):
    bert_module = hub.Module(
        BERT_MODEL_HUB,
        trainable=False)
    bert_inputs = dict(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids)
    bert_outputs = bert_module(
        inputs=bert_inputs,
        signature="tokens",
        as_dict=True)    
    output_layer = bert_outputs["pooled_output"]
    hidden_size = output_layer.shape[-1]
    # Create our own layer to tune for politeness data.
    output_weights = tf.get_variable("output_weights", [num_labels, hidden_size],initializer=tf.truncated_normal_initializer(stddev=0.02))
    output_bias = tf.get_variable("output_bias", [num_labels], initializer=tf.zeros_initializer())
    with tf.variable_scope("loss"):
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9) # Dropout helps prevent overfitting
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32) # Convert labels into one-hot encoding
        predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
        # If we're predicting, we want predicted labels and the probabiltiies.
        if is_predicting:
          return (predicted_labels, log_probs)
        # If we're train/eval, compute loss between predicted and actual label
        per_example_loss = -tf.reduce_sum(class_weight * one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, predicted_labels, log_probs)

def model_fn_builder(num_labels, learning_rate, num_train_steps,num_warmup_steps):
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)
        # TRAIN and EVAL
        if not is_predicting:
            (loss, predicted_labels, log_probs) = create_model(is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)
            train_op = bert.optimization.create_optimizer(loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)
            # Calculate evaluation metrics. 
            def metric_fn(label_ids, predicted_labels):
              accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
              true_pos = tf.metrics.true_positives(
                  label_ids,
                  predicted_labels)
              true_neg = tf.metrics.true_negatives(
                  label_ids,
                  predicted_labels)   
              false_pos = tf.metrics.false_positives(
                  label_ids,
                  predicted_labels)  
              false_neg = tf.metrics.false_negatives(
                  label_ids,
                  predicted_labels)      
              return {
                  "eval_accuracy": accuracy,
                  "true_positives": true_pos,
                  "true_negatives": true_neg,
                  "false_positives": false_pos,
                  "false_negatives": false_neg,
                  "Precision": precision,
                  "Recall": recall,
                  "AUC": auc
                  }
            eval_metrics = metric_fn(label_ids, predicted_labels)
        
            if mode == tf.estimator.ModeKeys.TRAIN:
              return tf.estimator.EstimatorSpec(mode=mode,
                loss=loss,
                train_op=train_op)
            else:
                return tf.estimator.EstimatorSpec(mode=mode,
                  loss=loss,
                  eval_metric_ops=eval_metrics)
        else:
            (predicted_labels, log_probs) = create_model(is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)
            predictions = {
                'probabilities': log_probs,
                'labels': predicted_labels
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    return model_fn

def create_estimator():       
    print("--------------------------------build estimator--------------------------------")
    # set hyperparameters
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    # Model configs
    SAVE_CHECKPOINTS_STEPS = 300
    SAVE_SUMMARY_STEPS = 100  
    # Compute train and warmup steps from batch size
    num_train_steps = 40
    num_warmup_steps = 4
    # Specify output directory and number of checkpoint steps to save
    run_config = tf.estimator.RunConfig(
        model_dir=OUTPUT_DIR,
        save_summary_steps=SAVE_SUMMARY_STEPS,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)
    #Initializing the model and the estimator
    model_fn = model_fn_builder(
      num_labels=len(label_list),
      learning_rate=LEARNING_RATE,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps)   
    estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config,
      params={"batch_size": BATCH_SIZE})
    print("--------------------------------estimator done--------------------------------")
    return estimator

def getPrediction(in_sentences,tokenizer,estimator):
    #A list to map the actual labels to the predictions
    labels = ['legal','Illegal']
    #Transforming the test data into BERT accepted form
    input_examples = [run_classifier.InputExample(guid="", text_a = in_sentences, text_b = None, label = 0) for x in in_sentences]
    #Creating input features for Test data
    input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
    #Predicting the classes 
    predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)  
    predictions = estimator.predict(predict_input_fn)
    return [(sentence, prediction['probabilities'],prediction['labels'], labels[prediction['labels']]) for sentence, prediction in zip(in_sentences, predictions)]

def inference(pred_data,tokenizer,estimator):
    prob_keyword = {} #存softmax後機率
    score_keyword = {} #存原始預測np to list
    sum_keyword={} #存原始預測np和
    sum_rate = 0
    keywords={} #存output 字分數
    results = getPrediction(pred_data,tokenizer,estimator)
    for i in range(len(results)):
        prob = list(softmax(results[i][1]))
        prob_keyword[results[i][0]] = prob[1]  # 預測為1之機率
        score = list(results[i][1])
        score_keyword[results[i][0]] = score
    sum_prob = sum(prob_keyword.values())
    avg = sum_prob/len(prob_keyword.items()) #預測平均分數
    for key,i in zip(score_keyword.keys(),score_keyword.values()):
        sum_keyword[key] = sum(i)
        sum_rate += sum(i)
    sorted_list = sorted(sum_keyword.items(), key=lambda d: d[1]) #機率排序
    # print('\nsorted_list :\n',sorted_list)
    rate = [ (v[0],v[1]/sum_rate) for v in sorted_list] #算字分數
    for item in rate:
        keywords[item[0]] = round(item[1]*100,2)
    # word = jieba.lcut(pred_data) #jieba 分詞
    tags = jieba.analyse.extract_tags(pred_data, topK=5) #jieba 抓關鍵字
    return round(avg*100,1),keywords,tags

# test predictor
# text2 = '天氣真好，做完了準備回家搂。' #label 0
# # # test2 = '包含適合使用支氣管擴張劑及皮質類固醇組合療法之患有氣喘的兒童與成人，嚴重慢性阻塞性肺部疾病，慢性支氣管炎合肺氣腫，嚴重氣喘，肝功能或腎不全之患者，副作用：頭痛、肌肉痙攣、關節痛、口腔與喉嚨的念珠病菌、肺炎，合併治療，可以協助快速緩解症狀，降低AUR等疾病惡化風險，DUODART，簡易處方資訊，治療具有症狀且攝護腺增大之攝護腺肥大症的第二線治療，減少攝護腺體積、改善尿流速率之效果，Avodart適尿通，治療具有症狀之攝護腺肥大者。而有緩解相關症狀、降低急性尿滯留之發生率、減少攝護腺肥大症相關手術必要性之效果' #label 1
# from predictor import set_init,inference
# tokenizer,estimator = set_init()
# prob,keywords,tags = inference(text2,tokenizer,estimator)
# print('預測違規機率:',prob,"%\n",tags,keywords)
