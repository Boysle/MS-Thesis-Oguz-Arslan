# -*- coding: utf-8 -*-
"""
Created on Wed May  1 13:09:58 2024

@author: USER
"""

import argparse, sys
from dataset_processor import DatasetProcessor
import numpy as np
import pandas as pd
import os
import evaluate
import torch
import gc
from trainer import TrainerForClassification, TrainerForConditionalGeneration, TrainerForGeneration
from datasets import Dataset
from transformers import AutoTokenizer

   
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    
    parser.add_argument("--dataset_name", help = """Name of the dataset, this dataset's train split will be used for training valid split will be used
                        for validation and test split will be used for final evaluation""", type = str, default = "emotion_single")
                        
    parser.add_argument("--task", help="""Task to be done, it can be anything from: classification, multi_label_classification, summarization, paraphrasing, title_generation
                        nli, semantic_similarity, ner, pos_tagging, question_answering, classification_by_generation, and question_generation""", type = str, default = "classification")
                        
    parser.add_argument("--task_format", help="""It can be either classification, conditional generation or generation""", type = str, default = "classification")
    
    parser.add_argument("--num_labels", help="""Number of labels, only used if the task format is classification""", type = int, default = 2)
    
    parser.add_argument("--model_keyword", help="""It can be any of the following: BERT, ROBERTA, BERTURK, mT5, mBART, TURNA, TurkishBERTweet, kanarya2b and kanarya750m""",
                        type = str, default = "TURNA")
    
    parser.add_argument("--max_input_length", help="""It determines the maximum input length, longer inputs will be truncated""", type = int, default = 256)
    
    parser.add_argument("--max_target_length", help="""It determines the maximum target length, it is also equal to the max new tokens in the generation""",
                        type = int, default = 128)
    
    parser.add_argument("--instruction_amount", help="""It should be equal to the number of different instructions for the same task""",
                        type = int, default = 1)
    
    parser.add_argument("--num_train_epochs", help="""Maximum number of epochs for training""", type = int, default = 1)
    
    parser.add_argument("--early_stopping_patience", help="""Training will stop if the eval loss doesn't decrease in this amount of
                        consecutive evaluation steps""", type = int, default = -1)
    
    parser.add_argument("--per_device_train_batch_size", help="""per device batch size during training""", type = int, default = 8)
    
    parser.add_argument("--per_device_eval_batch_size", help="""per device batch size during evaluation""", type = int, default = 8)
    
    parser.add_argument("--hf_token_hub", help="""token for huggingface""",  default = None)

    parser.add_argument("--eval_do_concat_batches", help = "whether to do nested concatenation for predictions during evaluation, it should be True normally except kanarya", default = True)

    parser.add_argument("--eval_per_epoch", help = "How many times evaluation should be done in an epoch", type = int, default = 3)

    parser.add_argument("--push_model_to_hub", help = """This flag should be used when the model is to be pushed to hf hub, this flag doesn't take any more arguments,
       if this flag is there in the command it means push_model_to_hub is set to true""", action = 'store_true')
    
    parser.add_argument("--hf_model_repo_name", help = "Repo name for model if push model to hub is true",  default = None)

    parser.add_argument("--private_data", help = """This flag should be used when the data is private, this flag doesn't take any more arguments,
       if this flag is there in the command it means private_data is set to true""", action = 'store_true')

    parser.add_argument("--scheduler", help = """If this flag is present a linear scheduler with 0.1 warmup is used""", action = 'store_true')

    args=parser.parse_args()
    
    
    model_name_dict = {"BERTURK": "dbmdz/bert-base-turkish-cased", "mT5": "google/mt5-large", "mBART": "facebook/mbart-large-cc25", "TURNA": "boun-tabi-LMG/TURNA",
                       "kanarya2b": "asafaya/kanarya-2b", "kanarya750m": "asafaya/kanarya-750m", "TurkishBERTweet": "VRLLab/TurkishBERTweet",
                      "BERT": "google-bert/bert-base-cased", "ROBERTA": "FacebookAI/roberta-base"}
    dataset_name = args.dataset_name
    task = args.task
    task_mode = ''    # either '', '[NLU]', '[NLG]', '[S2S]'
    task_format = args.task_format
    num_labels = args.num_labels
    model_keyword = args.model_keyword
    model_name = model_name_dict[model_keyword]
    
    max_input_length = args.max_input_length   
    #BERTURK, TurkishBERTweet, mBART için batch size 32 , TURNA ve mT5 için 8
    #BERTURK için 512 ttc4900'de çünkü modelin maxı o, mT5 da öyle, mBART'ın maxı 1024, TurkishBERTweet de 512, TURNA da 512 
    #o yüzden max input length 512 TTC için, MLSUM ve TRNews de 512 olacak 
    #offenseval 128 input length, TSATweets 64 input length, FactCheck 256 input length
    #(LLMler için full 1024 kullanmak okay (instruction ve türkçe özel olmayan tokenizerları baz alarak))
    max_target_length = args.max_target_length 
    #max_target mlsum için 128 (llmler için de 128), 
    instruction_number = args.instruction_amount
    
    model_numbers = {"TURNA": "11", "mT5": "12", "mBART": "13", "BERTURK": "14", "TurkishBERTweet": "15"}
    
    if dataset_name.startswith("ttc"):
        dataset_number = 16
    
    elif dataset_name.startswith("offenseval"):
        dataset_number = 17
    
    elif dataset_name.startswith("TSATweets"):
        dataset_number = 18
    
    elif dataset_name.startswith("factcheck"):
        dataset_number = 19
    
    elif dataset_name.startswith("mlsum"):
        dataset_number = 20
        
    elif dataset_name.startswith("trnews"):
        dataset_number = 21

    else: dataset_number = -1
        
    if task_format == "classification":
        task_number = 0
    
    else:
        task_number = 1
        
    run_name = model_numbers[model_keyword]+"-"+str(dataset_number)+"-"+str(task_number)+"-2"
    
    #if "no_instruction" in dataset_name:
    #    run_name += "-1"
    
   # elif "single_instruction" in dataset_name:
    #    run_name += "-2"
    
    #elif "multi_instruction" in dataset_name:
    #    run_name += "-3"
    
    early_stopping_patience = args.early_stopping_patience
    eval_per_epoch = args.eval_per_epoch
    pred_hf_repo_name = "Holmeister/"+run_name+"-preds"
    hf_token = args.hf_token_hub
    push_model_to_hub = args.push_model_to_hub
    hf_model_repo_name = args.hf_model_repo_name
    private = args.private_data
    scheduler = args.scheduler
    if model_keyword == "TurkishBERTweet":
       TurkishBERTweet_preprocess = True
    else:
       TurkishBERTweet_preprocess = False
       
    dataset_processor = DatasetProcessor(
            dataset_name=dataset_name, task=task, task_format=task_format, task_mode=task_mode,
            tokenizer_name=model_name, max_input_length=max_input_length, max_target_length=max_target_length,
            private = private, token = hf_token, TurkishBERTweet_preprocess = TurkishBERTweet_preprocess
    )
    
    train_dataset = dataset_processor.load_and_preprocess_data(split='train')
    eval_dataset = dataset_processor.load_and_preprocess_data(split='validation')
    test_dataset = dataset_processor.load_and_preprocess_data(split="test")
    training_params = {
        'num_train_epochs': args.num_train_epochs,
        'per_device_train_batch_size': args.per_device_train_batch_size,
        'per_device_eval_batch_size': args.per_device_eval_batch_size,
        'output_dir': run_name,
        'evaluation_strategy': 'steps',  #bu ve altındaki epoch da olabilir
        'save_strategy': 'steps',
        "eval_steps": int(np.ceil(len(train_dataset)/(args.per_device_train_batch_size*eval_per_epoch))),
        "save_steps": int(np.ceil(len(train_dataset)/(args.per_device_train_batch_size*eval_per_epoch))),
        'logging_steps': int(np.ceil(len(train_dataset)/args.per_device_train_batch_size)),  #böldüğümüz sayı batch size'a eşit
        #'predict_with_generate': True,
        "report_to": "none",
        "eval_do_concat_batches": args.eval_do_concat_batches,
        "hub_token": hf_token,
        "hub_model_id": hf_model_repo_name
    }
    if task_format == "conditional_generation" or task_format == "generation":
       training_params["predict_with_generate"] = True
       
    optimizer_parameters = {"BERTURK": {'optimizer_type': 'adamw', 'scheduler': True,"lr": 2e-5 }, "mT5": {'optimizer_type': 'adafactor', 'scheduler': False,"lr": 1e-3 },
                            "mBART": {'optimizer_type': 'adamw', 'scheduler': True,"lr": 2e-5 }, "TURNA": {'optimizer_type': 'adafactor', 'scheduler': False,"lr": 1e-3 },
                       "kanarya2b": {'optimizer_type': 'adamw', 'scheduler': True,"lr": 2e-5 }, "kanarya750m": {'optimizer_type': 'adamw', 'scheduler': True,"lr": 2e-5 },
                           "TurkishBERTweet": {'optimizer_type': 'adamw', 'scheduler': True,"lr": 2e-5 }, "BERT": {'optimizer_type': 'adamw', 'scheduler': scheduler,"lr": 2e-5 },
                           "ROBERTA": {'optimizer_type': 'adamw', 'scheduler': scheduler,"lr": 2e-5 } }
    
    optimizer_params = optimizer_parameters[model_keyword]
    #BERT, gptj ve mBART modeller için 1e-5, 2e-5, 3e-5, 4e-5, 5e-5 ve linear scheduler with warmup, TURNA ve mt5 için 1e-3 ve no scheduler
    
    def metrics_per_instruction(df, inst_number = instruction_number, task="sequence_classification"):
      preds = list(df["Prediction"])
      labels = list(df["Label"])
      metrics = []
      results = []
      if task == "sequence_classification":
        accuracy = evaluate.load("accuracy")
        precision = evaluate.load("precision")
        recall = evaluate.load("recall")
        f1 = evaluate.load("f1")
        metrics = [accuracy, precision, recall, f1]
    
      for i in range(inst_number):
        result = {}
        for metric in metrics:
          try:scores = metric.compute(predictions = preds[int(i*len(preds)/inst_number):int((i+1)*len(preds)/inst_number)], references = labels[int(i*len(preds)/inst_number):int((i+1)*len(preds)/inst_number)], average = "macro")
          except: scores = metric.compute(predictions = preds[int(i*len(preds)/inst_number):int((i+1)*len(preds)/inst_number)], references = labels[int(i*len(preds)/inst_number):int((i+1)*len(preds)/inst_number)])
          result.update(scores)
        results.append(result)
    
      results_df= pd.DataFrame.from_dict(results)
    
      return results_df
    
    #learning_rates = {"BERTURK": [1e-5, 2e-5, 4e-5] , "mT5": [5e-5, 1e-4, 2e-4], "mBART": [1e-5, 2e-5, 4e-5] , "TURNA": [5e-4, 1e-3, 2e-3],
    #                   "kanarya2b": [2e-5, 4e-5, 6e-5] , "kanarya750m": [2e-5, 4e-5, 6e-5], "TurkishBERTweet": [1e-5, 2e-5, 4e-5],
    #                 "BERT": [2e-5, 3e-5, 4e-5, 5e-5], "ROBERTA": [2e-5, 3e-5, 4e-5, 5e-5]}
    
    learning_rates_cls = {"BERTURK": [5e-5] , "mT5": [1e-3], "mBART": [5e-5] , "TURNA": [1e-3], "TurkishBERTweet": [5e-5]}
    learning_rates_cond_gen = {"mT5": [1e-3], "mBART": [5e-5] , "TURNA": [1e-3]}
    
    if task_format == "classification":
        for i in range(1):  #normalde burası 3 olacak 3 run için
          for lr in learning_rates_cls[model_keyword]:  #normalde burada [1:2] olmayacak farklı learning rateler için
            
            optimizer_params["lr"] = lr
            model_trainer = TrainerForClassification(
              model_name=model_name, task=task,
              training_params=training_params,
              optimizer_params=optimizer_params,
              model_save_path=run_name,
              num_labels = num_labels,
              postprocess_fn=dataset_processor.dataset.postprocess_data)
            trainer, model = model_trainer.train_and_evaluate(train_dataset, eval_dataset, test_dataset, early_stopping_patience = early_stopping_patience)
            
            if push_model_to_hub == True:
               trainer.push_to_hub()
             
            with torch.no_grad():
               del model_trainer
               del trainer
               del model
               gc.collect()
            preds_df = pd.read_csv(os.path.join(training_params['output_dir'], 'predictions.csv'))
            predictions_hf = Dataset.from_pandas(preds_df)
            predictions_hf.push_to_hub(pred_hf_repo_name, private = True, token = hf_token)
            #results_df = metrics_per_instruction(preds_df, inst_number = instruction_number, task = "sequence_classification")
            #results_df.to_csv(str(run_name)+"_"+str(lr)+"_results"+str(i)+".csv", index = False)
            
    elif task_format == "conditional_generation":
        for i in range(1):  #normalde burası 3 olacak 3 run için
          for lr in learning_rates_cond_gen[model_keyword]:  #normalde burada [1:2] olmayacak farklı learning rateler için
            optimizer_params["lr"] = lr
            model_trainer = TrainerForConditionalGeneration(
              model_name=model_name, task=task,
              training_params=training_params,
              optimizer_params=optimizer_params,
              model_save_path=run_name,
              max_input_length = max_input_length,
              max_target_length = max_target_length,
              postprocess_fn=dataset_processor.dataset.postprocess_data)
            trainer, model = model_trainer.train_and_evaluate(train_dataset, eval_dataset, test_dataset, early_stopping_patience = early_stopping_patience)
            
            if push_model_to_hub == True:
               trainer.push_to_hub()
               tokenizer = AutoTokenizer.from_pretrained(model_name)
               tokenizer.push_to_hub(repo_id = hf_model_repo_name, token = hf_token)
            with torch.no_grad():
              del model_trainer
              del trainer
              del model
              gc.collect()
            preds_df = pd.read_csv(os.path.join(training_params['output_dir'], 'predictions.csv'))
            predictions_hf = Dataset.from_pandas(preds_df)
            predictions_hf.push_to_hub(pred_hf_repo_name, private = True, token = hf_token)
            #results_df = metrics_per_instruction(preds_df, inst_number = instruction_number, task = "conditional_generation")
            #results_df.to_csv(str(run_name)+"_"+str(lr)+"_results"+str(i)+".csv", index = False)
    elif task_format == "generation":
        for i in range(1):  #normalde burası 3 olacak 3 run için
          for lr in learning_rates_cond_gen[model_keyword]:  #normalde burada [1:2] olmayacak farklı learning rateler için
            optimizer_params["lr"] = lr
            model_trainer = TrainerForGeneration(
              model_name=model_name, task=task,
              training_params=training_params,
              optimizer_params=optimizer_params,
              model_save_path=run_name,
              max_input_length = max_input_length,
              max_target_length = max_target_length,
              postprocess_fn=dataset_processor.dataset.postprocess_data)
            trainer, model = model_trainer.train_and_evaluate(train_dataset, eval_dataset, test_dataset, early_stopping_patience = early_stopping_patience)
            
            if push_model_to_hub == True:
               trainer.push_to_hub()
              
            with torch.no_grad():
              del model_trainer
              del trainer
              del model
              gc.collect()
            preds_df = pd.read_csv(os.path.join(training_params['output_dir'], 'predictions.csv'))
            predictions_hf = Dataset.from_pandas(preds_df)
            predictions_hf.push_to_hub(pred_hf_repo_name, private = True, token = hf_token)
            #results_df = metrics_per_instruction(preds_df, inst_number = instruction_number, task = "conditional_generation")
            #results_df.to_csv(str(run_name)+"_"+str(lr)+"_results"+str(i)+".csv", index = False)   
