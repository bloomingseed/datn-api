import math
import pandas as pd
import numpy as np

##
# load test dataset
##
dataset = pd.read_csv('./test.csv')
words_series = dataset["text"].transform(lambda x : len(x.split()))
boundaries = [100, 2000]
lower_indicies = words_series[ words_series <= boundaries[0]].index.tolist()
higher_indicies = words_series[ words_series >= boundaries[-1]].index.tolist()
dataset = dataset.drop(lower_indicies + higher_indicies)
dataset["words"] = words_series
category_labels = ["Politics", "Business", "World", "Entertainment", "Sports", "Health", "Life", "Education", "Legal", "Tech"]
source_labels = ["dantri.com.vn", "vnexpress.net", "thanhnien.vn", "vietnamnet.vn"]
##
# normalization module
##
import string
import re

def lower_case(text):
  return text.lower()

def remove_punctuations(text):
  return re.sub(r"[%s]" % string.punctuation, "", text)

def remove_redundant_whitespaces(text):
  return re.sub("\s\s+" , " ", text).strip()
  
def remove_numbers(text):
  return re.sub(r"[0-9]", '', text).strip()

def remove_nbsp(text):
  return re.sub(u'\xa0', u' ', text)

def remove_url(text):
  #pattern = r"https?://\S+|www\.\S+"
  pattern = r"https?://.+\S+"
  return re.sub(pattern, ' ', text)

def normalization(text):
  text = lower_case(text)
  text = remove_url(text)
  text = remove_punctuations(text)
  text = remove_numbers(text)
  text = remove_nbsp(text)
  text = remove_redundant_whitespaces(text)
  return text
  
##
# load pre-saved vectorizer, feature selector and models
##
def load_model(name):
    from joblib import load
    return load('./pre-saved/' + name + '.joblib')
    
vectorizer, selector = load_model('vectorization'), load_model('feature_selection')
my_models = {}
for name in ['mnb', 'svc', 'regression', 'mlp']:
    my_models[name] = load_model(name)

