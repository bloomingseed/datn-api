import math
import pandas as pd
import numpy as np

##
# load test dataset
##
dataset = pd.read_csv('./test.csv')
category_labels = ["Politics", "Business", "World", "Entertainment", "Sports", "Health", "Life", "Education", "Legal", "Tech"]
source_labels = ["dantri.com.vn", "vnexpress.net", "thanhnien.vn", "vietnamnet.vn"]
model_names = {'mnb': 'Multinomial Naive Bayes', 'svc': 'Linear Support Vector Classification', 'regression': 'Linear Regression', 'mlp': 'Multilayer Perceptron'}
model_accuracies = {'mnb': 0.8628005657708628, 'svc': 0.9346220336319346, 'regression': 0.9137199434229137, 'mlp': 0.9371365707999372}
##
# text normalization module
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

def preprocess(text):
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

##
# Define pipeline
##
class ProcessPipeline:
  def __init__(self, preprocessor, vectorizer, feature_selector):
    self.preprocessor = preprocessor
    self.tokenizer = vectorizer.build_analyzer()
    self.vectorizer = vectorizer
    self.selector = feature_selector
      
  def process(self, text, model = None):
    self.preprocess = self.preprocessor(text)
    self.tokenization = self.tokenizer(text)
    self.vectorization_ = self.vectorizer.transform([self.preprocess])
    self.feature_selection = self.selector.transform(self.vectorization_)
    self.prediction = model.predict(self.feature_selection) if model != None else None
    return self

process_pipeline = ProcessPipeline(preprocess, vectorizer, selector)

##
# trimmer utility
##
def trim_middle_string(text, n = 50):
    return text if len(text) <= 1.5 * n else '%s...hidden %d characters...%s' % (text[:n], len(text) - 2 * n, text[-n:])

def arr_to_s(a):
    return ', '.join([str(x) for x in a])

def trim_middle_array(a, n = 50):
    return arr_to_s(a) if len(a) <= 1.5 * n else arr_to_s(a[:50]) + '...hidden %d items...' % (len(a) - 2 * n) + arr_to_s(a[-50:])

##
# ISO date time string utility
##
def timestamp():
    from datetime import datetime
    return datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')

##
# manual validation
##
class ManualValidator:
  def __init__(self, dataset, pipeline):
    self.dataset = dataset
    self.pipeline = pipeline

  def validate(self, model, n=100):
    for i in range(n):
      row = self.dataset.sample()
      print(row.index.values[0], row.category_id.values[0], model.predict(self.pipeline.process(row.text.values[0]).feature_selection))
