import feature_transformers
import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

train_data = pickle.load(open("C:\\Users\\MarioH\\Desktop\\Uni\\Dissertation\\articles\\split-articles\\training\\trainarticles.pickle", "rb"))
test_data = pickle.load(open("C:\\Users\\MarioH\\Desktop\\Uni\\Dissertation\\articles\\split-articles\\testing\\testarticles.pickle", "rb"))
train_tags = pickle.load(open("C:\\Users\\MarioH\\Desktop\\Uni\\Dissertation\\articles\\split-articles\\training\\traintags.pickle", "rb"))
test_tags = pickle.load(open("C:\\Users\\MarioH\\Desktop\\Uni\\Dissertation\\articles\\split-articles\\testing\\testtags.pickle", "rb"))

x = train_data + test_data
y = train_tags + test_tags
del train_data, test_data, train_tags, test_tags
df = pd.DataFrame({'articles':x, 'tags':y})

x = df['articles']
y = df['tags']

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.8, random_state = 30) 
#X_train, X_test, y_train, y_test = x[: 600000], x[600000:], y[: 600000], y[600000:]

tfidf_pipe = Pipeline([('vectorize', CountVectorizer(max_features = 10000)),
  ('tfidf', TfidfTransformer())])

pos_pipe = Pipeline([('pos_perc', feature_transformers.PosPerc())])

char_pipe = Pipeline([('char_level', feature_transformers.CharLevel())])

word_pipe = Pipeline([('word_level', feature_transformers.WordLevel())])

sent_pipe = Pipeline([('sent_level', feature_transformers.SentLevel())])

doc_pipe = Pipeline([('doc_level', feature_transformers.DocLevel())])

embedding_pipe = Pipeline([('word_embedding', feature_transformers.WordEmbedding())])

full_pipeline = Pipeline([
    ('feat_union', FeatureUnion(transformer_list=[
          ('tfidf_pipeline', tfidf_pipe),
          #('pos_pipeline', pos_pipe),
          ('char_pipeline', char_pipe),
          #('word_pipeline', word_pipe),
          ('sent_pipeline', sent_pipe),
          ('doc_pipeline', doc_pipe),
          #('embedding_pipeline', embedding_pipe)
          ])),
    ('classify', DecisionTreeClassifier(max_depth = 5))
    ])

full_pipeline.fit(X_train, y_train)
y_pred = full_pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred)*100)
print("Precision:", precision_score(y_test, y_pred, pos_label="true")*100)
print("Recall:", recall_score(y_test, y_pred, pos_label="true")*100)
print("F1:", f1_score(y_test, y_pred, pos_label="true")*100)