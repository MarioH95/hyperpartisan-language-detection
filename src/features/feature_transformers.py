import nltk
import pickle
import pandas as pd
import numpy as np
import textstat
from textblob import TextBlob
from profanity_check import predict, predict_prob
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

class PosPerc(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def pos_frequency_calc(self, article):
        """Helper code to compute average word length of a name"""
        article = nltk.pos_tag(nltk.word_tokenize(article))
        possessive, personal, superlative, comparative = 0,0,0,0
        for i,j in article:
            if j == 'PRP$':
                possessive +=1
            elif j == 'PRP':
                personal +=1
            elif j == 'JJS':
                superlative +=1
            elif j == 'JJR':
                comparative +=1

        possessive = possessive / len(article) * 100
        personal = personal / len(article) * 100
        superlative = superlative / len(article) * 100
        comparative = comparative / len(article) * 100

        return pd.Series([possessive, personal, superlative, comparative])

    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        tqdm.pandas()
        df1 = pd.DataFrame({})
        df1[['possessive', 'personal', 'superlative', 'comparative']] = df.progress_apply(self.pos_frequency_calc)
        return df1

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self

class CharLevel(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def char_calc(self, article):
        """Helper code to compute average word length of a name"""
        character, question, exclamation =  0, 0, 0
        for words in article.split():
            for char in words:
                if char == "?":
                    question += 1
                elif char == "!":
                    exclamation += 1
                character += 1
        return pd.Series([character, question, exclamation])

    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        tqdm.pandas()
        df1 = pd.DataFrame({})
        df1[['character', 'question', 'exclamation']] = df.progress_apply(self.char_calc)
        return df1

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self

class WordLevel(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def word_calc(self, article):
        """Helper code to compute average word length of a name"""
        word, word6, word10, word13 = 0, 0, 0, 0
        for split_word in article.split():
            word += 1
            if len(split_word) >= 6:
                word6 += 1
            if len(split_word) >= 10:
                word10 += 1
            if len(split_word) >= 13:
                word13 += 1

        return pd.Series([word, word6, word10, word13])

    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        tqdm.pandas()
        df1 = pd.DataFrame({})
        df1[['word', 'wor6', 'word10', 'word13']] = df.progress_apply(self.word_calc)
        return df1

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self

class SentLevel(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def sent_calc(self, article):
        """Helper code to compute average word length of a name"""
        sent_count, sent_len = 0, 0
        for sent_split in nltk.sent_tokenize(article):
            sent_count +=1
            sent_len+=len(sent_split)
            
        return pd.Series([sent_count, sent_len])

    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        tqdm.pandas()
        df1 = pd.DataFrame({})
        df1[['sent_count', 'sent_len']] = df.progress_apply(self.sent_calc)
        return df1

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self

class WordEmbedding(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        array1 = np.load('C:\\Users\\MarioH\\Desktop\\Uni\\Dissertation\\feature-extraction\\word-embedding\\train_article_vectors.npy')
        array2 = np.load('C:\\Users\\MarioH\\Desktop\\Uni\\Dissertation\\feature-extraction\\word-embedding\\test_article_vectors.npy')
        concatenated = np.concatenate((array1,array2), axis = 0)

        return concatenated

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self

class DocLevel(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def doc_calc(self, article):
        """Helper code to compute average word length of a name"""
        flesch_ease = textstat.flesch_reading_ease(article)
        flesch_grade = textstat.flesch_kincaid_grade(article)
        gunning = textstat.gunning_fog(article)
        profanity = predict_prob([article])[0]
        polarity = TextBlob(article).sentiment.polarity
            
        return pd.Series([flesch_ease, flesch_grade, gunning, profanity, polarity])

    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        tqdm.pandas()
        df1 = pd.DataFrame({})
        df1[['flsech_ease', 'flesch_grade', 'gunning', 'profanity', 'polarity']] = df.progress_apply(self.doc_calc)
        return df1


    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self