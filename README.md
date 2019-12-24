Hyperpartisan Language Detection in News Articles

Hyperpartisan news is the reporting of events in a heavily one sided and often misleading manner. The facts are not necessarily false or fabricated, but the style and presentation of the text
can oftentimes heavily influence the conclusion a reader might arrive at. This can happen on both sides of the political spectrum, be it the right or left. 

It has been found that hyperpartisan news pieces from the left greatly resemble the style of the hyperpartisan news pieces of the right. They deploy the same style of writing and structure
in order to sway the opinions of readers in a particular direction. These styles are very different from articles that are not heavily biased and are solely written to report on the news without the intent to 
insert a conclusion that might heavily favour a certain side.

Given this information it is entirely possible to differentiate between news articles that hare heavily biased and articles that are not, which leads to the task at hand: 
When provided a news article text, decide whether it follows a hyperpartisan argumentation, i.e., whether it exhibits blind, prejudiced, or unreasoning allegiance to one party, faction, cause, or person.

This is to be achieved through the training of some form of machine learning model with the help of 600000 articles that were either tagged as biased or unbiased based on whether they were sourced from a hyperpartisan outlet or not.
These articles are to be used to extract information on the differences between hyperpartisan articles and non-hyperpartisan articles through the use of numerous techniques. 
This information is then given to the model to use as reference for any future articles it might be handed in order to decypher whether it is a hyperpartisan piece or not.
In order to validate the training, 150000 articles will be run through the model so it can predict wether each article is biased or not. The effectiveness will be measured on the accuracy of the predicitons. 


04/07/2019 - Experimented with CountVectorizer and TfidfVectorizer from the scikit-learn library. Used the entire collection of by-author articles as the corpus to fit, then vectorized each article against it.
This can be seen in the countvectorizer.py and countvector.py files. The tfidf equivalents can be seen in the tfidfvectorizer.py file and the tfidfvector file. The first fits the corpus and the second document transforms the articles in to their vectors.

Looked at punctuation in the text such as the question mark and the exclamation mark and the extent to which it was used in hyperpartisan articles was much greater.
Should this be looked at as an overall or only in articles where it is used?

Also tested a few ideas that did not work out too well. The average word count per article and the average sentence length was measured alongside the median and a wide range of percentiles to see if there was a great difference but it did not seem like there was.
Words from the unbiased and hyperpartisan sides were direactly compared and they were collected if the ratio of usage was higher than 1.5, in terms of total count, on either side.
This didn't prove useful either as it did not provide any new information that was not previously seen in previous analysis.

Researched possibilities regarding how words of similar meaning could be categorized as the same vector.
Also researched techniques to how semantics can be preserved when extracting data from text. 
Both of these lead to POS and word embeddings.

11-07-2019 - Created a week by week plan for the project. This can be found in the "11-07-2019" folder named "Initial Plan".
Also looked at the paper summarising the results of the SemEval 2019 task. 
A lot of techniques that I have explored have been utilised in a way. For hand crafted features, n-grams, punctuation marks, profanities and such were used. Readability scores and the like were also in use
which I do find interesting and might use.
They also looked at emotionality through the use of libraries which does seem to be present in biased articles.
Named entities do seem to work on political and religious figures which did come up in some analyisis that i have done aswell.
Most teams have some form of word embedding for the extraction of semantic features.

Some more techniques I found interesting come from the team that placed second. They used a bias score based on a corpus filled with words that might indicate some level of bias.
They also utilised POS to extract superlatives and comparatives and used their frequency in articles as the values.
They also calculated a subjectivity score using a Sentiment module.

Agreed upon task: Look into the different techniques I found in the SemEval paper and ones that I otherwise might have had in mind

30/07/2019 - I have collected all features that I will be experimenting on with the decision tree.

Count vector

TFIDF vector

Polarity score

6 x Different types of readability scores

Subjectivity Score

Doc2Vec values 

Empath scores for different lexical categories of every article (kind of like LWIC)

Count for comparative and superlative adjectives from POS tags

Count for personal and possessive and pronouns from POS tags

Character count

Difficult word count

Count for words longer than 6

Count for words longer than 10

Count for words longer than 13

Count for unique words 

Word count

Avg sentence length

Count for exclamation marks and question marks

Also run some tests with some of the features. Count vector and tfidf vector values both scored 49% with the decision tree but when used with a naive bayes classifier the count vectorizer scored 56%
and the tfidf score scored 58%. Could this be the incompatibility of the tfidf and the decision tree, or just the suboptimal settings I used for the decision tree. When sampling the tfidf vectors and reducing the number of features from 10k to 1000,
the accuracy of the decision tree hovers anywhere from 52 to 55, depending on the random samples provided. Polarity scores and subjectivity scores both scored around 56% with the decision tree.

04/08/2019 - Individual test results are ambiguous

13/08/2019 - Extracted vector averages from the articles and ran it through the decision tree and resulted in 58%. Then I went and re-done the way I extract articles and put them in a list to see if that would improve scores, but it did not.
After that, pipelines were used to extract and transform features and the scores have improved.