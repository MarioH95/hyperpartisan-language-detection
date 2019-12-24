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

https://www.aclweb.org/anthology/S19-2145/