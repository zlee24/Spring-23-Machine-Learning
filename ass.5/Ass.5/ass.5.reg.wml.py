import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

def create_dataset(file_path):
    file = open(file_path) # opening the dataset
    # these store the different data items in each line
    y = []
    X = []
    while True:
        tweet = file.readline().rstrip() # removing excess whitespace, endline chars at the end of each line
        split = tweet.split(' ') # splitting into list of words at space
        y.append(split[-1]) # last work of each sentence is the target
        words = split[:-1] #
        if words:
            sentence = ' '.join(words[2:])
        X.append(sentence)
        
        if not tweet:
            break
    return X,y

X_list,y_list = create_dataset("waseemDataSet5.txt")

def convert_to_dataframe(X):
    df = pd.DataFrame(X, columns=["tweet_text"])
    return df

X = convert_to_dataframe(X_list)

tweet_text = X["tweet_text"]

X_train, X_test, y_train, y_test = train_test_split(tweet_text, y_list, test_size=0.33, random_state=42)

vectorizer = CountVectorizer(ngram_range=(1, 100), token_pattern = r"(?u)\b\w+\b", analyzer='char')
print(Counter(y_test))

#----------------------
#Feature selection and vector creation above
#ML experiments below
#----------------------

scaler = StandardScaler(with_mean=False)

X_train_counts = vectorizer.fit_transform(X_train)
X_train_scaled = scaler.fit_transform(X_train_counts)

print('step one')

#clf = MultinomialNB().fit(X_train_scaled, y_train)

print('step two')

clf2 = RandomForestClassifier(verbose=3).fit(X_train_scaled, y_train)

print('step three')

X_test_counts = vectorizer.transform(X_test)
X_test_scaled = scaler.transform(X_test_counts)

print('step four')

#y_pred = clf.predict(X_test_counts)
y_pred_2 = clf2.predict(X_test_scaled)

print('step 5')

from sklearn.metrics import accuracy_score
#print(accuracy_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred_2))