import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.tree import DecisionTreeClassifier
from collections import Counter



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

# X_train = vectorizer.fit_transform(X_train)
# X_test = vectorizer.transform(X_test)

# dt = DecisionTreeClassifier(class_weight='balanced', max_depth=6, min_samples_leaf=3, max_leaf_nodes=20)

# dt.fit(X_train, y_train)
# y_pred = dt.predict(X_test)
