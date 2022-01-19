# python 3.7
# this code are used to analyse raw texts and also to make validation set
# it contain the distribution of articles on training data
# as well as words distribution on each topic
# at last, we added the article distribution of test data during the discussion process
# (it is not used before training finished)
# last 5 lines with '#' are used to form new training set and validation set

from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df = pd.read_csv('training.csv')
data = df.values[:]
corpus = data[:, 1]
topic = data[:, 2]
countV = CountVectorizer(max_df=0.6, min_df=0, max_features=10000)
y = le.fit_transform(topic)  # label-set
x = corpus

x_train = x
y_train = y

train_topic_dist = Counter(y_train)

plt.figure(figsize=(30, 10))
plt.bar(train_topic_dist.keys(), train_topic_dist.values())
plt.title('training data dist')
plt.xticks(sorted(train_topic_dist.keys()), list(le.classes_))
plt.show()

df = pd.read_csv('test.csv')
data = df.values[:]
corpus = data[:, 1]
topic = data[:, 2]
y = le.transform(topic)
x = corpus
test_topic_dist = Counter(y)

plt.figure(figsize=(30, 10))
plt.bar(test_topic_dist.keys(), test_topic_dist.values())
plt.title('test data dist')
plt.xticks(sorted(test_topic_dist.keys()), list(le.classes_))
plt.show()

from collections import defaultdict

word_number = defaultdict()
for ele in set(data[:, 2]):
    word_number[ele] = []

for ele in data:
    word_number[ele[2]].append(len(ele[1]))

plt.figure(figsize=(30, 10))
plt.boxplot(word_number.values(), labels=word_number.keys())
plt.show()

# x_train, x_vali, y_train, y_vali = train_test_split(x, topic, test_size=0.05263, stratify=topic)
# new_train = pd.DataFrame({'article_words': x_train, 'topic': y_train})
# new_train.to_csv('new_train.csv', index=True, sep=',')
# vali_train = pd.DataFrame({'article_words': x_vali, 'topic': y_vali})
# vali_train.to_csv('vali_train.csv', index=True, sep=',')
