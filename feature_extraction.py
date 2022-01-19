# python 3.7
# date 2020/4/8
# this file include all extraction methods used
from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder


# output of trans is feature X and result y
# output_result is used to log results


# can add other extract method
# also can mix sample method
class tf_idf:
    def __init__(self, max_df=0.6, min_df=0.01, max_features=10000):
        self.countV = TfidfVectorizer(max_df=max_df, min_df=min_df, max_features=max_features, use_idf=False)
        self.le = LabelEncoder()

    def fit_trans(self, file_name):
        # extract training data
        df = pd.read_csv(file_name)
        data = df.values[:]
        corpus = data[:, 1]
        topic = data[:, 2]

        y = self.le.fit_transform(topic)  # label-set
        # print(le.classes_)  # watch the label and topics---- 6 is for irrelevant

        # feature is tfidf, output y is coded by LabelEncoder
        transformer = self.countV
        tfidf = transformer.fit_transform(corpus)

        return tfidf, y

    # trans uesd to predict test sanmple
    def trans(self, filename='test.csv'):
        test_data = pd.read_csv(filename).values[:]
        test_corpus = test_data[:, 1]
        test_topic = test_data[:, 2]
        test_true = self.le.transform(test_topic)
        test_tfidf = self.countV.transform(test_corpus)

        return test_tfidf, test_true

    ### out_put need to encode y_true based on tf-idf self.le = LabelEncoder()
    ### if only need to use this part suggest to use LabelEncoder() during feature extraction

    def output_result(self, proba_matrix, predict_result, filename='test.csv'):
        label = self.le.classes_
        irrelevant = -1
        for i in range(len(label)):
            if label[i] == 'IRRELEVANT':
                irrelevant = i

        test_data = pd.read_csv(filename).values[:]
        test_topic = test_data[:, 2]
        test_true = self.le.transform(test_topic)
        print(f'recall score for test sample:', np.mean(recall_score(test_true, predict_result, average=None)))

        result = []
        topic_number = len(proba_matrix[0])
        for i in range(topic_number):
            if i != irrelevant:
                array = proba_matrix[:, i]
                sorted_index = np.argsort(-array)[0:10]
                result.append(sorted_index)
            else:
                result.append([0])

        for i in range(topic_number):
            if i != irrelevant:
                print('recommend text for topic: ', i, ' ::', self.le.classes_[i])
                for ele in result[i]:
                    print(
                        f'predict probability is {round(proba_matrix[ele, i], 4)}, '
                        f'true value is {test_true[ele]}, index is {ele}')
                print('----------------------------------')
        # y_test_pre = clf.predict(test_tfidf)
        # print(f'test result for {i} neighbour:', accuracy_score(test_true, y_test_pre))
        print('---------------------------------------------------')


class df:
    def __init__(self, max_df=0.8, min_df=0, max_features=2200):
        self.countV = CountVectorizer(max_df=max_df, min_df=min_df, max_features=max_features)
        self.le = LabelEncoder()

    def fit_trans(self, trainfile='new_train.csv'):
        # extract training data
        df = pd.read_csv(trainfile)
        data = df.values[:]
        corpus = data[:, 1]
        topic = data[:, 2]

        y = self.le.fit_transform(topic)  # label-set
        # print(le.classes_)  # watch the label and topics---- 6 is for irrelevant

        # feature is tfidf, output y is coded by LabelEncoder
        transformer = self.countV
        x = transformer.fit_transform(corpus)
        return x, y

    # trans uesd to predict test sanmple
    def trans(self, filename='test.csv'):
        test_data = pd.read_csv(filename).values[:]
        test_corpus = test_data[:, 1]
        test_topic = test_data[:, 2]
        test_true = self.le.transform(test_topic)
        test_x = self.countV.transform(test_corpus)
        return test_x, test_true

    ### out_put need to encode y_true based on tf-idf self.le = LabelEncoder()
    ### if only need to use this part suggest to use LabelEncoder() during feature extraction

    def output_result(self, proba_matrix, predict_result, filename='test.csv'):
        label = self.le.classes_
        irrelevant = -1
        for i in range(len(label)):
            if label[i] == 'IRRELEVANT':
                irrelevant = i

        test_data = pd.read_csv(filename).values[:]
        test_topic = test_data[:, 2]
        test_true = self.le.transform(test_topic)
        # print(f'recall score for test sample:', np.mean(recall_score(test_true, predict_result, average=None)))

        result = []
        topic_number = len(proba_matrix[0])
        for i in range(topic_number):
            if i != irrelevant:
                array = proba_matrix[:, i]
                sorted_index = np.argsort(-array)[0:10]
                result.append(sorted_index)
            else:
                result.append([0])

        for i in range(topic_number):
            if i != irrelevant:
                print('recommend text for topic: ', i, ' ::', self.le.classes_[i])
                for ele in result[i]:
                    print(
                        f'predict probability is {round(proba_matrix[ele, i], 4)}, '
                        f'true value is {test_true[ele]}, index is {ele}')
                print('----------------------------------')
        # y_test_pre = clf.predict(test_tfidf)
        # print(f'test result for {i} neighbour:', accuracy_score(test_true, y_test_pre))
        print('---------------------------------------------------')

    def result_threshhold(self, proba_matrix, predict_result, filename='test.csv', threshhold=0.4):
        label = self.le.classes_
        irrelevant = -1
        for i in range(len(label)):
            if label[i] == 'IRRELEVANT':
                irrelevant = i

        test_data = pd.read_csv(filename).values[:]
        article_num = test_data[:, 0]
        test_topic = test_data[:, 2]
        test_true = self.le.transform(test_topic)
        # print(f'recall score for test sample:', np.mean(recall_score(test_true, predict_result, average=None)))

        result = []
        topic_number = len(proba_matrix[0])
        for i in range(topic_number):
            if i != irrelevant:
                array = proba_matrix[:, i]
                sorted_index = np.argsort(-array)[0:10]
                result.append(sorted_index)
            else:
                result.append([0])
        out_put = []
        for i in range(topic_number):
            if i != irrelevant:
                print('recommend text for topic: ', i, ' ::', self.le.classes_[i])
                for ele in result[i]:
                    if round(proba_matrix[ele, i], 4) > threshhold:
                        # predict value, true value, probability, class name
                        out_put.append([i, test_true[ele], round(proba_matrix[ele, i], 4), self.le.classes_[i]])
                        print(
                            f'predict probability is {round(proba_matrix[ele, i], 4)}, '
                            f'true value is {test_true[ele]}, index is {article_num[ele]}')
                print('----------------------------------')
        # y_test_pre = clf.predict(test_tfidf)
        # print(f'test result for {i} neighbour:', accuracy_score(test_true, y_test_pre))
        print('---------------------------------------------------')
        return out_put


class var_tf_idf:
    def __init__(self, num=105):
        self.le = LabelEncoder()
        self.count = CountVectorizer()
        self.tfidf = TfidfVectorizer(norm='l2', smooth_idf=True, use_idf=True)
        self.selected_words = []
        self.Word_var = []
        self.x_test = None
        self.y_test = None
        self.num = num

    def get_counts(self, corpus):
        all_words = ','.join(corpus).split(',')
        word_counts = Counter(all_words)
        return word_counts

    def get_reduced_corpus(self, corpus):
        articles_words = map(lambda x: x.split(','), corpus)
        new_articles_words = []
        for article_words in list(articles_words):
            new_article_words = []
            for word in article_words:
                if word in list(set(self.selected_words)):
                    new_article_words.append(word)
            new_articles_words.append(','.join(new_article_words))
        return new_articles_words

    def fit_trans(self, trainfile='training.csv', testfile='test.csv'):
        # 提取训练的数据
        train_data = pd.read_csv(trainfile)
        # train_data = train_data[~train_data['topic'].isin(['IRRELEVANT'])]
        train_corpus = train_data['article_words'].values
        train_topic = train_data['topic'].values

        # 提取测试数据
        test_data = pd.read_csv(testfile)
        # test_data  = test_data [~test_data['topic'].isin(['IRRELEVANT'])]
        test_corpus = test_data['article_words'].values
        test_topic = test_data['topic'].values

        # 提取每个topic最高频的top30单词组成特征词
        divided_df_by_topic = map(lambda x: train_data[train_data['topic'].isin([x])], list(set(train_topic)))
        articles_by_topic = list(map(lambda x: x['article_words'].values, divided_df_by_topic))
        mostcommon_words_of_each_topic = list(
            map(lambda x: self.get_counts(x).most_common(self.num), articles_by_topic))

        for word_list in mostcommon_words_of_each_topic:
            for word_tuple in word_list:
                self.selected_words.append(word_tuple[0])
        reduced_train_corpus = self.get_reduced_corpus(train_corpus)
        reduced_test_corpus = self.get_reduced_corpus(test_corpus)

        for Wi in set(self.selected_words):
            P_Wi_Cj_list = []
            for same_topic_articles in list(articles_by_topic):
                articles_words = list(map(lambda x: x.split(','), same_topic_articles))
                D_Cj = len(articles_words)
                D_Wi_Cj = sum(map(lambda x: Wi in x, articles_words))
                P_Wi_Cj = D_Wi_Cj / D_Cj
                P_Wi_Cj_list.append(P_Wi_Cj)
            P_Wi_Cj_array = np.array(P_Wi_Cj_list)
            self.Word_var.append(np.var(P_Wi_Cj_array))

        self.tfidf.fit_transform(reduced_train_corpus)

        X_train_pre = self.tfidf.transform(reduced_train_corpus).todense()
        X_train = np.zeros(X_train_pre.shape)
        for i in range(X_train.shape[0]):
            X_train[i] = np.multiply(X_train_pre[i], self.Word_var)
        X_train = preprocessing.scale(X_train)
        X_train = csr_matrix(X_train)
        # X_train = tfidf.transform(train_corpus)
        y_train = self.le.fit_transform(train_topic)

        X_test_pre = self.tfidf.transform(reduced_test_corpus).todense()
        X_test = np.zeros(X_test_pre.shape)
        for i in range(X_test.shape[0]):
            X_test[i] = np.multiply(X_test_pre[i], self.Word_var)
        X_test = preprocessing.scale(X_test)
        X_test = csr_matrix(X_test)
        # X_test = tfidf.transform(test_corpus)
        y_test = self.le.transform(test_topic)

        self.x_test = X_test
        self.y_test = y_test
        return X_train, y_train

    def trans(self, filename=''):
        return self.x_test, self.y_test

    def output_result(self, proba_matrix, predict_result, filename='test.csv'):
        label = self.le.classes_
        irrelevant = -1
        for i in range(len(label)):
            if label[i] == 'IRRELEVANT':
                irrelevant = i

        test_data = pd.read_csv(filename).values[:]
        test_topic = test_data[:, 2]
        test_true = self.le.transform(test_topic)
        result = []
        topic_number = len(proba_matrix[0])
        for i in range(topic_number):
            if i != irrelevant:
                array = proba_matrix[:, i]
                sorted_index = np.argsort(-array)[0:10]
                result.append(sorted_index)
            else:
                result.append([0])

        for i in range(topic_number):
            if i != irrelevant:
                print('recommend text for topic: ', i, ' ::', self.le.classes_[i])
                for ele in result[i]:
                    print(
                        f'predict probability is {round(proba_matrix[ele, i], 4)}, '
                        f'true value is {test_true[ele]}, index is {ele}')
                print('----------------------------------')
        # y_test_pre = clf.predict(test_tfidf)
        # print(f'test result for {i} neighbour:', accuracy_score(test_true, y_test_pre))
        print('---------------------------------------------------')


class chi:
    def __init__(self, num=1600):
        # choose one of them
        self.model = SelectKBest(chi2, k=num)
        # self.model = SelectKBest(mutual_info_classif, k=num)
        # self.model = SelectKBest(f_classif, k=num)

        self.le = LabelEncoder()

        self.count = CountVectorizer()
        # self.tfidf = TfidfVectorizer()

    def fit_trans(self, file_name):
        # extract training data
        data = pd.read_csv(file_name).values[:]
        corpus = data[:, 1]
        topic = data[:, 2]
        y = self.le.fit_transform(topic)
        x = self.count.fit_transform(corpus)

        train_x = self.model.fit_transform(x, y)
        return train_x, y

    # trans uesd to predict test sanmple
    def trans(self, filename='test.csv'):

        test_data = pd.read_csv(filename).values[:]
        test_corpus = test_data[:, 1]
        test_topic = test_data[:, 2]
        y = self.le.transform(test_topic)
        x = self.count.transform(test_corpus)

        test_x = self.model.transform(x)
        return test_x, y

    def output_result(self, proba_matrix, predict_result, filename='test.csv'):
        label = self.le.classes_
        irrelevant = -1
        for i in range(len(label)):
            if label[i] == 'IRRELEVANT':
                irrelevant = i

        test_data = pd.read_csv(filename).values[:]
        test_topic = test_data[:, 2]
        test_true = self.le.transform(test_topic)
        # print(f'recall score for test sample:', np.mean(recall_score(test_true, predict_result, average=None)))

        result = []
        topic_number = len(proba_matrix[0])
        for i in range(topic_number):
            if i != irrelevant:
                array = proba_matrix[:, i]
                sorted_index = np.argsort(-array)[0:10]
                result.append(sorted_index)
            else:
                result.append([0])

        for i in range(topic_number):
            if i != irrelevant:
                print('recommend text for topic: ', i, ' ::', self.le.classes_[i])
                for ele in result[i]:
                    print(
                        f'predict probability is {round(proba_matrix[ele, i], 4)}, '
                        f'true value is {test_true[ele]}, index is {ele}')
                print('----------------------------------')
        # y_test_pre = clf.predict(test_tfidf)
        # print(f'test result for {i} neighbour:', accuracy_score(test_true, y_test_pre))
        print('---------------------------------------------------')

    def result_threshhold(self, proba_matrix, predict_result, filename='test.csv', threshhold=0.4):
        label = self.le.classes_
        irrelevant = -1
        for i in range(len(label)):
            if label[i] == 'IRRELEVANT':
                irrelevant = i

        test_data = pd.read_csv(filename).values[:]
        article_num = test_data[:, 0]
        test_topic = test_data[:, 2]
        test_true = self.le.transform(test_topic)
        # print(f'recall score for test sample:', np.mean(recall_score(test_true, predict_result, average=None)))

        result = []
        topic_number = len(proba_matrix[0])
        for i in range(topic_number):
            if i != irrelevant:
                array = proba_matrix[:, i]
                sorted_index = np.argsort(-array)[0:10]
                result.append(sorted_index)
            else:
                result.append([0])
        out_put = []
        for i in range(topic_number):
            if i != irrelevant:
                print('recommend text for topic: ', i, ' ::', self.le.classes_[i])
                for ele in result[i]:
                    if round(proba_matrix[ele, i], 4) > threshhold:
                        # predict value, true value, probability, class name
                        out_put.append([i, test_true[ele], round(proba_matrix[ele, i], 4), self.le.classes_[i]])
                        print(
                            f'predict probability is {round(proba_matrix[ele, i], 4)}, '
                            f'true value is {test_true[ele]}, index is {article_num[ele]}')
                print('----------------------------------')
        # y_test_pre = clf.predict(test_tfidf)
        # print(f'test result for {i} neighbour:', accuracy_score(test_true, y_test_pre))
        print('---------------------------------------------------')
        return out_put


class pca_df:
    def __init__(self, n_components=2000, max_feature=10000):
        self.pca = TruncatedSVD(n_components=n_components)
        self.le = LabelEncoder()
        self.count = CountVectorizer(max_df=0.8, min_df=0, max_features=max_feature)

    def fit_trans(self, file_name):
        # extract training data
        data = pd.read_csv(file_name).values[:]
        corpus = data[:, 1]
        topic = data[:, 2]
        y = self.le.fit_transform(topic)
        x = self.count.fit_transform(corpus)

        train_x = self.pca.fit_transform(x, y)
        return train_x, y

    # trans uesd to predict test sanmple
    def trans(self, filename='test.csv'):

        test_data = pd.read_csv(filename).values[:]
        test_corpus = test_data[:, 1]
        test_topic = test_data[:, 2]
        y = self.le.transform(test_topic)
        x = self.count.transform(test_corpus)

        test_x = self.pca.transform(x)
        return test_x, y

    def output_result(self, proba_matrix, predict_result, filename='test.csv'):
        label = self.le.classes_
        irrelevant = -1
        for i in range(len(label)):
            if label[i] == 'IRRELEVANT':
                irrelevant = i

        test_data = pd.read_csv(filename).values[:]
        test_topic = test_data[:, 2]
        test_true = self.le.transform(test_topic)

        result = []
        topic_number = len(proba_matrix[0])
        for i in range(topic_number):
            if i != irrelevant:
                array = proba_matrix[:, i]
                sorted_index = np.argsort(-array)[0:10]
                result.append(sorted_index)
            else:
                result.append([0])

        for i in range(topic_number):
            if i != irrelevant:
                print('recommend text for topic: ', i, ' ::', self.le.classes_[i])
                for ele in result[i]:
                    print(
                        f'predict probability is {round(proba_matrix[ele, i], 4)}, '
                        f'true value is {test_true[ele]}, index is {ele}')
                print('----------------------------------')
        # y_test_pre = clf.predict(test_tfidf)
        # print(f'test result for {i} neighbour:', accuracy_score(test_true, y_test_pre))
        print('---------------------------------------------------')
