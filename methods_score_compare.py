# python 3.7
# this file are used to compare the scores of different methods on each topic
# it contain 2 parts, firstly it compares all extraction methods based on logistic regression
# then it  compares classifiers based on df extraction
# the evaluation metric is f1 score

import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score

import classifier_model
import feature_extraction

topic = [str(i) for i in range(11)]


def extracting_compare():
    extract1 = feature_extraction.df()
    extract2 = feature_extraction.tf_idf()
    extract3 = feature_extraction.var_tf_idf()
    extract4 = feature_extraction.chi()
    extraction_score = []
    extract_list = [extract1, extract2, extract3, extract4]
    for i in extract_list:
        x_train, y_train = i.fit_trans('new_train.csv')
        x_vali, y_vali = i.trans('val_train.csv')
        clf = classifier_model.logistic(x_train=x_train,
                                        y_train=y_train)
        extraction_score.append(f1_score(y_vali, clf.predict(x_vali), average=None))

    pca = TruncatedSVD(n_components=2000)
    extract5 = feature_extraction.df(max_df=0.8, min_df=0, max_features=10000)
    x_train, y_train = extract5.fit_trans('new_train.csv')
    x_vali, y_vali = extract5.trans('val_train.csv')
    x_train = pca.fit_transform(x_train)
    x_vali = pca.transform(x_vali)
    clf = classifier_model.logistic(x_train=x_train,
                                    y_train=y_train)
    extraction_score.append(f1_score(y_vali, clf.predict(x_vali), average=None))

    # print('cross_validation:', extraction_score)
    plt.figure()
    plt.plot(topic, extraction_score[0], label='df')
    plt.plot(topic, extraction_score[1], label='tf idf')
    plt.plot(topic, extraction_score[2], label='var tf idf')
    plt.plot(topic, extraction_score[3], label='chi2')
    plt.plot(topic, extraction_score[4], label='pca df')
    plt.title('f1 scores of different extraction methods')
    plt.legend()
    plt.show()


def classifier_compare():
    classifier_score = []
    extract = feature_extraction.df()
    x_train, y_train = extract.fit_trans('new_train.csv')
    x_vali, y_vali = extract.trans('val_train.csv')
    clf1 = classifier_model.KNN(x_train=x_train,
                                y_train=y_train)
    classifier_score.append(f1_score(y_vali, clf1.predict(x_vali), average=None))

    clf2 = classifier_model.adaboost(x_train=x_train,
                                     y_train=y_train)
    classifier_score.append(f1_score(y_vali, clf2.predict(x_vali), average=None))

    clf3 = classifier_model.random_froest(x_train=x_train,
                                          y_train=y_train)
    classifier_score.append(f1_score(y_vali, clf3.predict(x_vali), average=None))

    clf4 = classifier_model.logistic(x_train=x_train,
                                     y_train=y_train)
    classifier_score.append(f1_score(y_vali, clf4.predict(x_vali), average=None))

    clf5 = classifier_model.svm(x_train=x_train,
                                y_train=y_train)
    classifier_score.append(f1_score(y_vali, clf5.predict(x_vali), average=None))

    clf6 = classifier_model.voting(x_train=x_train,
                                   y_train=y_train)
    classifier_score.append(f1_score(y_vali, clf6.predict(x_vali), average=None))

    plt.figure()
    plt.plot(topic, classifier_score[0], label='KNN')
    plt.plot(topic, classifier_score[1], label='adaboost')
    plt.plot(topic, classifier_score[2], label='forest')
    plt.plot(topic, classifier_score[3], label='logistic')
    plt.plot(topic, classifier_score[4], label='svm')
    plt.plot(topic, classifier_score[5], label='voting')
    plt.title('f1 scores of different classifiers')
    plt.legend()
    plt.show()


# df,tf-idf,var tf-idf,chi2, pca df
extracting_compare()
# KNN, adaboost, random forest, logistic regression, SVM, voting with logistic and svm
classifier_compare()
