# python 3.7
# this file is used to display the final results with 2 extraction methods -- df and chi2
# with one classifier voting, as voting is the combination of logistic and svm so it has the best result in validation set
# the final values are decided by the probability matrix and based on the observation of validation,
# we set the threshold to 0.4
# the score
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import SVC

import feature_extraction

topic = [str(i) for i in range(11)]


# method df
def output(extract):
    train_file = 'training.csv'
    test_file = 'test.csv'
    x_train, y_train = extract.fit_trans(train_file)
    clf1 = LogisticRegression(multi_class='ovr',
                              solver='saga',
                              max_iter=140,
                              n_jobs=-1)

    clf2 = SVC(probability=True,
               class_weight='balanced',
               C=1.2,
               max_iter=1500)

    clf = VotingClassifier(estimators=[('lr', clf1), ('svc', clf2)], voting='soft')
    clf.fit(x_train, y_train)

    x_test, y_test = extract.trans(test_file)
    proba_matrix = clf.predict_proba(x_test)
    result_matrix = clf.predict(x_test)
    print(classification_report(y_test, result_matrix, target_names=topic))
    result = extract.result_threshhold(proba_matrix, result_matrix, filename=test_file, threshhold=0.4)
    return result

# 0 to 10 is based on alphabetic order, irrelevant included, which is 6
def score(result):
    result = np.array(result)
    predict_value = result[:, 0]
    true_value = result[:, 1]
    print(classification_report(true_value, predict_value))


result1 = output(extract=feature_extraction.df())
result2 = output(extract=feature_extraction.chi())


score(result1)
score(result2)