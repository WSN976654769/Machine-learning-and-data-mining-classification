# python 3.7
# date 2020/4/8
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# two outputs predict value and probability matrix


def cross_validation(feature, result, classifier, score='f1_weighted'):
    cv_result = cross_val_score(classifier, feature, result, cv=10, n_jobs=-1, scoring=score, )
    return cv_result


class KNN:
    def __init__(self, x_train, y_train, neighbours=5, metrics='minkowski', cross_vlidate=0):
        self.x_train = x_train
        self.y_train = y_train
        self.clf = KNeighborsClassifier(n_neighbors=neighbours, metric=metrics, n_jobs=-1)
        self.clf.fit(self.x_train, self.y_train)

        if cross_vlidate == 1:
            cv_result = cross_validation(self.x_train, self.y_train, self.clf)
            print('mean value of 10fold cross validation by recall is :', np.mean(cv_result))

    # return the predict result value
    def predict(self, inputs):
        return self.clf.predict(inputs)

    # return the predict probability matrix
    def predict_proba(self, inputs):
        return self.clf.predict_proba(inputs)


# a very bad classifier........
class adaboost:
    def __init__(self, x_train, y_train, n_estimators=700, max_depth=1, cross_vlidate=0):
        self.x_train = x_train
        self.y_train = y_train
        self.clf = AdaBoostClassifier(n_estimators=n_estimators,
                                      base_estimator=DecisionTreeClassifier(max_depth=max_depth))
        self.clf.fit(self.x_train, self.y_train)
        if cross_vlidate == 1:
            cv_result = cross_validation(self.x_train, self.y_train, self.clf)
            print('mean value of 10fold cross validation by recall is :', np.mean(cv_result))

    # return the predict result value
    def predict(self, inputs):
        return self.clf.predict(inputs)

    # return the predict probability matrix
    def predict_proba(self, inputs):
        return self.clf.predict_proba(inputs)


#####  the method need to add more parameters
class random_froest:
    def __init__(self, x_train, y_train, esimators=70, cross_vlidate=0):
        self.x_train = x_train
        self.y_train = y_train
        self.clf = RandomForestClassifier(n_estimators=esimators, n_jobs=-1)
        self.clf.fit(self.x_train, self.y_train)

        if cross_vlidate == 1:
            cv_result = cross_validation(self.x_train, self.y_train, self.clf)
            print('mean value of 10fold cross validation by recall is :', np.mean(cv_result))

    # return the predict result value
    def predict(self, inputs):
        return self.clf.predict(inputs)

    # return the predict probability matrix
    def predict_proba(self, inputs):
        return self.clf.predict_proba(inputs)


class logistic:
    def __init__(self, x_train, y_train, max_iter=140, multi_class='ovr', cross_vlidate=0):
        self.x_train = x_train
        self.y_train = y_train
        self.clf = LogisticRegression(multi_class=multi_class,
                                      max_iter=max_iter,
                                      solver='saga',
                                      class_weight='balanced',
                                      n_jobs=-1)
        self.clf.fit(self.x_train, self.y_train)

        if cross_vlidate == 1:
            cv_result = cross_validation(self.x_train, self.y_train, self.clf)
            # print('mean value of 10fold cross validation by recall is :', np.mean(cv_result))
            self.cv_result = np.mean(cv_result)


    # return the predict result value
    def predict(self, inputs):
        return self.clf.predict(inputs)

    def cvresult(self):
        return self.cv_result

    # return the predict probability matrix
    def predict_proba(self, inputs):
        return self.clf.predict_proba(inputs)


class svm:
    def __init__(self, x_train, y_train, max_iter=1500, cross_vlidate=0):
        self.x_train = x_train
        self.y_train = y_train
        self.clf = SVC(probability=True,
                       class_weight='balanced',
                       max_iter=max_iter,
                       C=1.2,
                       gamma='scale'
                       )
        self.clf.fit(self.x_train, self.y_train)

        if cross_vlidate == 1:
            cv_result = cross_validation(self.x_train, self.y_train, self.clf)
            print('mean value of 10fold cross validation by recall is :', np.mean(cv_result))

    # return the predict result value
    def predict(self, inputs):
        return self.clf.predict(inputs)

    # return the predict probability matrix
    def predict_proba(self, inputs):
        return self.clf.predict_proba(inputs)


class voting:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        clf1 = LogisticRegression(multi_class='ovr',
                                  solver='saga',
                                  max_iter=150,
                                  n_jobs=-1, random_state=1)
        clf2 = SVC(probability=True,
                   class_weight='balanced',
                   max_iter=1500, random_state=1)
        self.clf = VotingClassifier(estimators=[('lr', clf1), ('svc', clf2)], voting='soft', weights=[1, 1])
        self.clf.fit(self.x_train, self.y_train)

    def predict(self, inputs):
        return self.clf.predict(inputs)

    def predict_proba(self, inputs):
        return self.clf.predict_proba(inputs)
