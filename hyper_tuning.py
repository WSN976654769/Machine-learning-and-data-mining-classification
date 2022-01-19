# python 3.7
# this file is used to tuning hyper parameters of all methods
# some phrases may consume much time, every function can run independently to show expected part
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier

import classifier_model
import feature_extraction


def df_feature_selection():
    results = []
    num_features = [i for i in range(1000, 2500, 100)]
    for i in num_features:
        extract = feature_extraction.df(max_df=0.8, min_df=0, max_features=i)
        train_file = 'new_train.csv'
        x_train, y_train = extract.fit_trans(train_file)
        clf = classifier_model.logistic(x_train=x_train, y_train=y_train, cross_vlidate=1)
        results.append(clf.cv_result)

    plt.figure()
    plt.plot(num_features, results)
    plt.title('DF features')
    plt.show()


def chi_feature_selection():
    results = []
    num_features = [i for i in range(1000, 2500, 100)]
    for i in num_features:
        extract = feature_extraction.chi(num=i)
        train_file = 'new_train.csv'
        x_train, y_train = extract.fit_trans(train_file)
        clf = classifier_model.logistic(x_train=x_train, y_train=y_train, cross_vlidate=1)
        results.append(clf.cv_result)

    plt.figure()
    plt.plot(num_features, results)
    plt.title('chi2 features')
    plt.show()


# use cross-validation in new_train.csv to tuning features
df_feature_selection()
chi_feature_selection()


def tuning_logistic():
    extract = feature_extraction.df()
    train_file = 'new_train.csv'
    x_train, y_train = extract.fit_trans(train_file)
    lr = LogisticRegression(class_weight='balanced', n_jobs=-1)
    lr_parameters = {'max_iter': [i for i in range(100, 300, 10)], 'multi_class': ['ovr', 'multinomial'],
                     'solver': ['saga', 'sag']}
    grid_lr = GridSearchCV(estimator=lr, param_grid=lr_parameters, scoring='f1_weighted', n_jobs=-1)
    grid_lr.fit(x_train, y_train)
    print('The best parameters for logistic regression is :', grid_lr.best_params_)


def tuning_svm():
    extract = feature_extraction.df()
    train_file = 'new_train.csv'
    x_train, y_train = extract.fit_trans(train_file)
    svm = SVC(max_iter=1500, class_weight='balanced')
    svm_parameters = {'C': [0.8, 1, 1.2], 'gamma': ['scale', 'auto']}
    grid_svm = GridSearchCV(estimator=svm, param_grid=svm_parameters, scoring='f1_weighted', n_jobs=-1)
    grid_svm.fit(x_train, y_train)
    print('The best parameters for SVM is :', grid_svm.best_params_)


# for these two tuning process just use df to fast speed,
# as the performance of df and chi2 is very close and usually have similar react with the changing parameters
# during the exploratory phrase
tuning_logistic()
tuning_svm()


def tuning_voting():
    topic = [str(i) for i in range(11)]
    extract = feature_extraction.df()
    train_file = 'new_train.csv'
    test_file = 'val_train.csv'
    x_train, y_train = extract.fit_trans(train_file)
    x_test, y_test = extract.trans(test_file)
    clf1 = LogisticRegression(multi_class='ovr',
                              solver='saga',
                              max_iter=140,
                              n_jobs=-1)

    clf2 = SVC(probability=True,
               class_weight='balanced',
               C=1.2,
               max_iter=1500)
    weights = [[2,1],[1,1],[1,2]]
    for w in weights:
        clf = VotingClassifier(estimators=[('lr', clf1), ('svc', clf2)], voting='soft',weights=w)
        clf.fit(x_train,y_train)
        result_matrix = clf.predict(x_test)
        print('weights used:',w)
        print(classification_report(y_test, result_matrix, target_names=topic))


# the only parameter need to be examined is 'weights' in voting classifier
tuning_voting()
