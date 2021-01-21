import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
pd.options.mode.chained_assignment = None


# ======== Evaluating models ================================================================================
def F_measure(TP, FP, TN, FN):
    # Calculating Recall, Precision and F1-Score
    Recall, Precision, F_Score = 0, 0, 0
    if (TP + FN) != 0:
        Recall = TP / (TP + FN)
    if (TP + FP) != 0:
        Precision = TP / (TP + FP)
    if (Recall + Precision) != 0:
        F_Score = (2 * Precision * Recall) / (Precision + Recall)
    print('Recall: ', Recall, 'Precision: ', Precision, 'F1_Score: ', F_Score)


def acc(predict, Output):
    p_right, p_false, e_right, e_false = 0, 0, 0, 0
    # Updating TP, FP, TN, and FN
    for i in range(len(predict)):
        if predict[i] == 0 and Output[i] == 0:
            e_right += 1
        elif predict[i] == 0 and Output[i] == 1:
            p_false += 1
        elif predict[i] == 1 and Output[i] == 1:
            p_right += 1
        elif predict[i] == 1 and Output[i] == 0:
            e_false += 1

    print('Accuracy = ', (e_right + p_right) / len(Output))
    F_measure(p_right, e_false, e_right, p_false)


# ======== Pre-processing the dataset ============================================================================
def best_feature(data, n):
    # Base on thee correlation between class and other features, we save the top 10 most correlated ones in f and
    # drop other features in out dataset
    i = 'Class'
    x = {}
    for j in data.columns:
        if i != j:
            x[j] = data[i].corr(data[j])
    x = dict(sorted(x.items(), key=lambda item: item[1], reverse=True))
    f, j = {}, 0
    for i in x:
        f[i] = x[i]
        j += 1
        if j == n:
            break

    for i in data.columns:
        if i not in f and i != 'Class':
            data.drop(columns=[i], inplace=True)

    return data


def IQR(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
    return data


def Z_Score(data):
    z_scores = stats.zscore(data)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    data = data[filtered_entries]
    return data


def Discretization(data, cat, col, i):
    x = list(np.arange(i))
    data[col] = pd.cut(data[col], i, labels=x)
    cat.append(col)
    return cat, data


def data_processing(data, cat):
    # Drop irrelevant features that do not help us in classification
    data.drop(columns=['idnum'], inplace=True)

    # Replace irrelevant data with NaN
    for i in data.columns:
        for j in data[i].unique():
            if type(j) == str:
                data.replace(j, np.nan, inplace=True)

    # Replacing NaN
    for i in data.columns:
        # if i is categorical, replace with most frequent value of i
        if i in cat:
            data[i].fillna(value=int(data[i].mode().values), inplace=True)
        # else replace with mean of i
        else:
            data[i].fillna(value=data[i].mean(), inplace=True)

    # Deleting duplicate data
    data.drop_duplicates(inplace=True)

    # Check if correlation of 2 features are more than 0.8, we can drop one of them
    x = {}
    for i in data.columns:
        for j in data.columns:
            if i != j:
                if data[i].corr(data[j]) > 0.8:
                    x[i] = j
    for i in x.keys():
        if i in data.columns:
            data.drop(columns=[x[i]], inplace=True)

    # Delete outlier data base on IQR or Z-Score
    x = list(np.setdiff1d(data.columns, cat))
    boxplot = data.boxplot(column=x, rot=45, fontsize=10)
    plt.show()
    data = Z_Score(data)
    data.reset_index(inplace=True)
    '''
    # We can convert the feature with continues values to discrete values with calling Discretization function
    cat, data = Discretization(data, cat, 'Age', 4)
    '''
    # Scaling the features with continues values to [0, 1]
    x = list(np.setdiff1d(data.columns, cat))
    for i in x:
        data[i] -= data[i].min()
        data[i] /= data[i].max()

    return data


# ====== Decision Tree =====================================================================================
def decision_tree(train, test):
    feature = train.columns[:-1]
    x = train[feature]
    y = train['Class']
    out = np.array(test['Class'].values)
    test = test[feature]
    feature = list(x.columns)
    y_name = ['0', '1']
    clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)
    clf.fit(x, y)
    plt.figure(figsize=(20, 12))
    tree.plot_tree(clf, feature_names=feature, class_names=y_name, fontsize=5, filled=True)
    plt.show()
    y_pred = clf.predict(test)
    print('Decision Tree')
    acc(y_pred, out)


# ====== K-Nearest =========================================================================================
def knn(train, test):
    feature = train.columns[:-1]
    x = train[feature]
    y = train['Class']
    out = np.array(test['Class'].values)
    test = test[feature]
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(x, y)
    y_pred = model.predict(test)
    print('K-Nearest Neighbors')
    acc(y_pred, out)


# ====== SVM ===============================================================================================
def svm(train, test):
    feature = train.columns[:-1]
    x = train[feature]
    y = train['Class']
    out = np.array(test['Class'].values)
    test = test[feature]
    clf = SVC(kernel='linear', C=1, gamma=0.1)
    clf.fit(x, y)
    # print("Suport vectors are:", clf.support_vectors_)
    y_pred = clf.predict(test)
    print('SVM')
    acc(y_pred, out)


# ==== Reading data =======================================================================================
df = pd.read_excel('dataset.xls')
df = pd.DataFrame(df)
data = df.copy(deep=True)
categorical = ['Sex', 'CP', 'HTN', 'FBS', 'Famhist', 'RestECG', 'prop', 'Exang', 'Slope', 'CA', 'Thal', 'LVF', 'Class']
# Call function to process the data
data = data_processing(data, categorical)
# Call function to find top 10 best features to use
data = best_feature(data, 10)
# Plotting the number of each class occurrence
number = np.array(data['Class'].value_counts())
df = pd.DataFrame({'Class': number}, index=[0, 1])
plot = df.plot.pie(y='Class', figsize=(5, 5), colors=('c', 'y'))
# Plotting instances features' values and classes
plt.show()
pd.plotting.parallel_coordinates(data, 'Class', color=('springgreen', 'grey'))
plt.show()
# Saving data
# data.to_csv('New_Data.csv')
# Split the data into train and test sets
train = data.copy(deep=True)
test = data.sample(frac=0.20)
train = train.drop(test.index, axis=0)
# Models
svm(train, test)
decision_tree(train, test)
knn(train, test)