from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
import pickle
from sklearn.tree import tree
from sklearn.neighbors import KNeighborsClassifier


# Read in muffin and cupcake ingredient data# Read i 
#recipes = pd.read_csv('test_liar_feature.csv')
recipes = pd.read_csv('liar_train_feature.csv')
#recipes

print("csv load done")













#recipes[['Article_len','Avg_Word_Len','CountOfNumbers','CountofExclamation','adjectives','WordCount','sent_neg','sent_neu','sent_pos']].as_matrix()

X = recipes[['Article_len','Avg_Word_Len','CountOfNumbers','CountofExclamation','adjectives','WordCount','sent_neg','sent_neu','sent_pos']].as_matrix()
y = np.array(recipes['label'])


X = preprocessing.scale(X)
#X_test = scaling.transform(X_test)

print("feature load done")


#x_train, x_test, y_train, y_test = tts(X, y, test_size=0.6)

#print("train-test done")

###classifiers

#clf_nb = MultinomialNB()
print("model start")

clf_svm = svm.LinearSVC(verbose=True)
#clf_lr = LogisticRegression(verbose = True)
#clf_tree = tree.DecisionTreeClassifier()
#clf_knn = KNeighborsClassifier(n_neighbors=5)
#clf_nb = MultinomialNB()
#########

##model save
print("training start.........")
print(".")
print("svm start")
clf_svm.fit(X, y)
filename = 'svm.sav'
pickle.dump(clf_svm, open(filename, 'wb'))
print("svm done")