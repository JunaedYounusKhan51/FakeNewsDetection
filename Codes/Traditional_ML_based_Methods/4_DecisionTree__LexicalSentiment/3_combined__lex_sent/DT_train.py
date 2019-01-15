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
recipes = pd.read_csv('fulltrain_Guardian_Nyt_binary_shuffled_feature.csv')
#recipes

print("csv load done")













#recipes[['Article_len','Avg_Word_Len','CountOfNumbers','CountofExclamation','adjectives','WordCount','sent_neg','sent_neu','sent_pos']].as_matrix()

X = recipes[['Article_len','Avg_Word_Len','CountOfNumbers','CountofExclamation','adjectives','WordCount','sent_neg','sent_neu','sent_pos']].as_matrix()
y = np.array(recipes['label'])


print("feature load done")


#x_train, x_test, y_train, y_test = tts(X, y, test_size=0.6)

#print("train-test done")

###classifiers

#clf_nb = MultinomialNB()
print("model start")

#clf_svm = svm.LinearSVC(verbose=True)
clf_lr = LogisticRegression(verbose = True)
clf_tree = tree.DecisionTreeClassifier()
clf_knn = KNeighborsClassifier(n_neighbors=5)
clf_nb = MultinomialNB()
#########

##model save
print("training start.........")
print(".")
print("tree start")
clf_tree.fit(X, y)
filename = 'tree.sav'
pickle.dump(clf_tree, open(filename, 'wb'))
print("tree done")




print(".")
print("lr start")
clf_lr.fit(X, y)
filename = 'lr.sav'
pickle.dump(clf_lr, open(filename, 'wb'))
print("lr done")




print(".")
print("knn start")
clf_knn.fit(X, y)
filename = 'knn.sav'
pickle.dump(clf_knn, open(filename, 'wb'))
print("knn done")



print(".")
print("nb start")
clf_nb.fit(X, y)
filename = 'nb.sav'
pickle.dump(clf_nb, open(filename, 'wb'))
print("nb done")






###





