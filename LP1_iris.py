from sklearn import tree
import graphviz
from sklearn.datasets import load_iris


iris = load_iris()

X = iris.data
Y = iris.target

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)

prediction = clf.predict(iris.data[:1, :])
print("The iris printed is ")
print(iris.data[:1, :])
print(iris.feature_names)
print(iris.target_names) 
print(prediction)

dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('treeIris.gv', view=True)