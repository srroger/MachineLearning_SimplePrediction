from sklearn import tree
import graphviz

#[height, weight, shoe size]
X = [ [181,80,44], [171,70,43], [160,60,38], [154,54,34], 
	[166,65,40], [190,90,47], [175,64,39], [177,70,40], [159,57,37],
	[171,75,42],[181,85,43] ]

Y = ['male', 'female', 'female', 'female', 
	'male', 'male', 'male', 'female', 'male',
	'female', 'male']

clf = tree.ExtraTreeClassifier()
clf = clf.fit(X,Y)

prediction = clf.predict([[192,70,43]])
print(prediction)

dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=['height','weight','shoe size'],  
                         class_names=['heightClass','weightClass','shoe size Class'],  
                         filled=True, rounded=True,  
                         special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('tree2.gv', view=True)