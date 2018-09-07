import numpy as np
import pandas as pd
import os
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
import math
import helper as hp

	

def minmaxnorm(matrix):
    for column in matrix: 
        minimum = (matrix[column]).min()
        maximum = (matrix[column]).max()
        matrix[column] = pd.DataFrame(((matrix[column])-minimum)/(maximum-minimum))
    return matrix
def zscorenorm(matrix):
    for column in matrix:
        if column!="zvalue":
            std = (matrix[column]).std()
            mean = (matrix[column]).mean()
            matrix[column] = pd.DataFrame(((matrix[column])-mean)/(std))
    return matrix

def gettype1data(birds):
	data = birds[len(birds)-1]
	del birds[len(birds)-1]
	y_long_test=[]
	y_lat_test=[]
	X_test=[]
	y_long_test.append(data["Y1"].as_matrix())
	y_lat_test.append(data["Y2"].as_matrix())	
	y_long_test = data["Y1"].as_matrix()
	y_lat_test = data["Y2"].as_matrix()
	del data["Y1"]
	del data["Y2"]
	X_test.append(data.as_matrix())
	X_test = data.as_matrix()
	y_long_train=[]
	y_lat_train=[]
	X_train=[]
	for bird in birds:
		data=bird
		y_long_train.append(data["Y1"].as_matrix())
		y_lat_train.append(data["Y2"].as_matrix()) 
		del data["Y1"]
		del data["Y2"]
		X_train.append(data.as_matrix())
	
	return X_train,y_long_train,y_lat_train,X_test,y_long_test,y_lat_test

def gettype2data(birds):
	X_train=[]
	y_long_train=[]
	y_lat_train=[]
	X_test=[]
	y_long_test=[]
	y_lat_test=[]
	for bird in birds:
		tn=len(bird)
		n=np.floor(tn*0.8).astype(int)
		
		y1=bird["Y1"]
		y2=bird["Y2"]
		del bird["Y1"]
		del bird["Y2"]
		bird=bird.as_matrix()
		y1=y1.as_matrix()
		y2=y2.as_matrix()
		X_train.append(bird[:n,:])
		X_test.append(bird[n:tn,:])
		y_long_train.append(y1[:n])
		y_lat_train.append(y2[:n])
		y_long_test.append(y1[n:tn])
		y_lat_test.append(y2[n:tn])

	return X_train,y_long_train,y_lat_train,X_test,y_long_test,y_lat_test

def distance(lon1,lat1,lon2,lat2):
	R = 6371000
	phi1 = math.radians(lat1)
	phi2 = math.radians(lat2)
	delphi = phi2-phi1
	dellambda = math.radians(lon2)-math.radians(lon1)
	a=math.sin(delphi/2.0)*math.sin(delphi/2.0)+math.cos(phi1)*math.cos(phi2)*math.sin(dellambda/2.0)*math.sin(dellambda/2.0)
	c=2.0*math.atan2(math.sqrt(a),math.sqrt(1-a))
	d = R * c
	return d	

path="features/regression/"
clusters=[0,1,2]
plot=False
data=pd.DataFrame()
birds=[]
birds2=[]
clusterno=2
for cluster in clusters:
        path2=path+"/"+str(cluster)
        if(cluster==clusterno):
	        for file in os.listdir(path2):
        	        if file.endswith(".csv"):
				temp=pd.read_csv(path2+"/"+file)
			birds.append(temp)
			birds2.append(temp.copy())

models = ["LinearRegression()"]
#models = ["SVR()"]
#models = ["MLPRegressor(hidden_layer_sizes=(100,200))"]

#models = ["tree.DecisionTreeRegressor(max_depth=50)"]

#type 1
print("Type 1 Results")
additional="type1"

X_train,y_long_train,y_lat_train,X_test,y_long_test,y_lat_test = gettype1data(birds)

print("Train")
trainresults = hp.traintype2(X_train,y_long_train,y_lat_train,models[0])
print(trainresults)

print("Validation")
valresults = hp.crossvalidatetype1(X_train,y_long_train,y_lat_train,models[0])
print(valresults)

print("Test")
testresults = hp.evaluatetype1(X_train,y_long_train,y_lat_train,X_test,y_long_test,y_lat_test,models[0],"Path Comparison for Type 1 Cluster "+str(clusterno+1),plot)
print(testresults)

print("cont Test")
testcontresults = hp.evaluatetype1cont(X_train,y_long_train,y_lat_train,X_test,y_long_test,y_lat_test,models[0],"Path Comparison for Type 1 Cluster "+str(clusterno+1),plot)
print(testcontresults)

#type 2
print("Type 2 results")

X_train,y_long_train,y_lat_train,X_test,y_long_test,y_lat_test = gettype2data(birds2)

additional="type2"

print("Train")
trainresults = hp.traintype2(X_train,y_long_train,y_lat_train,models[0])
print(trainresults)

print("Validation")
validresults = hp.crossvalidatetype2(X_train,y_long_train,y_lat_train,models[0],"Cross-Validation Graph for Type 2 Cluster "+str(clusterno+1))
print(validresults)

print("Test")
testresults = hp.evaluatetype2(X_train,y_long_train,y_lat_train,X_test,y_long_test,y_lat_test,models[0],"Path Comparison for Type 2 Cluster "+str(clusterno+1),plot)
print(testresults)

print("Cont Test")
conttestresults = hp.evaluatetype2cont(X_train,y_long_train,y_lat_train,X_test,y_long_test,y_lat_test,models[0],"Path Comparison for Type 2 Cluster "+str(clusterno+1),plot)
print(conttestresults)

'''
#print(data)
#data=zscorenorm(data)
#print(data)
scale=1.0
data["Y1"]=data["Y1"]*scale
data["Y2"]=data["Y2"]*scale
y_long=data["Y1"].as_matrix()
y_lat=data["Y2"].as_matrix()
print(y_long)
y_c = data[["Y1","Y2"]].as_matrix()
del data["Y1"]
del data["Y2"]
#del data["accx"]
#del data["accy"]
#del data["accz"]

data=zscorenorm(data)
X = data.as_matrix()
X,y_long,y_lat = shuffle(X,y_long,y_lat,random_state=17)

totallen=len(X)
trainlen=np.floor(totallen*0.8).astype(int)
testlen=totallen-trainlen
X_train=X[:trainlen,:]
X_test=X[trainlen:totallen,:]
y_long_train=y_long[:trainlen]
y_long_test=y_long[trainlen:totallen]
y_lat_train=y_lat[:trainlen]
y_lat_test=y_lat[trainlen:totallen]
y_c_train=y_c[:trainlen]
y_c_test=y_c[trainlen:totallen]
print(X_train.shape)


#######################################
#cross validate
#type 1

models = ["LinearRegression()"]
valresults = hp.crossvalidate(X_train,y_long_train,y_lat_train,models[0])
print(valresults)

testresults = hp.evaluate(X_train,y_long_train,y_lat_train,X_test,y_long_test,y_lat_test,models[0])
print(testresults)

'''

'''
model=LinearRegression()
model.fit(X_train,y_long_train)
print(model.score(X_test,y_long_test))
y_long_pred=model.predict(X_test)

model.fit(X_train,y_lat_train)
print(model.score(X_test,y_lat_test))
y_lat_pred=model.predict(X_test)


print(y_long_pred)
print(y_long_test)

print(y_lat_pred)
print(y_lat_test)

dists=[]
for i in range(0,len(y_lat_pred)):
	lon1=y_long_test[i]
	lat1=y_lat_test[i]
	lon2=y_long_pred[i]
	lat2=y_lat_pred[i]
	d=distance(lon1/scale,lat1/scale,lon2/scale,lat2/scale)
	#print(lon1,lat1,lon2,lat2)
	dists.append(d)

print(np.mean(dists))
print(np.std(dists))

'''


