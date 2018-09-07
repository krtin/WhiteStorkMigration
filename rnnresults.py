import pandas as pd
import numpy as np
import math
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

def geterror(longt,longp,latt,latp):
        dists=[]
        for i in range(0,len(longp)):
                d=distance(longt[i],latt[i],longp[i],latp[i])
                dists.append(d)
        return np.mean(dists),np.std(dists)

def remove(x):
	#print(x)
	x=x.split("[")[1]
	x=x.split("]")[0]
	x=float(x)
	return x
path="results/rnn/partial"
filetest="output_dimTest.csv"
filetrain="output_dimTrain.csv"
#filetest="output_s2Test.csv"
#filetrain="output_s2Train.csv"
filetesting="input_dimTest.csv"
filetraining="input_dimTrain.csv"
#filetesting="input_s2Test.csv"
#filetraining="input_s2Train.csv"

test=pd.read_csv(path+"/"+filetest)
train=pd.read_csv(path+"/"+filetrain)
#print(test)
test["longitude"] = test["longitude"].apply(lambda x: remove(x))
test["latitude"] = test["latitude"].apply(lambda x: remove(x))
train["longitude"] = train["longitude"].apply(lambda x: remove(x))
train["latitude"] = train["latitude"].apply(lambda x: remove(x))
#print(train)
lon1_test_p = test["longitude"].as_matrix()
lat1_test_p = test["latitude"].as_matrix()

lat1_train_p = train["latitude"].as_matrix()
lon1_train_p = train["longitude"].as_matrix()

test=pd.read_csv(path+"/"+filetesting)
train=pd.read_csv(path+"/"+filetraining)
#print(train)
lon1_test = test["longitude"].as_matrix()
lat1_test = test["latitude"].as_matrix()

lat1_train = train["latitude"].as_matrix()
lon1_train = train["longitude"].as_matrix()

print(lon1_train_p)

print("Train")
trainresults = geterror(lat1_train,lat1_train_p,lon1_train,lon1_train_p)
print(trainresults)

print("Test")
testresults = geterror(lat1_test,lat1_test_p,lon1_test,lon1_test_p)
print(testresults)

