import pandas as pd
import numpy as np  
from sklearn import svm 
from sklearn.utils import shuffle
import os.path
from sklearn.linear_model import LinearRegression
import math
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn import tree
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def draw_map(latTest,lonTest,latPre,lonPre,path):
    
    raw_dataTest = {'latitude': latTest,'longitude':lonTest }
    dfTest = pd.DataFrame(raw_dataTest, columns = ['latitude', 'longitude'])
    raw_dataPre = {'latitude': latPre,'longitude':lonPre }
    dfPre = pd.DataFrame(raw_dataPre, columns = ['latitude', 'longitude'])
    
    fig = plt.figure(figsize=(20,10))
    
    map = Basemap(projection='gall', resolution = 'l',area_thresh = 100000.0,lat_0=0, lon_0=0)
    map.drawcoastlines()
    map.drawcountries()
    map.fillcontinents(color = '#888888')
    map.drawmapboundary(fill_color='#f4f4f4')
    x,y = map(dfTest['longitude'].values, dfTest['latitude'].values)
    xP,yP = map(dfPre['longitude'].values, dfTest['latitude'].values)
    map.plot(x, y, 'ro', markersize=6)
    map.plot(xP, yP, 'ro', markersize=6)
    plt.savefig(path,format='png')
    plt.clf()

def zscorenorm(matrix):
    matrix = pd.DataFrame(matrix)
    for column in matrix:
        if column!="zvalue":
            std = (matrix[column]).std()
            mean = (matrix[column]).mean()
            matrix[column] = pd.DataFrame(((matrix[column])-mean)/(std))
    return matrix.as_matrix()

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
	for i in range(0,len(longt)):
		d=distance(longt[i],latt[i],longp[i],latp[i])
		dists.append(d)
	return np.mean(dists),np.std(dists)

def evaluatetype1cont(X_train,y_long_train,y_lat_train,X_test,y_long_test,y_lat_test,modelname,name,plot=False):
        path="figs/cont_"+"_".join(name.split(" "))+".png"
        model = eval(modelname)
	model2 = eval(modelname)
        X_train=np.concatenate(X_train,axis=0)
        y_long_train=np.concatenate(y_long_train,axis=0)
        y_lat_train=np.concatenate(y_lat_train,axis=0)
        y_long_pred=[]
	y_lat_pred=[]
	#train
        model.fit(X_train,y_long_train)
	model2.fit(X_train,y_lat_train)
	y_long_pred.append(model.predict(X_test[0].reshape(1,-1)))
	y_lat_pred.append(model2.predict(X_test[0].reshape(1,-1)))
	for i,X in enumerate(X_test):
		if(i>0):
			llong = y_long_pred[i-1]
			llat = y_lat_pred[i-1]
			X[1]=llong
			X[2]=llat						
			X=X.reshape(1, -1)
			y_long_pred.append(model.predict(X))
        		y_lat_pred.append(model2.predict(X))
    
        if(plot is True):
                #draw_map(y_lat_test,y_long_test,y_lat_pred,y_long_pred,path)
                plt.scatter(y_long_test,y_lat_test,color='r',label='Test',s=4)
                plt.scatter(y_long_pred,y_lat_pred,color='b',label='Predicted',s=2)
                plt.title(name)
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.legend()
                plt.ylim(33, 48) 
                plt.savefig(path,format="png")
                plt.clf()

        error,stdev = geterror(y_long_test,y_long_pred,y_lat_test,y_lat_pred)
        return [error,stdev,len(X_test)]

def evaluatetype1(X_train,y_long_train,y_lat_train,X_test,y_long_test,y_lat_test,modelname,name,plot=False):
	path="figs/"+"_".join(name.split(" "))+".png"
	model = eval(modelname)
	X_train=np.concatenate(X_train,axis=0)
	y_long_train=np.concatenate(y_long_train,axis=0)
	y_lat_train=np.concatenate(y_lat_train,axis=0)
	#train
	model.fit(X_train,y_long_train)
	y_long_pred=model.predict(X_test)
	model.fit(X_train,y_lat_train)
	y_lat_pred=model.predict(X_test)
	
	if(plot is True):
		#draw_map(y_lat_test,y_long_test,y_lat_pred,y_long_pred,path)
		plt.scatter(y_long_test,y_lat_test,color='r',label='Test',s=4)
                plt.scatter(y_long_pred,y_lat_pred,color='b',label='Predicted',s=2)
                plt.title(name)
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.legend()
		plt.ylim(33, 48)
                plt.savefig(path,format="png")
                plt.clf()

	error,stdev = geterror(y_long_test,y_long_pred,y_lat_test,y_lat_pred)
	return [error,stdev,len(X_test)]

def evaluatetype2cont(X_train,y_long_train,y_lat_train,X_test,y_long_test,y_lat_test,modelname,name,plot=False):
        path="figs/cont"+"_".join(name.split(" "))+".png"
        if plot is True:
                X_train=np.concatenate(X_train)
                y_long_train=np.concatenate(y_long_train)
                y_lat_train=np.concatenate(y_lat_train)
                X_test = X_test[1]
                y_long_test = y_long_test[1]
                y_lat_test = y_lat_test[1]
        else:
                X_train=np.concatenate(X_train)
                X_test=np.concatenate(X_test)
                y_long_train=np.concatenate(y_long_train)
                y_lat_train=np.concatenate(y_lat_train)
                y_long_test=np.concatenate(y_long_test)
                y_lat_test=np.concatenate(y_lat_test)

	y_long_pred=[]
        y_lat_pred=[]
        #X_train=zscorenorm(X_train)    
        #X_test=zscorenorm(X_test)        
	model = eval(modelname)
	model2 = eval(modelname)	
	model.fit(X_train,y_long_train)
        model2.fit(X_train,y_lat_train)
        y_long_pred.append(model.predict(X_test[0].reshape(1,-1)))
        y_lat_pred.append(model2.predict(X_test[0].reshape(1,-1)))
        for i,X in enumerate(X_test):
                if(i>0):
                        llong = y_long_pred[i-1]
                        llat = y_lat_pred[i-1]
                        X[1]=llong
                        X[2]=llat
                        X=X.reshape(1, -1)
                        y_long_pred.append(model.predict(X))
                        y_lat_pred.append(model2.predict(X))


        if(plot is True):
                #draw_map(y_lat_test,y_long_test,y_lat_pred,y_long_pred,path)   
                plt.scatter(y_long_test,y_lat_test,color='r',label='Test',s=4)
                plt.scatter(y_long_pred,y_lat_pred,color='b',label='Predicted',s=2)
                plt.title(name)
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.legend()
                plt.ylim(40, 42)
                plt.savefig(path,format="png")
                plt.clf()
        error,stdev = geterror(y_long_test,y_long_pred,y_lat_test,y_lat_pred)
        return [error,stdev,len(X_test)]


def evaluatetype2(X_train,y_long_train,y_lat_train,X_test,y_long_test,y_lat_test,modelname,name,plot=False):
	path="figs/"+"_".join(name.split(" "))+".png"
	if plot is True:
		X_train=np.concatenate(X_train)
		y_long_train=np.concatenate(y_long_train)
		y_lat_train=np.concatenate(y_lat_train)
		X_test = X_test[1]
		y_long_test = y_long_test[1]
		y_lat_test = y_lat_test[1]	
	else:
		X_train=np.concatenate(X_train)
        	X_test=np.concatenate(X_test)
        	y_long_train=np.concatenate(y_long_train)
        	y_lat_train=np.concatenate(y_lat_train)
        	y_long_test=np.concatenate(y_long_test)
        	y_lat_test=np.concatenate(y_lat_test)
	
	#X_train=zscorenorm(X_train)	
	#X_test=zscorenorm(X_test)        

	model = eval(modelname)
	#train
        model.fit(X_train,y_long_train)
        y_long_pred=model.predict(X_test)
        model.fit(X_train,y_lat_train)
        y_lat_pred=model.predict(X_test)
	
	if(plot is True):
		#draw_map(y_lat_test,y_long_test,y_lat_pred,y_long_pred,path)	
		plt.scatter(y_long_test,y_lat_test,color='r',label='Test',s=4)
        	plt.scatter(y_long_pred,y_lat_pred,color='b',label='Predicted',s=2)
        	plt.title(name)
        	plt.xlabel('Longitude')
        	plt.ylabel('Latitude')
        	plt.legend()
		plt.ylim(40, 42)
        	plt.savefig(path,format="png")
        	plt.clf()
        error,stdev = geterror(y_long_test,y_long_pred,y_lat_test,y_lat_pred)
        return [error,stdev,len(X_test)]

def traintype2(X_train,y_long_train,y_lat_train,modelname):
	X_train=np.concatenate(X_train)
	y_long_train=np.concatenate(y_long_train)
	y_lat_train=np.concatenate(y_lat_train)
	
	model = eval(modelname)
	model.fit(X_train,y_long_train)
	y_long_pred=model.predict(X_train)
	model.fit(X_train,y_lat_train)
	y_lat_pred=model.predict(X_train)
	
	error,stdev = geterror(y_long_train,y_long_pred,y_lat_train,y_lat_pred)
	return [error,stdev,len(X_train)]
	
def crossvalidatetype2(birds,y1,y2,modelname,name,plot=False):
        path="figs/"+"_".join(name.split(" "))+".png"
	k=[5,10,15,20,25,30,35,40]
        cverror=[]
        cvstdev=[]
        clf = eval(modelname)  
	length=0	
        for percent in k:
		
		X_train=[]
        	y_long_train=[]
        	y_lat_train=[]
        	X_test=[]
        	y_long_test=[]
        	y_lat_test=[] 
		for i,bird in enumerate(birds):
			tn=len(bird)
			n=np.floor((tn*(100.0-percent))/100.0).astype(int)
			X_train.append(bird[:n,:])
                	X_test.append(bird[n:tn,:])
                	y_long_train.append(y1[i][:n])
                	y_lat_train.append(y2[i][:n])
                	y_long_test.append(y1[i][n:tn])
                	y_lat_test.append(y2[i][n:tn])

   		error,stdev,num = evaluatetype2(X_train,y_long_train,y_lat_train,X_test,y_long_test,y_lat_test,modelname,name,plot=False)
   		length+=len(X_test)
                cverror.append(error)
                cvstdev.append(stdev)
			
	plt.plot(k,cverror,color='r',label='Error')
	plt.plot(k,cvstdev,color='b',label='Std Dev')
	plt.title(name)
	plt.ylabel('Cross Validation')
	plt.xlabel('Percentage of Test Data per Bird Path')
	plt.legend()
	plt.savefig(path,format="png")
	plt.clf()
        return[np.mean(cverror),np.mean(cvstdev),length]  	

def crossvalidatetype1(x_train,y1,y2,modelname):
	k=len(x_train)
	cverror=[]
	cvstdev=[]
	clf = eval(modelname)	
	length=0
	for i in range(0,k):	
		teston = x_train[i]
		testy = y1[i]
		testy2 = y2[i]
		
		learnon = np.concatenate([x for m,x in enumerate(x_train) if m!=i],axis=0)
		learny = np.concatenate([x for m,x in enumerate(y1) if m!=i],axis=0)
		learny2 = np.concatenate([x for m,x in enumerate(y2) if m!=i],axis=0)
		
                clf.fit(learnon, learny)
                yp1 = clf.predict(teston)
                clf.fit(learnon, learny2)
                yp2 = clf.predict(teston)
                error,stdev = geterror(testy,yp1,testy2,yp2)
                length+=len(teston)
		cverror.append(error)
                cvstdev.append(stdev)
	
	return[np.mean(cverror),np.mean(cvstdev),length]	


def crossvalidate(x_train,y1,y2,modelname):
	
	#y1=long, y2=lat
        k=5 
        datasets = np.array_split(x_train,k,axis=0)
        testsets = np.array_split(y1,k,axis=0)
	testsets2 = np.array_split(y2,k,axis=0) 
        cverror=[]
	cvstdev=[]
	for i in range(0,k):
		#cv train set
                learnon = np.concatenate((datasets[i % k], datasets[(i + 1) % k], datasets[(i + 2) % k], datasets[(i + 3) % k]),axis=0) 
                learny = np.concatenate((testsets[i % k], testsets[(i + 1) % k], testsets[(i + 2) % k], testsets[(i + 3) % k]),axis=0)
                learny2 = np.concatenate((testsets2[i % k], testsets2[(i + 1) % k], testsets2[(i + 2) % k], testsets2[(i + 3) % k]),axis=0)
		#cv test set
		teston = datasets[(i + 4) % k]
                testy = testsets[(i + 4) % k]
                testy2 = testsets2[(i + 4) % k]
		
		clf = eval(modelname)
                clf.fit(learnon, learny)
                yp1 = clf.predict(teston)
		clf.fit(learnon, learny2)
		yp2 = clf.predict(teston)
        	error,stdev = geterror(testy,yp1,testy2,yp2)
		cverror.append(error)
		cvstdev.append(stdev)
	
	#training error
	clf = eval(modelname)
        clf.fit(x_train, y1)
        yp=clf.predict(x_train)
	clf.fit(x_train, y2)
	yp2=clf.predict(x_train)
       	terror,tstdev = geterror(y1,yp,y2,yp2) 
	return[terror,tstdev,np.mean(cverror),np.mean(cvstdev)]
