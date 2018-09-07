import pandas as pd
import numpy as np
import os
from datetime import datetime
import time
path="features/lstm/type1/"
clusters=[0,1,2]

def createfeatures(data,cluster,file):
	data["timestamp"] = data["timestamp"].apply(lambda x: time.mktime(datetime.strptime(x.split(".")[0], "%Y-%m-%d %H:%M:%S").timetuple()))
	cols=data.columns
	cols=cols.tolist()
	cols.append("Y1")
	cols.append("Y2")
	#print(cols)
	longi=data["long"].as_matrix()
	lat=data["lat"].as_matrix()
	data=data.as_matrix()
	#print(data.shape[0])
	newdata=[]
	for i in range(1,data.shape[0]):
		temp=data[i]-data[i-1]
		args=np.argwhere(data[i-1]==0)
		np.put(temp,args,0)
		args=np.argwhere(data[i]==0)
		np.put(temp,args,0)
		temp[1]=longi[i]-temp[1]
		temp[2]=lat[i]-temp[2]
		newdata.append(temp)
	longi=np.array([np.delete(longi,0)])
	lat=np.array([np.delete(lat,0)])
	newdata=np.array(newdata)
	#print(longi.shape)
	#print(lat.shape)
	#print(newdata.shape)
	data=np.concatenate((newdata,longi.transpose(),lat.transpose()),axis=1)
	data=pd.DataFrame(data)
	#cols.append("Y1")
	#cols.append("Y2")
	data.columns = cols
	if not os.path.exists("features/regression/"+str(cluster)):	
		os.makedirs("features/regression/"+str(cluster))
	data.to_csv("features/regression/"+str(cluster)+"/"+file,index=False)	

for cluster in clusters:
	path2=path+"/"+str(cluster)
	if(cluster==0):
		continue
	for file in os.listdir(path2):
		if file.endswith(".csv"):
			print(file)
			createfeatures(pd.read_csv(path2+"/"+file),cluster,file)
		print(file)
	print(cluster)
