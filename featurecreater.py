import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def calacc(raw):
	if(raw==0):
		return "0 "+"0"+" 0"
	else:
		raw=raw.split(" ")
		accx=[]
		accy=[]
		accz=[]
		for i in range(0,len(raw)/3):
			accx.append(int(raw[i%3]))
			accy.append(int(raw[(i+1)%3]))
			accz.append(int(raw[(i+2)%3]))	
		#return np.mean(accx),np.mean(accy),np.mean(accz)
		return (str(np.mean(accx))+" "+str(np.mean(accy))+" "+str(np.mean(accz)))

def extractdata(bird,cluster):
	filename = "whitestorkgps.csv"
	#filename = "commoneider.csv"
	data = pd.read_csv(filename)
	data = data[data["visible"]==True]
	#print(data.columns)	
	data=data[data["tag-local-identifier"]==bird]		
	data=data[["timestamp","location-long","location-lat","ECMWF Interim Full Daily SFC Temperature (2 m above Ground)","ECMWF Interim Full Daily SFC-FC 10 Metre Wind Gust","ECMWF Interim Full Daily SFC Wind (10 m above Ground U Component)","ECMWF Interim Full Daily SFC Mean Sea Level Pressure","ECMWF Interim Full Daily SFC Wind (10 m above Ground V Component)"]]
	
	data=data.sort_values("timestamp")
	x = data["location-long"].as_matrix()
        y = data["location-lat"].as_matrix()
	data.columns=["timestamp","long","lat","temp","windgust","windU","pressure","windV"]
	data["map"]=data["timestamp"].str[:-7]
	
	filename="whitestorkacc1.csv"
	accdata = pd.read_csv("whitestorkacc1.csv")
	accdata=accdata.append([pd.read_csv("whitestorkacc2.csv"),pd.read_csv("whitestorkacc3.csv"),pd.read_csv("whitestorkacc4.csv")])
	#print(accdata.columns)
	accdata=accdata[accdata["tag-local-identifier"]==bird]
	accdata = accdata[["timestamp","eobs-accelerations-raw"]]
	accdata.columns=["time","acc"]
	accdata = accdata.sort_values("time")
	accdata["map"]=accdata["time"].str[:-7]
	data = pd.merge(data,accdata,how="left",on="map")
	del data["time"]
	del data["map"]
	del accdata
		
	data=data.fillna(0)
	data['acc'] = data['acc'].apply(lambda x: calacc(x))
	dataacc = pd.DataFrame(data.acc.str.split(' ',2).tolist(), columns = ['accx','accy','accz'])
	data=pd.concat([data,dataacc],axis=1)
	del data["acc"]
	data=data.sort_values("timestamp")
	#write type 1 features
	if not os.path.exists("features/lstm/type1/"+str(cluster)):
                        os.makedirs("features/lstm/type1/"+str(cluster))	
	data.to_csv("features/lstm/type1/"+str(cluster)+"/"+str(bird)+".csv",index=False)
	
	#plt.scatter(x, y, s=1)
        #plt.axis('off')
        #plt.savefig("distance"+str(bird)+".png",pad_inches=0)
		
	


folder="clusters"
clusters=[]
for root, dirs, files in os.walk('clusters', topdown=False):
	for name in dirs:
		clusters.append(name)

cluster_birds=[[] for x in xrange(0,len(clusters))]
for cluster in clusters:
	path=folder+"/"+str(cluster)
	for file in os.listdir(path):
		if file.endswith(".png"):
			cluster_birds[int(cluster)].append(file.split(".")[0][-4:])
	
print(cluster_birds[0])
print(len(cluster_birds))	
for index,birds in enumerate(cluster_birds):
	print(index)
	if(index!=0):
		for bird in birds:
			print(bird)
			extractdata(int(bird),index)
			
	
				
		
