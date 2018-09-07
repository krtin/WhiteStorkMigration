import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import AgglomerativeClustering,KMeans
import os
import cv2
from sklearn.externals import joblib
filename = "MPIO\ white\ stork\ lifetime\ tracking\ data\ \(2013-2014\)-gps.csv"
filename = "whitestorkgps.csv"
#filename = "commoneider.csv"
filename = "LesserKestrels.csv"
data = pd.read_csv(filename)

print(data.columns)

data = data[data["visible"]==True]
birds = data["tag-local-identifier"].unique()

birddata=[]
siftdata=[]
birdwisedata=[]
for bird in birds:
	temp = data[data["tag-local-identifier"]==bird]
	cluster_data = temp[["location-long","location-lat"]]
	x = cluster_data["location-long"].as_matrix()
	y = cluster_data["location-lat"].as_matrix()
	plt.scatter(x, y, s=1)
	plt.axis('off')
	plt.savefig("distance"+str(bird)+".png",pad_inches=0)
	plt.clf()
	image=Image.open("distance"+str(bird)+".png").convert("L")
	x = np.asarray(image)
	#print(x)
	sift = cv2.SIFT()
	kp = sift.detect(x,None)
	kp,des = sift.compute(x,kp)
	#siftdata = np.concatenate((clusterdata,np.copy(des)),axis=0)
	#x = x.flatten()
	#x = np.floor((x/255))
	#x = ((x-1)*-1).astype("int")
	#print(np.sum(x))
	birdwisedata.append(des)
	siftdata.extend(des.tolist())
	

features=500

kmeans_model = KMeans(n_clusters=features, random_state=1).fit(siftdata)
joblib.dump(kmeans_model, 'kmeansmodel_500.pkl') 

#kmeans_model = joblib.load('kmeansmodel.pkl')
 
for des in birdwisedata:
	results=kmeans_model.predict(des)
	tmp = np.zeros(features)
	for k in range(0,len(results)):
		tmp[results[k]] += 1 
	birddata.append(tmp)

print(len(birddata))
print("will begin main clustering")


n_clusters=9
kmeans = AgglomerativeClustering(n_clusters=n_clusters,affinity="cosine",linkage="complete").fit(birddata)
labels = kmeans.labels_
clustered = [[] for x in xrange(0,n_clusters)]
for index,clusterno in enumerate(labels):
	#print(birds[index])
	clustered[clusterno].append(birds[index])

for i in range(0,n_clusters):
	
	for bird in clustered[i]:
		if not os.path.exists("clusters/"+str(i)):
    			os.makedirs("clusters/"+str(i))
		os.rename("distance"+str(bird)+".png","clusters/"+str(i)+"/distance"+str(bird)+".png")

