import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

path = "features/lstm/type1/"
pathsm = "features/lstm/smoothed/"
clusters=[0,1,2]
for cluster in clusters:
	path2 = path+"/"+str(cluster)
	path3 = pathsm+"/"+str(cluster)	
	for file in os.listdir(path2):
		if file.endswith(".csv"):
			temp=pd.read_csv(path2+"/"+file)
		temp["timestamp"] = temp["timestamp"].apply(lambda x: time.mktime(datetime.strptime(x.split(".")[0], "%Y-%m-%d %H:%M:%S").timetuple()))
		#print(temp)
		temp=pd.ewma(temp,span=5)
		#print(temp)
		#break
		if not os.path.exists(path3):
			os.makedirs(path3)
		temp.to_csv(path3+"/"+file)
	#break

