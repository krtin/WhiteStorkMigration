# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 21:53:09 2016

@author: monica

Hyper parameters : Epoch, No of nodes in hidden layer
No of hidden layers
batch size drop out val
How many steps to look back
how many steps ahead to produce
"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD
import numpy as np
import csv
import math
from sklearn.metrics import mean_squared_error
#from pandas import DataFrame
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
#------------------- Global parameters ---------------------------------------
trainX = list()
trainY = list()
testX = list()
testY = list()

filename = './0/2541.csv'
testfile = './0/2561.csv'
earth_R = 6371 * 1000.0
#lat = list()
#lon = list()
#---------------------- Hyper parameters---------------------------------------
layers = [3,20,40,2]
no_epoch = 1
batch_size = 32
drop_out = 0.5
train_lastIndex = 0
test_lastIndex = 0
#----------------------------------------------------------------------------

def convt2cartesian(lat,lon):
    x = float(earth_R * math.cos(lat) * math.cos(lon))

    y = float(earth_R * math.cos(lat) * math.sin(lon))

    #z = R *sin(lat)
    return (x,y)
    
def read_csv(filename):
    trainlistX = list()
    trainlistY = list()
    #Read data and make seq matrix
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        reader = list(reader)
        for i in range(len(reader)):
            if i > 0:
                #Generate training seq here
                #x , y = convt2cartesian(float(reader[i][1]),float(reader[i][2]))
                #temp = float(reader[i][3])
                tmp = [float(reader[i][1]),float(reader[i][2]),float(reader[i][3])]
                #tmp = [x,y,temp]
                trainlistX.append(tmp)
                if i < (len(reader) - 1):
                    #xT1,yT1 = convt2cartesian(float(reader[i+1][1]),float(reader[i+1][2]))
                    #trainY.append([xT1,yT1])
                    trainlistY.append([float(reader[i+1][1]),float(reader[i+1][2])])
                else:
                    trainlistY.append([float(reader[i][1]),float(reader[i][2])])
                    #trainY.append([x,y])
    return (trainlistX,trainlistY)
        
def draw_map(latTest,lonTest,latPre,lonPre):
     
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
    plt.show()
    
def extract_data(dataset):
    lon= list()
    lat = list()
    dataset = list(dataset)
    
    for row in dataset:
        lon.append(row[0])
        lat.append(row[1])
    return (lon,lat)


def type2_split():
    global trainX, testX, trainY, testY
    
    test_lastIndex = trainX.shape[0]
    train_lastIndex = int(test_lastIndex * 0.8)
    
    print test_lastIndex, train_lastIndex
    
    testX = trainX[train_lastIndex:test_lastIndex,:]
    testY = trainY[train_lastIndex:test_lastIndex,:]
    trainX = trainX[0:train_lastIndex,:]
    trainY = trainY[0:train_lastIndex,:]
    print trainX.shape, testX.shape
    
def type1_split(testfile):
    global testX, testY
    
    testX,testY = read_csv(testfile)
    testX= np.array(testX)
    testY= np.array(testY)
    
trainX,trainY = read_csv(filename)       
trainX = np.array(trainX)
trainY = np.array(trainY)
type1_split()
#type2_split(testfile)

trainX = np.reshape(trainX, (trainX.shape[0],1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0],1, testX.shape[1]))

model = Sequential()
model.add(LSTM(layers[1], input_dim=layers[0],return_sequences=True,activation='softmax'))
model.add(LSTM(layers[2], input_dim=layers[1],return_sequences=False,activation='softmax'))
model.add(Dropout(drop_out))
model.add(Dense(output_dim=layers[3],activation='linear'))
sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="mse", optimizer=sgd)
model.fit(trainX, trainY, nb_epoch=no_epoch, batch_size=batch_size, verbose=2)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

lonTest , latTest = extract_data(testY)
lonPre , latPre = extract_data(testPredict)

trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
testScore = math.sqrt(mean_squared_error(testY, testPredict))

print('Train Score: %.2f RMSE' % (trainScore))
print('Test Score: %.2f RMSE' % (testScore))

draw_map(latTest,lonTest,latPre,lonPre)