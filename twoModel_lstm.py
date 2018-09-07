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
trainX_lat = list()
testX_lat = list()
trainY_lat = list()
testY_lat = list()

trainX_lon = list()
testX_lon = list()
trainY_lon = list()
testY_lon = list()

filename = './0/2541.csv'
testfile = './0/2561.csv'
earth_R = 6371 * 1000.0
#lat = list()
#lon = list()
#---------------------- Hyper parameters---------------------------------------
layers = [2,20,40,1]
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
    trainlistX_lat = list()
    trainlistY_lat = list()

    trainlistX_lon = list()
    trainlistY_lon = list()
    
    #Read data and make seq matrix
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        reader = list(reader)
        for i in range(len(reader)):
            if i > 0:
                #Generate training seq here
                tmp = [float(reader[i][1]),float(reader[i][3])]
                trainlistX_lon.append(tmp)
                tmp = [float(reader[i][2]),float(reader[i][3])]
                trainlistX_lat.append(tmp)
                if i < (len(reader) - 1):
                    trainlistY_lon.append(float(reader[i+1][1]))
                    trainlistY_lat.append(float(reader[i+1][2]))
                else:
                    trainlistY_lon.append(float(reader[i][1]))
                    trainlistY_lat.append(float(reader[i][2]))
    return (trainlistX_lat,trainlistY_lat ,trainlistX_lon,trainlistY_lon )
        
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
    
def extract_data(lat_data,lon_data):
    lon= list()
    lat = list()
    dataset_lat = list(lat_data)
    dataset_lon = list(lon_data)
    
    for i in range(len(dataset_lat)):
        lat.append(dataset_lat[i][0])
        lon.append(dataset_lon[i][0])
    return (lon,lat)


def type1_split():
    global trainX_lat, testX_lat, trainY_lat, testY_lat
    global trainX_lon, testX_lon, trainY_lon, testY_lon
    
    test_lastIndex = trainX_lat.shape[0]
    train_lastIndex = int(test_lastIndex * 0.8)
    
    print test_lastIndex, train_lastIndex
    
    testX_lat = trainX_lat[train_lastIndex:test_lastIndex,:]
    testY_lat = trainY_lat[train_lastIndex:test_lastIndex]
    trainX_lat = trainX_lat[0:train_lastIndex,:]
    trainY_lat = trainY_lat[0:train_lastIndex]

    testX_lon = trainX_lon[train_lastIndex:test_lastIndex,:]
    testY_lon = trainY_lon[train_lastIndex:test_lastIndex]
    trainX_lon = trainX_lon[0:train_lastIndex,:]
    trainY_lon = trainY_lon[0:train_lastIndex]
    
    
def type2_split(testfile):
    global testX, testY
    
    testX,testY = read_csv(testfile)
    testX= np.array(testX)
    testY= np.array(testY)
    
def model_RNN(traindata,testdata):
    model = Sequential()
    model.add(LSTM(layers[1], input_dim=layers[0],return_sequences=True,activation='softmax'))
    model.add(LSTM(layers[2], input_dim=layers[1],return_sequences=False,activation='softmax'))
    model.add(Dropout(drop_out))
    model.add(Dense(output_dim=layers[3],activation='linear'))
    sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="mse", optimizer=sgd)
    model.fit(traindata[0], traindata[1], nb_epoch=no_epoch, batch_size=batch_size, verbose=2)

    trainPredict = model.predict(traindata[0])
    testPredict = model.predict(testdata[0])
    
    return (model,trainPredict,testPredict)
    
trainX_lat,trainY_lat,trainX_lon,trainY_lon = read_csv(filename)       
trainX_lat = np.array(trainX_lat)
trainY_lat = np.array(trainY_lat)
trainX_lon = np.array(trainX_lon)
trainY_lon = np.array(trainY_lon)
type1_split()
#type2_split(testfile)

trainX_lat = np.reshape(trainX_lat, (trainX_lat.shape[0],1, trainX_lat.shape[1]))
testX_lat = np.reshape(testX_lat, (testX_lat.shape[0],1, testX_lat.shape[1]))

trainX_lon = np.reshape(trainX_lon, (trainX_lon.shape[0],1, trainX_lon.shape[1]))
testX_lon = np.reshape(testX_lon, (testX_lon.shape[0],1, testX_lon.shape[1]))

model_lat, trainPredict_lat, testPredict_lat = model_RNN([trainX_lat,trainY_lat],[testX_lat,testY_lat])
model_lon, trainPredict_lon, testPredict_lon = model_RNN([trainX_lon,trainY_lon],[testX_lon,testY_lon])

draw_map(list(testY_lat),list(testY_lon),list(testPredict_lat),list(testPredict_lon))