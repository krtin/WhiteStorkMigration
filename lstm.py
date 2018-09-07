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
from keras.layers.core import Dense, Dropout, Masking
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD
import numpy as np
import csv
import math
from sklearn.metrics import mean_squared_error
#from pandas import DataFrame
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas
from sklearn.model_selection import train_test_split
#------------------- Global parameters ---------------------------------------
trainX_lat = list()
testX_lat = list()
trainY_lat = list()
testY_lat = list()

trainX_lon = list()
testX_lon = list()
trainY_lon = list()
testY_lon = list()

r_trainX_lat = list()
r_testX_lat = list()
r_trainY_lat = list()
r_testY_lat = list()

r_trainX_lon = list()
r_testX_lon = list()
r_trainY_lon = list()
r_testY_lon = list()

#filename = './0/2541.csv'
#testfile = './0/2561.csv'
filename = 'features/lstm/type1/0/2541.csv'
testfile = 'features/lstm/type1/0/2561.csv'
earth_R = 6371 * 1000.0
sequence_length = 5
ave_points = 1.0
span = 6 #1 * 24 * 60.0/5
#lat = list()
#lon = list()
#---------------------- Hyper parameters---------------------------------------
layers = [sequence_length,100,100,1]
no_epoch = 10
batch_size = 16
drop_out = 0.2
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
    
    sum_lon = 0
    sum_lat = 0
    #temperature = list()
    #Read data and make seq matrix
    
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        reader = list(reader)
        for i in range(len(reader)):


            if i > 0:
                #Generate training seq here
                
                if (i%10 == 0):
                #tmp = [float(reader[i][1]),float(reader[i][3])]
                    trainlistX_lon.append(sum_lon)
                #tmp = [float(reader[i][2]),float(reader[i][3])]
                    trainlistX_lat.append(sum_lat)
                if i < (len(reader) - 1):
                    trainlistY_lon.append(float(reader[i+1][1]))
                    trainlistY_lat.append(float(reader[i+1][2]))
                else:
                    trainlistY_lon.append(float(reader[i][1]))
                    trainlistY_lat.append(float(reader[i][2]))

    final_Xlat = []
    print len(trainlistX_lat) -sequence_length
    for index in range(len(trainlistX_lat) - sequence_length):
        final_Xlat.append(trainlistX_lat[index: index + sequence_length])
        #print len(final_X)
    final_Xlat = np.array(final_Xlat)

    final_Xlon = []
    print len(trainlistX_lon) -sequence_length
    for index in range(len(trainlistX_lon) - sequence_length):
        final_Xlon.append(trainlistX_lon[index: index + sequence_length])
        #print len(final_X)
    final_Xlon = np.array(final_Xlon)

    return (final_Xlon,final_Xlat ,trainlistY_lon,trainlistY_lat)

def read_csv2(filename):
    global r_trainX_lat ,r_testX_lat, r_trainY_lat, r_testY_lat
    global r_trainX_lon ,r_testX_lon, r_trainY_lon, r_testY_lon

    trainlistX_lat = list()
    trainlistY_lat = list()

    trainlistX_lon = list()
    trainlistY_lon = list()
    
    sum_lon = 0
    sum_lat = 0
    temperature = list()
    #Read data and make seq matrix
    
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        reader = list(reader)
        for i in range(len(reader)):
            if i > 0:
                #Generate training seq here
                sum_lon += float(reader[i][1])
                sum_lat += float(reader[i][2])
                if (i%ave_points == 0.0):
                #tmp = [float(reader[i][1]),float(reader[i][3])]
                    trainlistX_lon.append(sum_lon/ave_points)#,float(reader[i][3])])
                #tmp = [float(reader[i][2]),float(reader[i][3])]
                    trainlistX_lat.append(sum_lat/ave_points)#,float(reader[i][3])])
                    temperature.append(float(reader[i][3]))
                    sum_lon = 0
                    sum_lat = 0


    r_trainX_lon = np.array(trainlistX_lon)
    r_trainX_lat = np.array(trainlistX_lat)

    for i in range(len(trainlistX_lon)-1):
        r_trainY_lon.append(trainlistX_lon[i+1])
        r_trainY_lat.append(trainlistX_lat[i+1])

    r_trainY_lon.append(trainlistX_lon[-1])
    r_trainY_lat.append(trainlistX_lat[-1])

    raw_dataTest = {'latitude': trainlistX_lat,'longitude':trainlistX_lon }
    dfTest_smooth = pandas.DataFrame(raw_dataTest, columns = ['latitude', 'longitude'])


    smoothdata = pandas.ewma(dfTest_smooth,span=span)
    
    trainlistX_lat = smoothdata["latitude"].tolist()
    trainlistX_lon = smoothdata["longitude"].tolist()
    
    for i in range(len(trainlistX_lon)-1):
        trainlistY_lon.append(trainlistX_lon[i+1])
        trainlistY_lat.append(trainlistX_lat[i+1])

    trainlistY_lon.append(trainlistX_lon[-1])
    trainlistY_lat.append(trainlistX_lat[-1])

    print "lenghts of average lists: ", len(trainlistX_lat), len(trainlistY_lat)
    print "lenghts of average lists: ", len(trainlistX_lon), len(trainlistY_lon)
    
    
    txl = []
    txlo = []
    for i in range(len(trainlistX_lat)):
        txl.append([trainlistX_lat[i],temperature[i]])
        txlo.append([trainlistX_lon[i],temperature[i]])

    trainlistX_lon = txlo
    trainlistX_lat = txl

    final_Xlat = []
    
    for index in range(len(trainlistX_lat) - sequence_length):
        final_Xlat.append(trainlistX_lat[index: index + sequence_length])
        #print len(final_X)
    final_Xlat = np.array(final_Xlat)

    final_Xlon = []
    
    for index in range(len(trainlistX_lon) - sequence_length):
        final_Xlon.append(trainlistX_lon[index: index + sequence_length])
        #print len(final_X)
    final_Xlon = np.array(final_Xlon)

    return (final_Xlon,final_Xlat,trainlistY_lon,trainlistY_lat)


def draw_map(latTest,lonTest,latPre,lonPre,c1,c2):
    from matplotlib import pyplot as plt

    raw_dataTest = {'latitude': latTest,'longitude':lonTest }
    dfTest = pandas.DataFrame(raw_dataTest, columns = ['latitude', 'longitude'])
    raw_dataPre = {'latitude': latPre,'longitude':lonPre }
    dfPre = pandas.DataFrame(raw_dataPre, columns = ['latitude', 'longitude'])
    
    fig = plt.figure(figsize=(20,10))
    
    map = Basemap(projection='gall', resolution = 'l',area_thresh = 100000.0,lat_0=0, lon_0=0)
    map.drawcoastlines()
    map.drawcountries()
    map.fillcontinents(color = '#888888')
    map.drawmapboundary(fill_color='#f4f4f4')
    x,y = map(dfTest['longitude'].values, dfTest['latitude'].values)
    xP,yP = map(dfPre['longitude'].values, dfTest['latitude'].values)
    map.plot(x, y, 'ro', markersize=6,color=c1)
    map.plot(xP, yP, 'ro', markersize=6,color=c2)
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
    global r_trainX_lat ,r_testX_lat, r_trainY_lat, r_testY_lat
    global r_trainX_lon ,r_testX_lon, r_trainY_lon, r_testY_lon
    
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

    r_testY_lat = trainY_lat[train_lastIndex:test_lastIndex]
    r_trainY_lat = trainY_lat[0:train_lastIndex]
    r_testY_lon = trainY_lon[train_lastIndex:test_lastIndex]
    r_trainY_lon = trainY_lon[0:train_lastIndex]
    
    
def type2_split(testfile):
    global testX, testY
    
    testX,testY = read_csv(testfile)
    testX= np.array(testX)
    testY= np.array(testY)
    
def model_RNN(traindata,testdata):
    X_train, X_test, y_train, y_test = train_test_split(traindata[0], traindata[1], test_size=0.2)

    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(X_train.shape[1], traindata[0].shape[2])))
    model.add(LSTM(layers[1], input_shape=(X_train.shape[1], X_train.shape[2]),return_sequences=True,activation='linear'))
    #model.add(Dropout(drop_out))

    #for x in xrange(1,len(layers)):
    model.add(LSTM(layers[2],return_sequences=False,activation='linear'))
    #model.add(Dropout(drop_out))

    model.add(Dense(output_dim=layers[3],activation='linear'))
    #model.add(Dropout(drop_out))

    sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="mse", optimizer= 'adamax')

    #validation_data=(testdata[0], testdata[1]),
    hist = model.fit(X_train, y_train, nb_epoch=no_epoch, batch_size=batch_size, verbose=2,validation_data=(X_test, y_test))

    trainPredict = model.predict(traindata[0])
    testPredict = model.predict(testdata[0])
    
    return (model,trainPredict,testPredict,hist)
    
trainX_lon,trainX_lat ,trainY_lon,trainY_lat = read_csv2(filename)  

trainX_lat = np.array(trainX_lat)
trainY_lat = np.array(trainY_lat)
trainX_lon = np.array(trainX_lon)
trainY_lon = np.array(trainY_lon)

print "before split", trainX_lat.shape, trainY_lat.shape
#type2_split(testfile)
type1_split()
print "After split", trainX_lat.shape, trainY_lat.shape

trainX_lat = np.reshape(trainX_lat, (trainX_lat.shape[0],trainX_lat.shape[1],2))
testX_lat = np.reshape(testX_lat, (testX_lat.shape[0],testX_lat.shape[1],2))

trainX_lon = np.reshape(trainX_lon, (trainX_lon.shape[0],trainX_lon.shape[1],2))
testX_lon = np.reshape(testX_lon, (testX_lon.shape[0],testX_lon.shape[1],2))

print "actual model shape after reshape: " , trainX_lat.shape, testX_lat.shape

resultFile = open("results/rnn/partial/input_s2Train.csv", 'wb')
wr = csv.writer(resultFile, dialect='excel')
wr.writerow(['longitude','latitude'])
for i in range(len(trainY_lat)):
    wr.writerow([trainY_lon[i],trainY_lat[i],])

resultFile = open("results/rnn/partial/input_s2Test.csv", 'wb')
wr = csv.writer(resultFile, dialect='excel')
wr.writerow(['longitude','latitude'])
for i in range(len(testY_lat)):
    wr.writerow([testY_lon[i],testY_lat[i],])



model_lat, trainPredict_lat, testPredict_lat, hist_lat = model_RNN([trainX_lat,trainY_lat],[testX_lat,testY_lat])
model_lon, trainPredict_lon, testPredict_lon, hist_lon = model_RNN([trainX_lon,trainY_lon],[testX_lon,testY_lon])

hist_lon_dict = hist_lon.history
hist_lat_dict = hist_lat.history

plt.figure(1)
ax = plt.subplot(211)
#plt.plot(hist_lon_dict['loss'], hist_lon_dict['val_loss'], hist_lat_dict['loss'], hist_lat_dict['val_loss'])
loss1 = plt.plot(range(len(hist_lon_dict['val_loss'])), hist_lon_dict['val_loss'], 'r', label='Validation Loss')
loss2 = plt.plot(range(len(hist_lon_dict['loss'])), hist_lon_dict['loss'], 'g', label='Training Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
plt.legend(handles=[loss1, loss2])
ax.set_title('Longitude Model Loss Curve')


ax = plt.subplot(212)
loss1 = plt.plot(range(len(hist_lat_dict['val_loss'])), hist_lat_dict['val_loss'], 'r')
loss2 = plt.plot(range(len(hist_lat_dict['loss'])), hist_lat_dict['loss'], 'g')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
plt.legend(handles=[loss1, loss2])
ax.set_title('Latitude Model Loss Curve')

#plt.plot(hist_lon_dict['loss'], hist_lon_dict['val_loss'])
#plt.subplot(212)
#plt.plot(hist_lat_dict['loss'], hist_lat_dict['val_loss'])

resultFile = open("type1_Train.csv", 'wb')
wr = csv.writer(resultFile, dialect='excel')
wr.writerow(['longitude','latitude'])
for i in range(len(trainX_lat)):
    wr.writerow([trainPredict_lon[i],trainPredict_lat[i],])

resultFile = open("type1_Test.csv", 'wb')
wr = csv.writer(resultFile, dialect='excel')
wr.writerow(['longitude','latitude'])
for i in range(len(trainX_lat)):
    wr.writerow([testPredict_lon[i],testPredict_lat[i],])

draw_map(list(testY_lat),list(testY_lon),list(testPredict_lat),list(testPredict_lon),'r','g')
draw_map(list(trainY_lat),list(trainY_lon),list(trainPredict_lat),list(trainPredict_lon),'m','y')
draw_map(list(trainY_lat[:len(testY_lat)]),list(trainY_lon[:len(testY_lon)]),list(testY_lat),list(testY_lon),'r','b')


