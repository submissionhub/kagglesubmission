import numpy as np
import math
import pandas as pand
import matplotlib.pyplot as plt
from sklearn import linear_model as LM
from sklearn.preprocessing import PolynomialFeatures as PM
import data_fn as dt



DATA = pand.read_csv('College\Projects\ML_Kaggle\data_train.csv',sep=',').to_numpy()
TEST = pand.read_csv('College\Projects\ML_Kaggle\data_test.csv',sep=',').to_numpy()
 

(m,n) = DATA.shape
print('total dataset length: ',m)

#shorten data by factor of 'f' for faster debugging of algorithms
#f = 1
#k = int (m/f)
#DATA = DATA[0:k,:]
#(m,n) = DATYA.shape
#print ('short dataset length: ',m)





#remove rows with missing values
(dataCLEAN) = dt.cleanNUM(DATA,[1,3,5,10])
#(dataCLEAN) = dt.cleanstr(dataCLEAN,[1,3,5,10])

(m,n) = dataCLEAN.shape
print ('clean dataset length: ',m)

#split into test/train samples 
(dataTRAIN,dataTEST) = dt.split(dataCLEAN,0.6)

print('TEST/TRAIN length: ',int(m*0.6),' , ',int(m*0.4))

##
#test on sumbission data
##

#dataTRAIN = dataCLEAN
#dataTEST = TEST

#numerical data
#year of record/age/size of city/height/income (1,3,5,10)
dataNUM = dataTRAIN[:,1:2]
dataNUM = np.append(dataNUM,dataTRAIN[:,3:4],axis = 1)
dataNUM = np.append(dataNUM,dataTRAIN[:,5:6],axis = 1)
dataNUM = np.append(dataNUM,dataTRAIN[:,10:11],axis = 1)

testNUM = dataTEST[:,1:2]
testNUM = np.append(testNUM,dataTEST[:,3:4],axis = 1)
testNUM = np.append(testNUM,dataTEST[:,5:6],axis = 1)
testNUM = np.append(testNUM,dataTEST[:,10:11],axis = 1)


#data best represented by binary encoding
#gender(use 'one-hot' encoding)/university degree('one-hot')/wears glasses(already binary)/hair color('one-hot') (2,7,8,9)
dataBIN = dataTRAIN[:,2:3]
dataBIN = np.append(dataBIN,dataTRAIN[:,7:10],axis = 1)

testBIN = dataTEST[:,2:3]
testBIN = np.append(testBIN,dataTEST[:,7:10],axis = 1)

#from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import OneHotEncoder

#binCLEAN = dt.cleanbin(dataBIN)

#encodeL = LabelEncoder()
#dataL_En = encodeL.fit_transform(binCLEAN[:,1])

#encodeOH = OneHotEncoder(sparse= False)
#dataL_En = dataL_En.reshape(len(dataL_En),1)
#dataOH_En = encodeOH.fit_transform(dataL_En)
#print(dataOH_En)




#other data that requires transformation before inclusion in model
#country(convert to gdp?)/profession (custom float distribution encoding?) (4,6)
dataABS = dataTRAIN[:,4:5]
dataABS = np.append(dataABS,dataTRAIN[:,6:7],axis = 1)

testABS = dataTEST[:,4:5]
testABS = np.append(testABS,dataTEST[:,6:7],axis = 1)

#income values (11)
dataANS = dataTRAIN[:,11]
testANS = dataTEST[:,11]
#test ids
testITER = dataTEST[:,0]





M = dt.mean(testNUM)
testNUM = dt.addmean(testNUM,M)


##linear model

reg = LM.LinearRegression()
reg.fit(dataNUM,dataANS)



#print(reg.coef_)
#print(reg.intercept_)
#print(reg.predict([[2002,34,134674,189]]))


predL = []

#l = 0
#print('predicting linear')
#for row in testNUM:
#    try :
#        l = reg.predict([row])
#        predL = np.append(predL,[l])
#    except: 
#        i = 0
#        for n in row :
#            if (math.isnan(n)):
#                print ('replacing ', n , ' with ', M[i])
#                row[i] = M[i]
#            i = 1+i 
#        l = reg.predict([row])
#        predL = np.append(predL,l,axis = 0)

        
#np.savetxt('linear_predictions.csv',predL,delimiter=',')



##polynomial model

ply = LM.LinearRegression()
poly = PM(degree = 3)
dataPOLY = poly.fit_transform(dataNUM)
ply.fit(dataPOLY,dataANS)
testPOLY = poly.fit_transform(testNUM)

print('\npredicting polynomial') 
predP = []
p = 0
j = 0
for row in testPOLY:
    #print(row)
    try :
        p = ply.predict([row])
        predP = np.append(predP,[p])
    except: 
        i = 0
        for n in row :
            if (math.isnan(n)):
                print ('replacing ', n , ' with ', M[i])
                row[i] = M[i]
            i = 1+i 
        p = ply.predict([row])
        predP = np.append(predP,p)
    j = 1+j

np.savetxt('polynomial_predictions.csv',predP,delimiter=',')

#calculate cost
cost = predP - testANS
costTotal = 0
for c in cost : costTotal = costTotal + c*c
costTotal = (costTotal/len(cost))**(0.5)
print('estimated cost:',costTotal)


#print first n rows
n = 5
print ('\nsample predictions first ',n)
print ('    prediction      :      cost')
i = 0
for val in predP:
    print (val,cost[i])
    i = i+1
    if(i == n ): break





