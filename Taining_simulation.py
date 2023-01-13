print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utilis import *
from sklearn.model_selection import  train_test_split

#step1

path = 'myData'
data = importDataInfo(path)

#step2

data = balanceData(data,display=False)

#step3

imagesPath, steerings = loadData(path,data)
#print(imagesPath[0],steerings[0])

#step4

xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2,random_state=10)
#print('Total Training Images: ',len(xTrain))
#print('Total Validation Images: ',len(xVal))

#step5,6,7

#step8

model = createModel()
model.summary()

#step 9

history = model.fit(batchGen(xTrain, yTrain, 100, 1),
                                  steps_per_epoch=300,
                                  epochs=10,validation_data=batchGen(xVal, yVal, 100, 0),
                                  validation_steps=200)

#step 10

model.save('model.h5')
print('Model Saved')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()


