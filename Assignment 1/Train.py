from Model import create_model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(x_test.shape)


'''let's Reshape our vector from size,28,28 to size,784 '''
#flatten_size = 28 * 28 
#x_train = x_train.reshape(x_train.shape[0], flatten_size)
#x_test = x_test.reshape(x_test.shape[0], flatten_size)
x_train = x_train.astype('float32') #Convert Uint8 values 0-255 to float
x_test = x_test.astype('float32')
x_train /= 255                      #To normalize the values from 0-255 to 0-1
x_test /= 255

#Convert Train and Test data to one-hot
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

#---------------------------------------------
# The following method would create the model 
#---------------------------------------------
model = create_model()

def MSE(P,Y):
    return np.square(np.subtract(Y,P)).mean()

def Hinge(P,Y):
    return np.sum(np.max(0,(1-Y*P)))

def CrossEntropy(P,Y):
    loss = np.multiply(Y, np.log(P))
    return -np.mean(np.sum(loss))

#Accuracy between predictions and target assuming both are one-hot encoded.
def Accuracy(P, Y):
    acc = np.mean(np.argmax(P,axis=-1) == np.argmax(Y,axis=-1))
    acc *= 100
    return acc

def ConfusionMatrix(P,Y):
    pred = np.argmax(P,axis=-1)
    y_lab = np.argmax(Y, axis=-1)
    confusion_matrix = np.zeros((P.shape[1],P.shape[1]))
    for x,y in zip(pred, y_lab):
        confusion_matrix[x,y] += 1
    return confusion_matrix

def predict(model,X):
    # Returns the last layer output in Batchsize*classes shape
    pred = model.forward(X)
    return pred[-1]

def train(model,X,y):
    layer_activations = model.forward(X)
    layer_inputs = [X]+layer_activations
    predicts = layer_activations[-1]
    mse_loss = MSE(predicts,y)
    loss = CrossEntropy(predicts,y)
    model.backward(layer_inputs,y)
    return

def ROC(P, Y):
    cf_matrix = ConfusionMatrix(P, Y)
    FP = cf_matrix.sum(axis=0) - np.diag(cf_matrix) 
    FN = cf_matrix.sum(axis=1) - np.diag(cf_matrix)
    TP = np.diag(cf_matrix)
    TN = cf_matrix.sum() - (FP + FN + TP)
    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP) 
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    FDR = FP/(TP+FP)
    ACC = (TP+TN)/(TP+FP+FN+TN)

    
train_log = []
batchsize = 100
epochs = 10

#Training

for epoch in range(epochs):
    for i in range(0, x_train.shape[0], batchsize):
        x_train_mini = x_train[i:i + batchsize]
        y_train_mini = y_train[i:i + batchsize]
        train(model,x_train_mini,y_train_mini)
        
        
    outs = predict(model,x_train)
    train_log.append(Accuracy(outs,y_train))
    print(train_log)

print("Train accuracy:",train_log)

finalPredicts = predict(model,x_test)
testAcc = Accuracy(finalPredicts, y_test)
print("Final Test accuracy of the model after training for",epochs, " epochs:", testAcc)
print("Confusion Matrix:", ConfusionMatrix(finalPredicts,y_test))

ROC(finalPredicts,y_test)
plt.plot(train_log,label='train accuracy')
plt.show()








