#Setting backend for matplotlib
import matplotlib
matplotlib.use("Agg")

#Importing required libraries
from neural_model.minivggnet_rish import MiniVGGNet
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras import backend as K
from imutils import build_montages
import matplotlib.pyplot as plt
import numpy as np
import cv2

NUM_EPOCHS = 25
INIT_LR = 1e-2
BS = 32

print("Loading the fashion_mnist data")
((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()

if K.image_data_format() == "channels_first":
    trainX = trainX.reshape((trainX.shape[0], 1, 28, 28))
    testX = testX.reshape((trainY.shape[0], 1, 28, 28))

else:
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))

trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

trainY = np_utils.to_categorical(trainY, 10)
testY = np_utils.to_categorical(testY, 10)

labelNames = ["top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]

print("Compiling model")
opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR/ NUM_EPOCHS)
model = MiniVGGNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("Training model")
H = model.fit(trainX, trainY, batch_size=BS, validation_data=(testX, testY), epochs=NUM_EPOCHS)

preds = model.predict(testX)

print("Evaluating Network")
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1), target_names=labelNames))

#Plotting the training loss and accuracy
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")

model.save("fashion_mnist.model")

images = []

for i in np.random.choice(np.arange(0, len(testY)), size=(16,)):
    #classify the clothing
    probs = model.predict(testX[np.newaxis, i])
    prediction = probs.argmax(axis=1)
    label = labelNames[prediction[0]]

    if K.image_data_format() == "channels_first":
        image = (testX[i][0] * 255.0).astype("uint8")
    else:
        image = (testX[i] * 255.0).astype("uint8")

    color = (0, 255, 0)

    if prediction[0] != np.argmax(testY[i]):
        color = (0, 0, 255)

    image = cv2.merge([image] * 3)
    image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
    cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (0, 255, 0), 2)
    images.append(image)

montage = build_montages(images, (96, 96), (4, 4))[0]
cv2.imwrite("fashion_mnist.png", montage)
