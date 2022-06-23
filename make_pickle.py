import cv2
import glob
import numpy as np
import pickle   
x_train = []
y_train = []
x_test = []
y_test = []

def make_pickle(fname, stt):
    for filename in glob.glob("D:\MOPHAT\DATASETS_SPLIT\TRAIN"+"/" +str(fname)+"/*.png"):
        img = cv2.resize(cv2.imread(filename), (150,150))
        x_train.append(img)
        y_train.append(stt)
    for filename in glob.glob("D:\MOPHAT\DATASETS_SPLIT\TEST"+"/" +str(fname)+"/*.png"):
        img = cv2.resize(cv2.imread(filename), (150,150))
        x_test.append(img)
        y_test.append(stt)

make_pickle("1.CAT", 0)
make_pickle("2.DOG", 1)
make_pickle("3.ROOSTER", 2)
make_pickle("4.CRICKET", 3)
make_pickle("5.DUCK", 4)
make_pickle("6.FROG", 5)
make_pickle("7.MONKEY", 6)
make_pickle("8.GOAT", 7)
make_pickle("9.ELEPHANT", 8)
make_pickle("10.COW", 9)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
with open("Classification_CNN.pickle", "wb") as f:
    pickle.dump([(x_train, y_train), (x_test, y_test)], f)
print("finish")

