#Belgium 2018
#Author: Mathis Van Eetvelde


#Imports
import numpy as np 
import matplotlib.pyplot as plt
import os
import time
import sys
import cv2
import random
from tqdm import tqdm
import pickle
import tensorflow as tf 
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Flatten
import h5py

DATADIR = "Z:/ML_DATASETS/walk-or-run/walk_or_run_train/train"
CATEGORIES = ["walk","run"]
IMG_SIZE = 50
training_data = []

def msg(num):
	if(num == 1):
		print("{}- Creating dataset!".format(time.asctime())) 
	elif(num == 2):
		print("{}- Loading dataset!".format(time.asctime()))
	elif(num == 3):
		print("{}- Starting training!".format(time.asctime()))
	elif(num == 4):
		print("{}- Finished training!".format(time.asctime()))		
	return 0 

def create_training_data():
	training_data = []

	print("Creating training_data and resizing:")
	for category in CATEGORIES:
		X,y = [],[]
		path = os.path.join(DATADIR, category)
		classnum = CATEGORIES.index(category)
		
		for img in tqdm(os.listdir(path)):
			try:	
				img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
				new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
				#plt.imshow(new_array, cmap="gray")
				#plt.show()
				training_data.append([new_array, classnum])
				
			except Exception as e:
				pass
	
	random.shuffle(training_data)
	print("Sorting data:")

	for features, label in tqdm(training_data):
		X.append(features)
		y.append(label)

	X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
	X = X/255
	if True:
		pickleSave(X,y)
	return X,y



def pickleSave(X,y):
	print("Saving XY Dataset")
	try:
		pickle_out = open("X.pickle", "wb")
		pickle.dump(X, pickle_out)
		pickle_out.close()

		pickle_out = open("y.pickle", "wb")
		pickle.dump(y, pickle_out)
		pickle_out.close()
		print("Saving XY Dataset Complete")
	except Exception as e:
		print(e)

def pickleLoad():
	print("Loading XY Dataset")
	try:
		X = pickle.load(open("X.pickle","rb"))
		y = pickle.load(open("y.pickle","rb"))

		print("Loading XY Dataset Complete")	
	except Exception as e:
		print(e)
		exit()
	return X,y

def initModel():
	try:
		if(sys.argv[1]=='c'):  #add c to cmd command to "Create" dataset
			msg(1)
			X,y= create_training_data()
		else:
			msg(2)
			X,y= pickleLoad()
	except:
		msg(2)
		X,y= pickleLoad()

	return X,y


def train():
	msg(3)
	try:
		model = Sequential()
		model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2,2)))

		model.add(Conv2D(64, (3,3)))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2,2)))

		model.add(Flatten())
		model.add(Dense(64))

		model.add(Dense(1))
		model.add(Activation("sigmoid"))

		model.compile(loss="binary_crossentropy",
						optimizer="adam",
						metrics=['accuracy'])

		model.fit(X, y, batch_size=120,epochs=200, validation_split=0.1)
	except Exception as e:
		print(e)
	if True:
		model.save('walkorrunmodel.h5')
	msg(4)

X, y = initModel()
train()