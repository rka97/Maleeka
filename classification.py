import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from common import image_autocrop
from skimage import io
from os import listdir
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from generate_dataset import init_letters
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from generate_dataset import GENERATED_DATASET_DIRECTORY
from sklearn.neighbors import KNeighborsClassifier

"""
	14 features:
		4 global:
			* height / width
			* Number of white pixels / number of black pixels.
			* Number of vertical transitions
			* Number of horizontal transitions.
		10 local:
			* White Pixels in Region 1/ Black Pixels in Region 1.
			* White Pixels in Region 2/ Black Pixels in Region 2.
			* White Pixels in Region 3/ Black Pixels in Region 3.
			* White Pixels in Region 4/ Black Pixels in Region 4.
			* White Pixels in Region 1/ White Pixels in Region 2.
			* White Pixels in Region 3/ White Pixels in Region 4.
			* White Pixels in Region 1/ White Pixels in Region 3.
			* White Pixels in Region 2/ White Pixels in Region 4.
			* White Pixels in Region 1/ White Pixels in Region 4
			* White Pixels in Region 2/ White Pixels in Region 3.
"""


def extract_features(img):
	cropped_img = image_autocrop(img)
	img_height, img_width = cropped_img.shape
	white_1 = np.sum(cropped_img[:img_height // 2, :img_width // 2])
	white_2 = np.sum(cropped_img[:img_height // 2, img_width // 2:])
	white_3 = np.sum(cropped_img[img_height // 2:, :img_width // 2])
	white_4 = np.sum(cropped_img[img_height // 2:, img_width // 2:])
	black_1 = (img_height // 2) * (img_width // 2) - white_1
	black_2 = (img_height // 2) * (img_width - img_width // 2) - white_2
	black_3 = (img_height - img_height // 2) * (img_width // 2) - white_3
	black_4 = (img_height - img_height // 2) * \
	           (img_width - img_width // 2) - white_4
	white = white_1 + white_2 + white_3 + white_4
	black = img_height * img_width - white
	horizontal_transitions = np.sum(np.abs(np.diff(cropped_img)))
	vertical_transitions = np.sum(np.abs(np.diff(cropped_img, axis=0)))

	return np.array([
		[
			img_height / (img_width + 1),
			white / (black + 1),
			vertical_transitions,
			horizontal_transitions,
			white_1 / (black_1 + 1),
			white_2 / (black_2 + 1),
			white_3 / (black_3 + 1),
			white_4 / (black_4 + 1),
			white_1 / (white_2 + 1),
			white_3 / (white_4 + 1),
			white_1 / (white_3 + 1),
			white_2 / (white_4 + 1),
			white_1 / (white_4 + 1),
			white_2 / (white_3 + 1)
		]
	])


def load_data():
	data = np.empty(shape=[0, 14])
	labels = np.empty(shape=[0, 1])
	for letter, letter_data in init_letters.items():
		forms = letter_data["Forms"]
		for form in forms:
			images = listdir(GENERATED_DATASET_DIRECTORY + letter + '/' + form + '/')
			for img_name in images:
				image = io.imread(GENERATED_DATASET_DIRECTORY +
				                  letter + '/' + form + '/' + img_name)
				features = extract_features(image)
				data = np.append(data, features, axis=0)
				labels = np.append(labels, letter + "_" + form)
		break
	return data, labels


def main():
	data, labels = load_data()
	print("data loaded")
	onehot_encoder = OneHotEncoder(sparse=False)
	labels = labels.reshape(len(labels), 1)
	onehot_encoded = onehot_encoder.fit_transform(labels)
	print("data encoded")
	X_train, X_test, y_train, y_test = train_test_split(
	    data, labels, test_size=0.20)
	print("data split")


	model = KNeighborsClassifier(n_neighbors=9)
	# model = SVC(kernel='linear')

	# Train the model using the training sets
	model.fit(X_train,y_train.ravel()) 
	print("classifier fit")

	y_pred = model.predict(X_test)
	print(confusion_matrix(y_test,y_pred))
	print(classification_report(y_test,y_pred))
	
	
	# # invert encoded letter
	# inverted = onehot_encoder.inverse_transform([onehot_encoded[0, :], onehot_encoded[5000]])
	# print(inverted)

main()
