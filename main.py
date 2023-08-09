
import os
import random
import pygame
import pickle

import numpy
from numpy import dot # numpy.dot(x,y) = sum(x * y) = x @ y

import uuid
# uuid.uuid3(uuid.NAMESPACE_DNS, '')


# def load_dataset():
# 	with numpy.load("mnist.npz") as file:
# 		# convert from RGB to Unit RGB
# 		x_train = file['x_train'].astype("float32") / 255

# 		# reshape from (60000, 28, 28) into (60000, 784)
# 		x_train = numpy.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))

# 		# labels
# 		y_train = file['y_train']

# 		# convert to output layer format
# 		y_train = numpy.eye(10)[y_train]

# 		return x_train, y_train

def uniform(w, h):
	return numpy.random.uniform(-0.5, 0.5, (w, h))

def zeros(w, h):
	return numpy.zeros((w, h))

def sigmoid(x):
	return 1 / (1 + numpy.exp(-x))

def sigmoid_derivative(x):
	return x * (1 - x)

def load_image(path):
	image = pygame.image.load(path)
	image_array = pygame.surfarray.array3d(image)
	array = numpy.zeros(image_array.shape[:2])
	for x in range(array.shape[0]):
		for y in range(array.shape[1]):
			array[x,y] = sum(image_array[x, y]) / 3 / 255
	return numpy.reshape(array, (784, 1))

def load_dataset(path):
	table = numpy.load(path)
	data = []
	w,h = table.shape[:2]
	for x in range(w):
		for y in range(h):
			image = numpy.reshape(table[x,y], (784, 1))
			label = zeros(10,1)
			label[y] = 1
			data.append((image, label))
	return data


class Neural_Network:
	epoch_counter = 0

	def __init__(self, *args, rate=0.01, loader='new'):
		self.rate = rate
		self.W = [] # weights
		self.B = [] # bias
		self.I = [] # inputs

		for index in range(len(args[:-1])):
			self.W.append(uniform(args[index+1], args[index]))
			self.B.append(zeros(args[index+1], 1))

	def epoch(self, data):
		print(f"Эпоха №{self.epoch_counter}")
		for image, label in data:
			output = self.forward_propagation(image)
			self.back_propagation(output, label)
		self.epoch_counter += 1

	def forward_propagation(self, input):
		self.I = []
		for bias, weights in zip(self.B, self.W):
			self.I.append(input)
			input = sigmoid(bias + weights @ input)
		return input

	def back_propagation(self, output, label):
		# Loss / Error calculation
		# e_loss += 1 / len(output) * np.sum((output - label) ** 2, axis=0)
		# e_correct += int(np.argmax(output) == np.argmax(label))

		delta = output - label
		for weights, bias, input in zip(self.W[::-1], self.B[::-1], self.I[::-1]):
			weights += -self.rate * delta @ numpy.transpose(input)
			bias += -self.rate * delta

			# дельта для следующей итерации
			delta = numpy.transpose(weights) @ delta * sigmoid_derivative(input)


if __name__ == '__main__':
	# data = load_dataset('table.numpy') # датасет на 200 изображений
	# random.shuffle(data)

	# nn = Neural_Network(28*28, 64, 16, 10)

	# for _ in range(10):
	# 	nn.epoch(data)

	# with open('nn.pickle', 'wb') as file:
	# 	pickle.dump(nn, file)

	with open('nn.pickle', 'rb') as file:
		nn = pickle.load(file)

	image = load_image('test.png')
	output = numpy.reshape(nn.forward_propagation(image), (10)).tolist()
	index = output.index(max(output))
	print('test.png содержит число:', index)
 