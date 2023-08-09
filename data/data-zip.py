import pygame
import numpy

def load_image(path):
	image = pygame.image.load(path)
	image_array = pygame.surfarray.array3d(image)
	array = numpy.zeros(image_array.shape[:2])
	for x in range(array.shape[0]):
		for y in range(array.shape[1]):
			array[x,y] = sum(image_array[x,y]) / 3 / 255
	return array

table = numpy.zeros((20, 10, 784))

for y in range(10):
	for x in range(20):
		array = load_image(f'{y}/{x}.png')
		table[x,y] = numpy.reshape(array, (784))

numpy.save('table', table)