import pygame
import numpy

image = pygame.image.load('data.png')
image_array = pygame.surfarray.array3d(image)
array = numpy.zeros(image_array.shape[:2])

for x in range(array.shape[0]):
	for y in range(array.shape[1]):

		# чёрно-белый фильтр
		c = int(sum(image_array[x,y]) / 3)

		# избавление от шума
		c = (0 if c < 128 else c)
		image_array[x,y] = [c,c,c]

image = pygame.surfarray.make_surface(image_array)
pygame.image.save(image, 'data-new.png')