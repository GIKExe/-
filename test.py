
import pygame
import numpy

def load_image(path):
	image = pygame.image.load(path)
	image_array = pygame.surfarray.array3d(image)
	array = numpy.zeros(image_array.shape[:2])
	for x in range(array.shape[0]):
		for y in range(array.shape[1]):
			array[x, y] = sum(image_array[x, y]) / 3 / 255
	return array

# # test 1
# array = 1 - load_image('test.png')
# # print(array)
# numpy.save('test', array)

# # test 2
# array = numpy.load('test.npy')
# image_array = numpy.zeros((*array.shape, 3), dtype='u1')
# w,h = array.shape

# for x in range(w):
# 	for y in range(h):
# 		c = int(array[x,y] * 255)
# 		image_array[x,y] = [c,c,c]
# image = pygame.surfarray.make_surface(image_array)
# pygame.image.save(image, 'test_gray.png')

