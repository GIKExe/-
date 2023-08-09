import pygame
import numpy

table = numpy.load('table.numpy')

def get_image(x,y):
	array = numpy.reshape(table[x,y], (28,28))
	image_array = numpy.zeros((28,28,3), dtype='u1')
	for x in range(28):
		for y in range(28):
			c = int(array[x,y] * 255)
			image_array[x,y] = [c,c,c]
	image = pygame.surfarray.make_surface(image_array)
	return pygame.transform.scale(image, (280,280))

win = pygame.display.set_mode((280,280))
running = True
clock = pygame.time.Clock()

x = 0
y = 0
image = get_image(x,y)

while running:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False

		elif event.type == pygame.KEYDOWN:
			if event.key == pygame.K_ESCAPE:
				running = False

			elif event.key == pygame.K_LEFT:
				if x > 0:
					x -= 1
					image = get_image(x,y)

			elif event.key == pygame.K_RIGHT:
				if x < 19:
					x += 1
					image = get_image(x,y)

			elif event.key == pygame.K_UP:
				if y > 0:
					y -= 1
					image = get_image(x,y)

			elif event.key == pygame.K_DOWN:
				if y < 9:
					y += 1
					image = get_image(x,y)

	win.fill((30,30,30))
	win.blit(image, (0,0))
	pygame.display.flip()
	clock.tick(60)