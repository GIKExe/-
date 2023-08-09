import pygame
import numpy

image = pygame.image.load('data-new.png')
raw = pygame.Surface((42, 39))

for y in range(10):
	for x in range(20):
		raw.blit(image, (-x*42, -y*39))
		tile = pygame.transform.scale(raw, (28,28))
		pygame.image.save(tile, f'{y}/{x}.png')