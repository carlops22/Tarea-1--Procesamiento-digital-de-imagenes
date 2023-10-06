#%% Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

#%% Parte a

'''
Cargue la imagen heart.png y despliegue la imagen en escala de grises junto con su histo-
grama.
'''
#Cargar imagen y convertirla a escala de grises
heart_bgr = cv2.imread("heart.png")
heart_gray = cv2.cvtColor(heart_bgr, cv2.COLOR_BGR2GRAY)

#Desplegar imagen en escala de grises junto a su histograma

hist,bins = np.histogram(heart_gray,256,[0,255]) ##histograma de escala de grises

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(heart_gray, cmap='gray')
plt.title('Imagen en escala de grises')
plt.axis('off')

plt.subplot(1,2,2)
plt.bar(range(len(hist)), hist)
plt.title('Histograma de imagen')

plt.show()