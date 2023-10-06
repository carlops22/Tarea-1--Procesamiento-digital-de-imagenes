
#%% Desplegar imagen original
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
# Ejercicio 1


## Parte 1
## Imagen cargada con Pillow
rgb_imagen = Image.open("rgb.png")

## Convertir imagen a un arreglo numpy
rgb_array= np.array(rgb_imagen)
## Mostrar imagen original usando plt

plt.figure(figsize=(6, 6))
plt.imshow(rgb_imagen)
plt.title("Imagen RGB original")
plt.axis('off')
plt.show()
rgb_imagen.close()

# %% Mostrar 3 canales de color
## Separar canales de color 
black_background = np.zeros_like(rgb_array)
r_imagen= black_background.copy()
r_imagen[:,:,0] = rgb_array[:,:,0]

g_imagen= black_background.copy()
g_imagen[:,:,1] = rgb_array[:,:,1]

b_imagen= black_background.copy()
b_imagen[:,:,2] = rgb_array[:,:,2]


## Mostrar cada canal usando subplots
plt.figure(figsize=(6, 6))
 
plt.subplot(1,3,1)
plt.imshow(r_imagen,cmap='Reds')
plt.axis('off')
plt.title("Canal rojo")

plt.subplot(1,3,2)
plt.imshow(g_imagen, cmap='Greens')
plt.axis('off')
plt.title("Canal verde")

plt.subplot(1,3,3)
plt.imshow(b_imagen,cmap='Blues')
plt.axis('off')
plt.title("Canal azul")


plt.tight_layout()
plt.show()


#%% Obtenga, por separado, las combinaciones de colores de cada circulo
## Cargar imagen original con cv2
rgb_imagen = cv2.imread("rgb.png")
rgb_imagen = cv2.cvtColor(rgb_imagen, cv2.COLOR_BGR2RGB)

## Extraer canales utilizando split()
R,G,B = cv2.split(rgb_imagen)

## Aplicar máscara de los canales extraidos a la imagen original
red_img = cv2.bitwise_and(rgb_imagen, rgb_imagen, mask=R)
green_img = cv2.bitwise_and(rgb_imagen, rgb_imagen, mask=G)
blue_img = cv2.bitwise_and(rgb_imagen, rgb_imagen, mask=B)

## Mostrar los resultados horizontalmente con subplots
plt.figure(figsize=(6, 6))

plt.subplot(1,3,1)
plt.imshow(red_img)
plt.axis('off')
plt.title("Mascara roja")

plt.subplot(1,3,2)
plt.imshow(green_img)
plt.axis('off')
plt.title('Mascara verde')

plt.subplot(1,3,3)
plt.imshow(blue_img)
plt.axis('off')
plt.title('Mascara azul')

plt.show()

# %% De la imagen original elimine las letras R G y B ydespliegue la imagen a color sin estas letras.
# Cambiar el tipo de los canales a variable uint8 para extraer componentes conexas
red_uint8 = R.astype('uint8')
green_uint8 = G.astype('uint8')
blue_uint8 = B.astype('uint8')

labels_r, red = cv2.connectedComponents(red_uint8, connectivity = 4)
labels_g, green = cv2.connectedComponents(green_uint8, connectivity = 4)
labels_b, blue = cv2.connectedComponents(blue_uint8, connectivity = 4)

##Obtener mascara de componentes que no son letras
mask_red=red==2
mask_green=green==1
mask_blue=blue==1

##Cambiar a variable uint8 para aplicar mascaras
mask_red = mask_red.astype('uint8')
mask_green = mask_green.astype('uint8')
mask_blue = mask_blue.astype('uint8')
## Aplicar mascaras a imagen
red_imagen = cv2.bitwise_and(rgb_imagen, rgb_imagen, mask=mask_red)
green_imagen = cv2.bitwise_and(rgb_imagen, rgb_imagen, mask=mask_green)
blue_imagen = cv2.bitwise_and(rgb_imagen, rgb_imagen, mask=mask_blue)



## Apilar imagenes con merge 
# Hacer imagen con las mismas dimensiones que la original
merged_image = np.zeros_like(rgb_imagen)
## Asignar a cada canal su color respectivo
merged_image[:, :, 0] = mask_red * rgb_imagen[:, :, 0]  # Red channel
merged_image[:, :, 1] = mask_green * rgb_imagen[:, :, 1]  # Green channel
merged_image[:, :, 2] = mask_blue * rgb_imagen[:, :, 2]  # Blue channel


## Mostrar imagen original sin letras
plt.figure(figsize=(6, 6))
plt.imshow(merged_image)
plt.axis('off')
plt.title("Imagen Sin Letras")
plt.show()


