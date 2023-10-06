#%% imports

import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image, ImageEnhance

#%% Parte a
'''
Cargue y despliegue tanto imagen a color, en escala de grises y el histograma de la imagen en
escala de grises. Comente su correspondencia con la composicion de la imagen
'''

## Cargar imagen de color y en escala de grises
mountains_color = cv2.imread("mountains.jpg")
mountains_gray = cv2.cvtColor(mountains_color, cv2.COLOR_BGR2GRAY)

#&& Desplegar imagen a color

plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(mountains_color, cv2.COLOR_BGR2RGB)) ##convertir BGR a RGB
plt.title("Imagen a Color")
plt.axis('off')
plt.show()

#%% Desplegar imagen en escala de grises

plt.figure(figsize=(6, 6))
plt.imshow(mountains_gray, cmap='gray')
plt.title("Imagen en escala de grises")
plt.axis('off')
plt.show()


#%% Desplegar histograma de escala de grises

hist,bins = np.histogram(mountains_gray,256,[0,255]) ##histograma de escala de grises

plt.figure(figsize=(6,6))
plt.bar(range(len(hist)), hist)
plt.title('Histograma de escala de grises')
plt.xlabel('Intensidad de pixeles')
plt.ylabel('Frecuencia (cantidad)')
plt.show()

"""
En este histograma se puede ver la distribución de intensidades de la imagen,
con un peak inicial cercano al 0 podemos decir que existen areas muy oscuras en la imagen,
también se puede observar la presencia de pequeños valles y alta concentración entre
0-120 aproximadamente, esto quiere decir que la imagen concentra la mayoría de pixeles en areas
más oscuras y por tanto en cuestion de brillo podría concluirse que es una imagen
que tiende a ser oscura.

"""

# %% Parte b
'''
 Despliegue el histograma acumulado de la imagen (de intensidades en escala de grises).
'''

hist_acc = np.cumsum(hist) ##acumulación del histograma anterior

##Despliegue del histograma
plt.figure(figsize=(6,6))
plt.bar(range(len(hist_acc)),hist_acc)
plt.title('Histograma acumulado')
plt.xlabel('Intensidad de pixeles')
plt.ylabel('Frecuencia acumulada')
plt.show()


# %% Parte c
'''
Realice un ajuste del brillo de la imagen a color utilizando el m ́etodo de la biblioteca PIL con
los factores 0.5 y 1.5 . Despliegue las im ́agenes resultantes en una sola figura y comente al
respecto
'''

## Cargar imagen con Pillow Image
mountains_original = Image.open("mountains.jpg")
## Ajuste de brillo con factor 0.5
mountains_B_0_5 = ImageEnhance.Brightness(mountains_original).enhance(0.5)

## Ajuste de brillo con factor 1.5
mountains_B_1_5= ImageEnhance.Brightness(mountains_original).enhance(1.5)

## Despliegue de imagenes resultantes
plt.figure(figsize=(12, 4))

plt.subplot(1,3,1)
plt.imshow(mountains_original)
plt.title('Original')
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(mountains_B_0_5)
plt.title('Brillo factor 0.5')
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(mountains_B_1_5)
plt.title('Brillo factor 1.5')
plt.axis('off')

plt.tight_layout()
plt.show()

'''
Como se puede observar, se puede ver un oscurecimiento en la imagen original aplicando
el factor 0.5 y como se aclara bastante aplicando el factor 1.5 aunque aun mantiene
zonas oscuras en la imagen.
'''
# %% Parte d

'''
Realice ajuste de contraste de la imagen usando la biblioteca PIL con dos factores de contraste
uno mayor y otro menor que 1. Luego despliegue tres figuras, una que contenga la imagen
original junto a su histograma y las otras con las im ́agenes resultantes de los ajustes de
contraste y sus histogramas respectivos. Utilice im ́agenes en escala de grises para mostrar sus
resultados y realizar el procesamiento. Comente acerca de los resultados obtenidos.
'''
# usando Pillow convertir array a Imagen

mountains_gris = Image.fromarray(mountains_gray)

# Usando los factores 0.5 y 1.5 de contraste

mountains_C_0_5 = ImageEnhance.Contrast(mountains_gris).enhance(0.5)
mountains_C_1_5 = ImageEnhance.Contrast(mountains_gris).enhance(1.5)


# Ploteo imagen gris + su histograma
hist,bins = np.histogram(mountains_gris,256,[0,255]) 

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.imshow(mountains_gris, cmap='gray')
plt.title('Imagen original (escala de grises)')

plt.subplot(1,2,2)
plt.bar(range(len(hist)),hist)
plt.title('Histograma imagen original')

plt.show()

# Ploteo imagen contraste 0.5 + su histograma

hist_c_0_5,bins_c_0_5 = np.histogram(mountains_C_0_5,256,[0,255]) 
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.imshow(mountains_C_0_5, cmap='gray')
plt.title('Imagen contraste 0.5')

plt.subplot(1,2,2)
plt.bar(range(len(hist_c_0_5)),hist_c_0_5)
plt.title('Histograma imagen contraste 0.5')

plt.show()

# Ploteo imagen contraste 1.5 + su histograma

hist_c_1_5,bins_c_1_5 = np.histogram(mountains_C_1_5,256,[0,255]) 
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.imshow(mountains_C_1_5, cmap='gray')
plt.title('Imagen contraste 1.5')

plt.subplot(1,2,2)
plt.bar(range(len(hist_c_1_5)),hist_c_1_5)
plt.title('Histograma imagen contraste 1.5')

plt.show()
# %% Parte e
'''
Realice la ecualizaci ́on de la imagen. Para ello debe convertir a modelo YUV, y ecualizar solo
el canal de intensidades (o escala de grises) de la imagen. Para la conversi ́on puede utilizar el
comando cv2.COLOR BGR2YUV
'''
 # Conversión a modelo YUV

mountains_yuv = cv2.cvtColor(mountains_color, cv2.COLOR_BGR2YUV)

# extraer solo el canal de intensidades :Y
canal_y = mountains_yuv[:,:,0]

# ecualizar el canal de intensidades
mountains_ec_y = cv2.equalizeHist(canal_y)

# reemplazar el canal de intensidades ecualizado en la imagen

mountains_yuv[:,:,0] = mountains_ec_y

#%% Parte f

'''
Despliegue la imagen original (a color) junto a la imagen ecualizada (a color). Comente su
resultado. La Figura 3 muestra la imagen ecualizada de referencia.
'''
# convertir de vuelta a BGR
mountains_ec = cv2.cvtColor(mountains_yuv, cv2.COLOR_YUV2BGR)

# Plotear imagen original de color vs imagen ecualizada a color

plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(mountains_color, cv2.COLOR_BGR2RGB)) ##convertir BGR a RGB
plt.title("Imagen a Color")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(mountains_ec, cv2.COLOR_BGR2RGB)) ##convertir BGR a RGB
plt.title("Imagen ecualizada")
plt.axis('off')
plt.show()
# %% Parte g

'''
Despliegue el histograma original de la imagen junto al histograma de la imagen ecualizada, y
despliegue el histograma acumulado original de la imagen junto al histograma acumulado de la
imagen ecualizada (histogramas de intensidades en escala de grises). Comente sus resultados.
'''
# Calcular el histograma ecualizado (canal y) y histograma acumulado
hist_y, bins_y = np.histogram(canal_y, 256, [0, 255])
hist_acc_y = np.cumsum(hist_y)

# Plotear histograma original vs histograma ecualizado
plt.figure(figsize=(12, 4))
plt.subplot(1,2,1)
plt.title('Histograma original') 
plt.bar(range(len(hist)),hist,width=3) 


plt.subplot(1,2,2)
plt.title('Histograma ecualizado') 
plt.bar(range(len(hist_y)),hist_y,width=3) 

# Plotear histograma original acumulado vs histograma acumulado ecualizado

plt.figure(figsize=(12, 4))
plt.subplot(1,2,1)
plt.title('Histograma acumulado original') 
plt.bar(range(len(hist_acc)),hist_acc, width=3) 


plt.subplot(1,2,2)
plt.title('Histograma acumulado ecualizado') 
plt.bar(range(len(hist_acc_y)),hist_acc_y, width=3) 

'''
 Se puede observar claramente del histograma acumulado que la imagen fue ecualizada correctamente
 por la forma uniforme del histograma acumulado.
'''
# %% Parte h

'''
Binarice la imagen original en escala de grises con un umbral de selecci ́on que permita se-
parar los tonos oscuros de los claros como se muestra en la Figura 4. Para la selecci ́on del
umbral ap ́oyese en el histograma. Explique el porqu ́e del umbral seleccionado. Aplique una
binarizacion con el umbral de Otsu y compare sus resultados.
'''
'''
En base a los dos histogramas, decidi guiarme por el ecualizado que nos muestra un mínimo local
alrededor de 50, esto es para resaltar aún más los detalles y mejorar el contraste de la imagen
'''
#Binarización
umbral= 50
_, mountains_binary = cv2.threshold(mountains_gray, umbral, 255, cv2.THRESH_BINARY)

plt.figure(figsize=(6, 6))
plt.imshow(mountains_binary, cmap="gray")
plt.title('Imagen binarizada')
plt.axis('off')
plt.show()
'''
A método de comparación, si utilizamos un umbral más bajo, considerando el histograma original, alrededor 
de 20, se ve demasiado saturada y se pierde bastante detalle de la imagen
'''

umbral= 20
_, mountains_binary = cv2.threshold(mountains_gray, umbral, 255, cv2.THRESH_BINARY)

plt.figure(figsize=(6, 6))
plt.imshow(mountains_binary, cmap="gray")
plt.title('Imagen binarizada')
plt.axis('off')
plt.show()
# %%
