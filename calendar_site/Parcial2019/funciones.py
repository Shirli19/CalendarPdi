# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import math
import argparse
import cv2 as cv
from matplotlib import pyplot as plt
import scipy.stats
from sklearn.metrics import mean_squared_error
from scipy.stats import mode
import time
from skimage.morphology import medial_axis, skeletonize


###cv2.circle(img,centro_como_tupla, radio, color, -1)

#########################Lut's##########################
# Negativo de la imagen
def mapeo_negativo():
    #lookup = np.zeros(256)
    #for i in xrange(256):
    #    lookup[i] = 255 - i
    lookup=np.arange(256)
    lookup=np.fliṕ(lookup)
    return lookup

# Lut Lineal
def mapeo_ad(a, d):
    lookup = np.zeros(256)

    for i in xrange(256):
        lookup[i] = i * a + d
        lookup[i] = max(lookup[i], 0)
        lookup[i] = min(lookup[i], 255)
    return lookup

#Lut a tramos
def mapeo_tramos():
    lookup = np.zeros(256)

    for i in range(256):
        if i>0 and i<255:
            lookup[i] = i * 10
        else:
            lookup[i] = i
        lookup[i] = max(lookup[i], 0)
        lookup[i] = min(lookup[i], 255)
    return lookup

## Lut Gamma
def mapeo_potencia(exp,c):
    # Mapeo de potencia: eleva a la exp y multiplica por c
    lookup = np.zeros(256)

    for i in xrange(256):
        lookup[i] = c * pow(i, exp)
        lookup[i] = max(lookup[i], 0)
        lookup[i] = min(lookup[i], 255)
    return lookup

## Lut Gamma2
def mapeo_exp(c = 2):
    # Mapeo exponencial: hace e elevado al elemento de la LUT
    lookup = np.zeros(256)

    for i in xrange(256):
        lookup[i] = c * math.exp(i)
    lookup = 255 * lookup / lookup.max()
    return lookup

## LUT logaritmica
def mapeo_logaritmico(c):
    lookup = np.zeros(256)

    for i in xrange(256):
        lookup[i] = c * math.log(1 + i)
    lookup = 255 * lookup / lookup.max()
    return lookup

def binarizacionPorTramos(img,inf,sup):
    tablaLut=np.array(range(0,256)) #Genero un array desde 0 a 255
    tablaLut[0:inf]=0 #Trunco
    tablaLut[sup:256]=0
    return cv.LUT(img,tablaLut) #Aplico la transformación

#Sumar imagenes
def interpolar(img1, img2, alpha):
    #img3 = img1 * alpha + img2 * (1-alpha)
    #return img3
    img=cv.addWeighted(img1,alpha,img2,1-alpha,0)
    return img


#Diferencia de imagenes
def diferencia(img1, img2):
    #img3 = img1 - img2
    # img3 -= img3.min()
    # img3 *= 255/img3.max()
    #img3 += 255
    #img3 /= 2
    img3=cv.subtract(img1,img2)
    return img3

## Lista de operaciones aritmeticas
def operaciones_aritmeticas(img1, img2, tipo):
    TYPES = {
        "SUMA": cv.add(img1, img2),
        "RESTA": cv.subtract(img1, img2),
        "DIVISION": cv.divide(img1, img2),
        "MULTIPLICACION": cv.multiply(img1, img2),
    }
    return TYPES[tipo]

## Division de imagenes
def division(img1, img2):
    img2 = cv.LUT(img2, mapeo_negativo())
    img3 = img1*img2
    img3 /= img3.max()
    img3 *= 255
    return img3

def ecualizar(img):
    # Devuelve la imagen ecualizada y su histograma
    img_ec = cv.equalizeHist(img)
    hist_ec = cv.calcHist([img_ec], [0], None, [256], [0, 256])
    return img_ec, hist_ec

###################Filtros Espaciales y TDF#######################################

## FIltro Homomorfico
def filterHomomorfico(rows, cols, corte, gL, gH, order):
    """Filtro de magnitud homomorfico"""
    magnitud = np.zeros((rows, cols))
    corte *= rows
    for k in range(rows):
        for l in range(cols):
            d2 = dist([k,l], [rows//2, cols//2])
            magnitud[k,l] = (gH - gL)*(1 - np.exp(-order*(d2/(corte**2)))) + gL
    return np.fft.ifftshift(magnitud)

## Distancia Ecuclidea
def dist(a, b):
    """Distancia Euclidea"""
    return np.linalg.norm(np.array(a) - np.array(b))

def filtroMediana(img,kernel):
    return cv.medianBlur(img,kernel)

def filtroGauss(img,kernel,sigma):
    #cv2.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]])
    return cv.GaussianBlur(img,kernel,sigma)

def filtroPromedio(img,kernel):
    ker=np.ones((kernel,kernel))/(kernel*kernel)
    return cv.filter2D(img,-1,ker)

def filtroSuma1(img,kernel,medio):
    ker=-np.ones((kernel,kernel))
    ker[medio,medio]=kernel*kernel
    return cv.filter2D(img,-1,ker)

def filtroSuma0(img,kernel,medio):
    ker=-np.ones((kernel,kernel))
    ker[medio,medio]=(kernel*kernel)-1
    return cv.filter2D(img,-1,ker)

def mask_difusa(img, n):
    data = img.copy()
    kernel = np.ones((n, n), np.float32) / (n * n)
    difusa = cv.filter2D(data, ddepth=-1, kernel=kernel)
    return (((data - difusa) + 255) / 2).astype('uint8')


def alta_potencia(img, A):
    # se podria pasar el tamaño del kernel, y sumarle al kernel hp el valor de A al centro del kernel, segun teoria
    data = img.copy()
    kn_hp = [-1, -1, -1, -1, 8, -1, -1, -1, -1]  # kernel high pass, suma 0
    # kn_hp = [-1, -1, -1, -1, 9, -1, -1, -1, -1]  # kernel high pass, suma 1
    kn_hp = np.resize(kn_hp, (3, 3))
    kn_ap = np.zeros((3, 3), int)  # kernel all pass
    kn_ap[int(3/2), int(3/2)] = kn_ap[int(3/2), int(3/2)] + A
    kernel = kn_ap + kn_hp
    filtrada = cv.filter2D(data, ddepth=-1, kernel=kernel, borderType=cv.BORDER_ISOLATED)
    return filtrada


###########################Ruido#################################3
## Ruido uniforme
def generar_ruido(img, mu, sigma):
    # img: imagen a la cual se le agrega ruido
    # mu: media
    # sigma: desviacion estandar
    [alto, ancho] = img.shape
    img_re = cv.normalize(img, 0, 1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_64F)
    ruido = np.random.normal(mu, sigma, [alto, ancho]).astype('f')
    img_r = img_re + ruido
    img_r = cv.normalize(img_r, 0, 1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_64F)
    return img_r

def ruidoSalPimienta(img, s_vs_p, cantidad):
    # Parametros de entrada
    # img: imagen
    # s_vs_p: relacion de sal y pimienta (0 a 1)
    # cantidad: cantidad de ruido

    # Funcion para ensuciar una imagen con ruido sal y pimienta
    (alto, ancho) = img.shape
    # generar ruido tipo sal
    n_sal = np.ceil(cantidad * img.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(n_sal)) for i in img.shape]
    img[coords] = 255
    # generar ruido tipo pimienta
    n_pim = np.ceil(cantidad * img.size * (1.0 - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(n_pim)) for i in img.shape]
    img[coords] = 0
    return img

def ruidoGaussiano(img, mu, sigma):
    # img: imagen a la cual se le agrega ruido
    # mu: media
    # sigma: desviacion estandar
    [alto, ancho] = img.shape
    img = img.astype(np.float16) / 255
    ruido = np.random.normal(mu, sigma, [alto, ancho]).astype('f')
    img_r = img + ruido
    img_r = 255 * img_r / img_r.max()
    return img_r

def ruidoRayleigh(img, a):
    (alto, ancho) = img.shape
    img = img.astype(np.float16) / 255
    ruido = np.random.rayleigh(a, [alto, ancho]).astype('f')
    img_r = img + ruido
    img_r = 255 * img_r / img_r.max()
    return img_r

def ruidoUniforme(img, a, b):
    (alto, ancho) = img.shape
    img = img.astype(np.float16) / 255
    ruido = np.random.uniform(a, b, [alto, ancho]).astype('f')
    img_r = img + ruido
    img_r = 255 * img_r / img_r.max()
    return img_r

def ruidoExponencial(img, a):
    (alto, ancho) = img.shape
    img = img.astype(np.float16) / 255
    ruido = np.random.exponential(a, [alto, ancho]).astype('f')
    img_r = img + ruido
    img_r = 255 * img_r / img_r.max()
    return img_r

def ruidoGamma(img, a, b):
    (alto, ancho) = img.shape
    img = img.astype(np.float16) / 255
    ruido = np.random.gamma(a, b, [alto, ancho]).astype('f')
    img_r = img + ruido
    img_r = 255 * img_r / img_r.max()
    return img_r

#########################Filtros de Ruido##########################
def copiarBorde(img, top, bottom, left, right):
    return cv.copyMakeBorder(img, top, bottom, left, right,cv.BORDER_REFLECT_101)

def filtro_punto_medio(img, vecindad):
    # vecindad: pixeles alrededor del centro
    H, W = img.shape
    nueva = np.zeros((H, W))
    rellena = copiarBorde(img, vecindad, vecindad, vecindad, vecindad)
    rellena[vecindad:(H+vecindad), vecindad:(W+vecindad)] = np.copy(img)
    for i in range(vecindad, H+vecindad):
        for j in range(vecindad, W+vecindad):
            maximo = np.max(rellena[i-vecindad:i+vecindad+1, j-vecindad:j+vecindad+1], axis=(0, 1))
            minimo = np.min(rellena[i-vecindad:i+vecindad+1, j-vecindad:j+vecindad+1], axis=(0, 1))
            nueva[i-vecindad, j-vecindad] = (maximo+minimo) / 2
    nueva=nueva.astype(np.uint8)
    return nueva

#def filtro_punto_medio(img, vecindad):
#    # vecindad: pixeles alrededor del centro
#    H, W = img.shape
#    nueva = np.zeros((H, W))
#    rellena = np.zeros((H+2*vecindad, W+2*vecindad))
#    rellena[vecindad:(H+vecindad), vecindad:(W+vecindad)] = np.copy(img)
#    for i in range(vecindad, H+vecindad):
#        for j in range(vecindad, W+vecindad):
#            maximo = np.max(rellena[i-vecindad:i+vecindad+1, j-vecindad:j+vecindad+1], axis=(0, 1))
#            minimo = np.min(rellena[i-vecindad:i+vecindad+1, j-vecindad:j+vecindad+1], axis=(0, 1))
#            nueva[i-vecindad, j-vecindad] = (maximo+minimo) / 2
#    return nueva

def filtro_media_alfa(img, vecindad, descarte):
    # vecindad: pixeles alrededor del centro
    # descarte: d/2 mas bajos y d/2 mas altos
    H, W = img.shape
    nueva = np.zeros((H, W))
    rellena = copiarBorde(img, vecindad, vecindad, vecindad, vecindad)
    rellena[vecindad:(H+vecindad), vecindad:(W+vecindad)] = np.copy(img)
    mn = (2*vecindad+1)**2
    for i in range(vecindad, H+vecindad):
        for j in range(vecindad, W+vecindad):
            ordenado = np.sort(rellena[i-vecindad:i+vecindad+1, j-vecindad:j+vecindad+1].ravel())  # ravel == flatten
            media = np.mean(ordenado[((descarte+1)//2):(mn-(descarte // 2))])
            nueva[i-vecindad, j-vecindad] = media
    return nueva

#def filtro_media_alfa(img, vecindad, descarte):
#    # vecindad: pixeles alrededor del centro
#    # descarte: d/2 mas bajos y d/2 mas altos
#    H, W = img.shape
#    nueva = np.zeros((H, W))
#    rellena = np.zeros((H+2*vecindad, W+2*vecindad))
#    rellena[vecindad:(H+vecindad), vecindad:(W+vecindad)] = np.copy(img)
#    mn = (2*vecindad+1)**2
#    for i in range(vecindad, H+vecindad):
#        for j in range(vecindad, W+vecindad):
#            ordenado = np.sort(rellena[i-vecindad:i+vecindad+1, j-vecindad:j+vecindad+1].ravel())  # ravel == flatten
#            media = np.mean(ordenado[((descarte+1)//2):(mn-(descarte // 2))])
#            nueva[i-vecindad, j-vecindad] = media
#    return nueva

def filtroMediaGeometrica(img, m, n):
    (s, t) = img.shape
    for i in range(0, s-m+1):
        for j in range(0, t-n+1):
            acum = 1
            for k in range(i, i+m):
                for o in range(j, j+n):
                    acum = acum * img[k, o]
            img[i,j] = float(pow(acum, 1.0/(m*n)))
    return img

def filtroMediaContraarmonica(img, Q, s, t):
    # Si Q vale 0 da la media aritmética (Q > 0 elimina pimienta)
    # Si Q vale -1 va la media armónica (elimina sal)
    (m, n) = img.shape
    for i in range(0, m-s+1):
        for j in range(0, n-t+1):
            cont1 = 0
            cont2 = 0
            for k in range(i, i+s):
                for o in range(j, j+t):
                    cont1 = cont1 + np.power(img[k, o], Q+1)
                    cont2 = cont2 + np.power(img[k, o], Q)
            img[i, j] = cont1 / cont2
    return img

def filter_adaptative(source, varRuido):
    # Filtro adaptativo de 3 x 3
    # Se le pasa la 1 canal de la imagen
    # Devuelve la imagen filtrada

    final = source.copy().astype(np.uint8)
    rows = source.shape[0]
    cols = source.shape[1]
    members = [source[0, 0]] * 9
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            members[0] = source[y - 1, x - 1]
            members[1] = source[y, x - 1]
            members[2] = source[y + 1, x - 1]
            members[3] = source[y - 1, x]
            members[4] = source[y, x]
            members[5] = source[y + 1, x]
            members[6] = source[y - 1, x + 1]
            members[7] = source[y, x + 1]
            members[8] = source[y + 1, x + 1]

            media = np.mean(members)
            varLocal = np.std(members)

            if int(varLocal) is not 0:
                aux = (source[y, x] - (varRuido/varLocal) * (source[y, x]-media))
                aux = min(aux, 255)
                aux = max(aux, 0)
                final[y, x] = int(aux)

    final.astype(np.uint8)
    return final

def filter_adaptative2(source, varRuido, size):
    final = source.copy().astype(np.uint8)
    cols, rows = source.shape
    for y in range(size/2, rows - size/2):
        for x in range(size/2, cols - size/2):
            x1 = max(0, x-size/2)
            x2 = min(cols+1, x+size/2)
            y1 = max(0, y-size/2)
            y2 = min(rows+1, y+size/2)

            media = np.mean(source[x1:x2,y1:y2])
            varLocal = np.std(source[x1:x2,y1:y2])

            if np.isnan(varLocal):
                print("hola")

            if int(varLocal) is not 0:
                aux = (source[x,y] - (varRuido/varLocal) * (source[x,y]-media))
                aux = min(aux, 255)
                aux = max(aux, 0)
                final[x, y] = int(aux)

    final.astype(np.uint8)
    return final

def filter_midPoint3(source):
    # Filtro de punto medio de 3 x 3
    # Se le pasa la 1 canal de la imagen
    # Devuelve la imagen filtrada

    final = source.copy().astype(np.uint16)
    rows = source.shape[0]
    cols = source.shape[1]
    members = [source[0, 0]] * 9
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            members[0] = source[y - 1, x - 1]
            members[1] = source[y, x - 1]
            members[2] = source[y + 1, x - 1]
            members[3] = source[y - 1, x]
            members[4] = source[y, x]
            members[5] = source[y + 1, x]
            members[6] = source[y - 1, x + 1]
            members[7] = source[y, x + 1]
            members[8] = source[y + 1, x + 1]

            a = max(members)
            b = min(members)
            c = (a.astype(int) + b.astype(int)) / 2
            final[y, x] = c

    final.astype(np.uint8)
    return final
#########################Color######################################
def rodajas(img):
    [altura, ancho] = img.shape

    auximg = img*0
    imgs = np.zeros([altura, ancho, 8])

    for i in xrange(8):
        mask = 2**i
        cv.bitwise_and(img, mask, auximg)
        imgs[:, :, i] = np.copy(auximg).astype("uint8")

    return imgs

def pasarHSV(img):
    return cv.cvtColor(img, cv.COLOR_BGR2HSV)

def pasarRGB(img):
    return cv.cvtColor(img, cv.COLOR_HSV2BGR)

def pasarRGBtoGray(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def pasarBinario(img,rango):
    ret,th = cv.threshold(img,rango,255, 0)
    return th

def colorComplementario(imgOr,rgb=1):
    img=np.copy(imgOr)
    if (rgb==1):
        img=pasarHSV(img)
    img[:,:,0]=np.abs(img[:,:,0]-179)
    img[:,:,2]=cv.bitwise_not(img[:,:,2])
    if(rgb==1):
        img=pasarRGB(img)
    return img

def ecualizar_img_color(imgOr,rgb=1):
    img=np.copy(imgOr)
    if (rgb==1):
        img=pasarHSV(img)
    img[:,:,2] = cv.equalizeHist(img[:,:,2])
    if(rgb==1):
        img=pasarRGB(img)
    return img

def mascaraRango(img,bajo,alto):
    ##bajo=np.array([Hb,Sb,Vb], dtype = "uint8")
    ##alto=np.array([Ha,Sa,Va], dtype = "uint8")
    return cv.inRange(img,bajo,alto)

def trackbarMascaraRango(imgOr):
    imgHSV=pasarHSV(imgOr)
    H=100
    S=150
    V=20
    rangoH=10
    rangoS=80
    rangoV=80
    H_viejo=100
    S_viejo=150
    V_viejo=20
    rangoH_viejo=10
    rangoS_viejo=80
    rangoV_viejo=80
    #Creo la imagen donde insertar los trackbars
    cv.namedWindow('image')
    # Creo trackbars para recibir un color
    cv.createTrackbar('H','image',0,255,nothing)
    cv.createTrackbar('S','image',0,255,nothing)
    cv.createTrackbar('V','image',0,255,nothing)
    cv.createTrackbar('rangoH','image',0,125,nothing)
    cv.createTrackbar('rangoS','image',0,125,nothing)
    cv.createTrackbar('rangoV','image',0,125,nothing)
    while(1):
        cv.imshow('image',imgOr)
        #Para cerrar presionando escape
        #Agregar esto en cualquier loop que contenga un imshow
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break
        H=cv.getTrackbarPos('H','image')
        S=cv.getTrackbarPos('S','image')
        V=cv.getTrackbarPos('V','image')
        rangoH=cv.getTrackbarPos('rangoH','image')
        rangoS=cv.getTrackbarPos('rangoS','image')
        rangoV=cv.getTrackbarPos('rangoV','image')
        #Obtengo los colores de las posiciones de la barra
        if (H_viejo!=H or S_viejo!=S or V_viejo!=V
                or rangoH_viejo!=rangoH or rangoS_viejo!=rangoS or rangoV_viejo!=rangoV):
            H_viejo=H
            S_viejo=S
            V_viejo=V
            rangoH_viejo=rangoH
            rangoS_viejo=rangoS
            rangoV_viejo=rangoV
            H=cv.getTrackbarPos('H','image')
            S=cv.getTrackbarPos('S','image')
            V=cv.getTrackbarPos('V','image')
            rangoH=cv.getTrackbarPos('rangoH','image')
            rangoS=cv.getTrackbarPos('rangoS','image')
            rangoV=cv.getTrackbarPos('rangoV','image')
            bajo=np.array([H-rangoH,S-rangoS,V-rangoV], dtype = "uint8")
            alto=np.array([H+rangoH,S+rangoS,V+rangoV], dtype = "uint8")
            mask=mascaraRango(imgHSV,bajo,alto)
            cv.imshow('image1',mask)
            #cv.waitKey()

    cv.destroyAllWindows()
    return mask



######################################General##########################################

# A partir de una linea horizontal (altura), calcula en donde se ubican las líneas verticales que separan blanco de negro.
# Ejemplo: Si la imagen es un blister de medicamentos en donde las pastillas son blancas y el fondo negro, al pasarle una
# altura = 55, se fijará en esa línea horizontal, en qué posiciones de X están las pastillas (blanco)
def obtenerPosDesdeAltura(img,altura):
    # perfil de intensidad para esa altura
    perfilh = img[altura, :]
    ceros = []
    for index, valor in enumerate(perfilh):
        if valor == 0:
            ceros.append(index)
    ceros.append(img.shape[1])
    centros = []

    # calculamos los centros entre dos espacios negros
    for i, val in enumerate(ceros):
        if i == 0:
            continue
        else:
            if ceros[i] != (ceros[i - 1] + 1):
                centros.append((ceros[i] + ceros[i - 1]) / 2)

    return centros

def redimensionar(img,escalaVer,escalaHor):
    img = cv.resize(img, (0, 0), None, escalaVer, escalaHor)
    return img

def deteccionCirculos(img,dp=1.2,minDist=100):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # detect circles in the image
    #image: 8-bit, single channel image. If working with a color image, convert to grayscale first.
    #method: Defines the method to detect circles in images. Currently, the only implemented method 
    #       is cv2.HOUGH_GRADIENT, which corresponds to the Yuen et al. paper.
    #dp: This parameter is the inverse ratio of the accumulator resolution to the image resolution
    #        (see Yuen et al. for more details). Essentially, the larger the dp gets, the smaller 
    #           the accumulator array gets.
    #minDist: Minimum distance between the center (x, y) coordinates of detected circles. 
    #       If the minDist is too small, multiple circles in the same neighborhood as the original 
    #       may be (falsely) detected. If the minDist is too large, then some circles may not be detected at all.
    #param1: Gradient value used to handle edge detection in the Yuen et al. method.
    #param2: Accumulator threshold value for the cv2.HOUGH_GRADIENT method. The smaller the threshold is,
    #        the more circles will be detected (including false circles). The larger the threshold is, 
    #           the more circles will potentially be returned.
    #minRadius: Minimum size of the radius (in pixels).
    #maxRadius: Maximum size of the radius (in pixels).

    circles = cv.HoughCircles(gray, cv.cv.CV_HOUGH_GRADIENT, dp, minDist)
    
    # ensure at least some circles were found
    if circles is not None:
    	# convert the (x, y) coordinates and radius of the circles to integers
    	circles = np.round(circles[0, :]).astype("int")
    
    	# loop over the (x, y) coordinates and radius of the circles
    	for (x, y, r) in circles:
    		# draw the circle in the output image, then draw a rectangle
    		# corresponding to the center of the circle
    		cv2.circle(output, (x, y), r, (0, 255, 0), 4)
    		cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)


# Elimino los valores que son sucesivos ej:
# Si delFL es True elimino primer y ultimo elemento: [1, 2, 3, 10, 12, 13, 14, 15] ->
# Si delFL es False: [1, 2, 3, 10, 12, 13, 14, 15] -> [1, 3, 10, 12, 15]
def eliminar_contiguos(lista, delFL=False):
    lista2 = []
    # hasta el penultimo elemento
    for i, valor in enumerate(lista[:-1]):
        if lista[i] + 1 != lista[i + 1]:
            lista2.append(lista[i])
            lista2.append(lista[i + 1])

    if not delFL:
        lista2.insert(0, lista[0])
        lista2.append(lista[-1])
    return sorted(list(set(lista2)))

## Error cuadratico medio
def ECM(img1, img2):
    # Error cuadrático medio (Root mean square)
    img = img1 - img2
    return math.sqrt(sum(n * n for n in img.flatten()) / len(img.flatten()))

## Capturar punto en una imagen
def capturar_punto(event, x, y, flags, param):
    global mouseX,mouseY, imgn
    if event == cv.EVENT_LBUTTONDOWN:
            print("posicion elegida: (",x,",",y,"), presione 'a' o 'c' para confirmar.")
            mouseX,mouseY = x,y

## Eligir un punto en la imagen
def elegir_punto(image):
    global imgn
    imgn = image
    cv.namedWindow("image")
    cv.setMouseCallback("image", capturar_punto)

    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv.imshow("image", imgn)
        key = cv.waitKey(20) & 0xFF

        if key == 27:
            break
        elif key == ord('a'):
            return [mouseX, mouseY]
        elif key == ord('c'):
            return [mouseX, mouseY]

def detectar_rectangulos(img):
    # solo para cuadrados perfectos y aliniados
    ret, img_bin = cv.threshold(img, thresh=127, maxval=255, type=cv.THRESH_BINARY_INV)
    contours, hierarchy = cv.findContours(img_bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    rectangulos = []
    for contour in contours:
        (x, y, w, h) = cv.boundingRect(contour)
        rectangulos.append([x, y, w, h])
    return rectangulos


def detectar_circulos(img):
    centros = radios = []
    contours = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    contours.sort(key=lambda x: cv.boundingRect(x)[0])

    # mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    for c in contours:
        (x, y), r = cv.minEnclosingCircle(c)
        center = (int(x), int(y))
        r = int(r)
        if 6 <= r <= 50:
            # cv.circle(mask, center, r, (0, 255, 0), 2)
            centros.append(center)
            radios.append(r)
    return centros, radios

def obtenerRoi(img):
    # Select ROI
    r = cv.selectROI(img)
    # Crop image
    imCrop = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    # Display cropped image
    cv.imshow("Image", imCrop)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return imCrop

def obtenerRoiHSV(img):
    # Select ROI
    r = cv.selectROI(img)
    # Crop image
    imCrop = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    #imCrop=ecualizar_img_color(imCrop,rgb=1)
    # Display cropped image
    cv.imshow("Image", imCrop)
    cv.waitKey(0)
    cv.destroyAllWindows()
    imCropHSV=pasarHSV(imCrop)
    
    histograma(imCropHSV[:,:,0])
    histograma(imCropHSV[:,:,1])
    histograma(imCropHSV[:,:,2])
    return imCrop

## EL practico del blister Sirve para obtener posiciones alienadas
## Te devuleve el centro del recuadro vacio
def verTabla(img):
    ## Otengo las dimensiones de la imagen
    fila,columna = img.shape
    
    ##Recorro de forma vertical para obtener las filas de pastillas
    indVer=[] #Aca almaceno los limites de las filas
    i=0
    while (i <fila):
        j=0
        if (np.count_nonzero(img[i,:] == 255) > 0):
            indVer.append(i)
            j=i
            while (np.count_nonzero(img[j,:] == 255) > 0):
                j=j+1

            indVer.append(j)
        if (j!=0):
            i=j
        else:
            i=i+1

    ##Recorro de forma horizontal para obtener las columnas de las pastillas
    indHor=[] #Almaceno los indices de las columnas
    i=0
    while (i <columna):
        j=0
        if (np.count_nonzero(img[:,i] == 255) > 0):
            indHor.append(i)
            j=i
            while (np.count_nonzero(img[:,j] == 255) > 0):
                j=j+1

            indHor.append(j)
        if (j!=0):
            i=j
        else:
            i=i+1
    ##Veo que lugares estan vacios
    vacias=[]#Almaceno los pares xy que estan vacios
    if (len(indHor)!=0 or len(indVer)!=0):  
        k=0
        while (k < len(indHor)-1):
            j=0
            print(k)
            while (j < len(indVer)-1):
                if (np.count_nonzero(img[int((indVer[j]+indVer[j+1])/2),indHor[k]:indHor[k+1]] == 255)>5):
                    vacias.append((int((indVer[j]+indVer[j+1])/2),int((indHor[k]+indHor[k+1])/2)))
                j=j+2
            k=k+2

    return vacias
########################################Graficos##############################
def graficar(img,maximo,minimo,mapa,titulo=''):
    ventana = plt.figure()
    # ventana.canvas.set_window_title(titulo)
    plt.axis("off")
    plt.imshow(img, vmax=maximo, vmin=minimo, cmap=mapa)
    plt.title(titulo)


def histograma(img, title="Histograma"):
    histB=cv.calcHist([img],[0],None,[256],[0,256])
    plt.plot(histB)
    plt.show()


#############################Segmentacion###############################################

def bordes_roberts(img, umbral1, umbral2):
    roberts_cross_v = np.array([[0, 0, 0],
                                [0, 1, 0],
                                [0, 0, -1]])

    roberts_cross_h = np.array([[0, 0, 0],
                                [0, 0, 1],
                                [0, -1, 0]])

    horizontal = cv.filter2D(img, -1, roberts_cross_h).astype(np.uint16)
    vertical = cv.filter2D(img, -1, roberts_cross_v).astype(np.uint16)

    output_image = np.sqrt(np.square(horizontal) + np.square(vertical)).astype(np.uint8)

    ret, output_image = cv.threshold(output_image, umbral1, umbral2, cv.THRESH_BINARY)
    return output_image


def bordes_prewitt(img, umbral1, umbral2):
    kernelx = np.array([[1, 1, 1],
                        [0, 0, 0],
                        [-1, -1, -1]])

    kernely = np.array([[-1, 0, 1],
                        [-1, 0, 1],
                        [-1, 0, 1]])
    img_prewittx = cv.filter2D(img, -1, kernelx).astype(np.uint16)
    img_prewitty = cv.filter2D(img, -1, kernely).astype(np.uint16)

    output_image = np.sqrt(np.square(img_prewittx) + np.square(img_prewitty)).astype(np.uint8)
    ret, output_image = cv.threshold(output_image, umbral1, umbral2, cv.THRESH_BINARY)
    return output_image


def bordes_sobel(img, umbral1, umbral2):
    img_sobelx = cv.Sobel(img, cv.CV_16U, 1, 0, ksize=3)
    img_sobely = cv.Sobel(img, cv.CV_16U, 0, 1, ksize=3)
    # output_image = img_sobelx + img_sobely

    output_image = np.sqrt(np.square(img_sobelx) + np.square(img_sobely)).astype(np.uint8)
    ret, output_image = cv.threshold(output_image, umbral1, umbral2, cv.THRESH_BINARY)

    return output_image


def bordes_laplaciano(img, umbral1, umbral2):
    output_image = cv.Laplacian(img, cv.CV_8U)
    ret, output_image = cv.threshold(output_image, umbral1, umbral2, cv.THRESH_BINARY)

    return output_image


def bordes_LoG(img, umbral1, umbral2):
    kernel = np.array([[0, 0, -1, 0, 0],
              [0, -1, -2, -1, 0],
              [-1, -2, 16, -2, -1],
              [0, -1, -2, -1, 0],
              [0, 0, -1, 0, 0]])
    output_image = cv.filter2D(img, -1, kernel)
    ret, output_image = cv.threshold(output_image, umbral1, umbral2, cv.THRESH_BINARY)

    return output_image

def bordes_canny(img,umbral1,umbral2):
    return cv.Canny(img,umbral1,umbral2)

def detectarBordes(img, umbral1, umbral2, tipo="ro"):
    if(tipo=="ro"):
        result=bordes_roberts(img, umbral1, umbral2)
    elif(tipo=="pr"):
        result=bordes_prewitt(img, umbral1, umbral2)
    elif(tipo=="so"):
        result=bordes_sobel(img, umbral1, umbral2)
    elif(tipo=="la"):
        result=bordes_laplaciano(img, umbral1, umbral2)
    elif(tipo=="lo"):
        result=bordes_LoG(img, umbral1, umbral2)
    elif(tipo=="ca"):
        result=bordes_canny(img,umbral1,umbral2)
    return result

def segmentarColorHS(img, subimg, varF=30):
    rows,cols = img.shape[0:2]
    # Centroides de H y S
    medH = np.mean(subimg[:,:,0])
    medS = np.mean(subimg[:,:,1])

    # Calculo los radios de  H y S como la varianza de cada componente
    rH = np.std(subimg[:,:,0]) * varF
    rS = np.std(subimg[:,:,1]) * varF

    H = img[:, :, 0]
    S = img[:, :, 1]

    # Recorro imgCompleta y veo si esta dentro de los circulos en cada canal
    umbColor = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            if H[i, j] <= medH + rH and H[i, j] >= medH - rH:
                if S[i, j] <= medS + rS and S[i, j] >= medS - rS:
                    umbColor[i, j] = 255
    umbColor = 255 - umbColor
    return umbColor

def segmentarColorSV(img, subimg, var1=15, var2=10):
    rows,cols = img.shape[0:2]
    # Centroides de H y S
    medS = np.mean(subimg[:,:,1])
    medV = np.mean(subimg[:,:,2])

    # Calculo los radios de  H y S como la varianza de cada componente
    rS = np.std(subimg[:,:,1]) * var1
    rV = np.std(subimg[:,:,2]) * var2

    S = img[:, :, 1]
    V = img[:, :, 2]

    # Recorro imgCompleta y veo si esta dentro de los circulos en cada canal
    umbColor = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            if S[i, j] <= medS + rS and S[i, j] >= medS - rS:
                if V[i, j] <= medV + rV and V[i, j] >= medV - rV:
                    umbColor[i, j] = 255
    umbColor = 255 - umbColor
    return umbColor

def componentesConectadas(img):
    # You need to choose 4 or 8 for connectivity type
    connectivity = 4
    # Perform the operation
    output = cv.connectedComponentsWithStats(img, connectivity, cv.CV_32S)
    # Get the results
    # The first cell is the number of labels
    num_labels = output[0]
    # The second cell is the label matrix
    labels = output[1] #Imagen donde cada componente conectada tiene un valor de intensidad diferente
    # The third cell is the stat matrix
    stats = output[2]
    # The fourth cell is the centroid matrix
    centroids = output[3] #EL CENTROIDE[0,0] ES EL DEL FONDO!!!!
    print ('Cantidad de elementos = ', num_labels - 1)
    cont=num_labels-1

    #PASO 4 (Dibujar circulos sobre las rosas)
    imgOR_2=np.copy(img)

    radius = 12
    centros=[]
    for i in range(num_labels-1):
        center = int(centroids[i+1, 0]), int(centroids[i+1, 1])
        centros.append(center)
        cv.circle(imgOR_2, center, radius, (0, 255, 0), 2)
    return imgOR_2,cont,centros



# Me canse de copiar y pegar siempre lo mismo, asi que va un intento de generalizar:
#   -source: imagen de bordes a la cual se aplica la transformada de Hough
#   -n: Cantidad de lineas a dibujar. Por defecto es 1. Para dibujar todas las lineas pasar -1
#   -rhosensv: Precision de rho en pixeles
#   -phisensv: Precision de phi en radianes
#   -hardness: Solo las lineas con una cantidad mayor a estas seran devueltas
#   -plotimg: Imagen sobre la cual graficar las lineas
#   -color: Color de las lineas a graficar
def hough(source, n = 1, rhosensv = 1, phisensv = np.pi / 180, hardness = 90, plotimg = None, color = (0, 0, 255)):
    rows, cols = source.shape
    lines = cv.HoughLines(source, rhosensv,phisensv, hardness)
    # Gráfica de la primera línea detectada (la más fuerte)
    if n == -1:
        n = lines.shape[0]
    points = []
    for line in lines[0:n]:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = max(int(x0 + 1000 * (-b)), 0)
            y1 = max(int(y0 + 1000 * (a)), 0)
            x2 = min(int(x0 - 1000 * (-b)), cols)
            y2 = min(int(y0 - 1000 * (a)), rows)
            points.append([(x1, y1), (x2, y2)])
            if plotimg is not None:
                cv.line(plotimg, (x1, y1), (x2, y2), color, 2)
    return lines, points

# EL siguiente es el hough del Seba Fenoglio
def houghSF(data,umbral1=100,umbral2=150,thresh=80,threshold= 10, minLineLength = 20,maxLineGap = 10):
    # Data imagen en escala de grises
    # umbral1 y umbral2 es para canny
    # thresh:Numero minimo de intersecciones
    # threshold: Minimo numero de intersecciones para houghLinesP
    # minLineLength: Cantidad minima de pixeles para ser considerados linea
    # maxLineGap: Brecha maxima entre 2 puntos para considerarlos en una liena
    img= data[:,:]/255
    imgP= img.copy()
    H,W=img.shape

    #1. Calcular el gradiente de la imagen 
    edges= cv.Canny(data,umbral1,umbral2)
    #2. umbralizar el resultado. Hecho en Canny

    #3. HoughLines
    lines= cv.HoughLines(edges,1,np.pi/180,thresh)
    #agregar lineas Hough
    for rho,theta in lines[:,0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv.line(img,(x1,y1),(x2,y2),255,2)

    #3. HoughLinesP
    lines = cv.HoughLinesP(edges,1,np.pi/180,threshold,minLineLength,maxLineGap)
    for x1,y1,x2,y2 in lines[:,0]:
        cv.line(imgP,(x1,y1),(x2,y2),255,2)

    pegadas= np.concatenate((edges,img,imgP),axis=1)
    return pegadas

####################Funciones de Hough por separado####################################################
##Theta = 0 son lineas verticales
## theta = 1.5707964 son lineas horizontales

def houghComun(imgOr,bin=1,umbral1=100,umbral2=150,thresh=80):
    imgC=np.copy(imgOr)
    if (bin==1):
        imgG= cv.cvtColor(imgC,cv.COLOR_BGR2GRAY)
    else:
        imgG=np.copy(imgOr)
    img= imgG[:,:]/255
    imgP= img.copy()
    H,W=img.shape

    #1. Calcular el gradiente de la imagen 
    edges= cv.Canny(imgG,umbral1,umbral2)
    #2. umbralizar el resultado. Hecho en Canny

    #3. HoughLines
    lines= cv.HoughLines(edges,1,np.pi/180,thresh)
    #agregar lineas Hough
    for rho,theta in lines[:,0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv.line(img,(x1,y1),(x2,y2),255,2)
    return img,lines

def houghProb(imgOr,bin=1,umbral1=100,umbral2=150,threshold= 10, minLineLength = 20,maxLineGap = 10):
    imgC=np.copy(imgOr)
    if (bin==1):
        imgG= cv.cvtColor(imgC,cv.COLOR_BGR2GRAY)
    else:
        imgG=np.copy(imgOr)
    img= imgG[:,:]/255
    imgP= img.copy()
    H,W=img.shape

    #1. Calcular el gradiente de la imagen 
    edges= cv.Canny(imgG,umbral1,umbral2)
    #3. HoughLinesP
    lines = cv.HoughLinesP(edges,1,np.pi/180,threshold,minLineLength,maxLineGap)
    for x1,y1,x2,y2 in lines[:,0]:
        cv.line(imgP,(x1,y1),(x2,y2),255,2)

    return imgP,lines

### Funciones trackbar para los algoritmos de hough
def nothing(x):
    pass

def trackbarHoughProb(imgOr,bin):
    treshold_viejo=80
    treshold=80
    umbral1_viejo=100
    umbral1=100
    umbral2_viejo=150
    umbral2=150
    minLineLength_viejo=20
    minLineLength=20
    maxLineGap_viejo=10
    maxLineGap=10
    #Creo la imagen donde insertar los trackbars
    cv.namedWindow('image')
    # Creo trackbars para recibir un color
    cv.createTrackbar('umbral1','image',0,255,nothing)
    cv.createTrackbar('umbral2','image',0,255,nothing)
    cv.createTrackbar('treshold','image',0,255,nothing)
    cv.createTrackbar('minLineLength','image',0,255,nothing)
    cv.createTrackbar('maxLineGap','image',0,255,nothing)
    while(1):
        cv.imshow('image',imgOr)
        #Para cerrar presionando escape
        #Agregar esto en cualquier loop que contenga un imshow
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break
        treshold=cv.getTrackbarPos('treshold','image')
        umbral1= cv.getTrackbarPos('umbral1','image')
        umbral2= cv.getTrackbarPos('umbral2','image')
        minLineLength= cv.getTrackbarPos('minLineLength','image')
        maxLineGap= cv.getTrackbarPos('maxLineGap','image')
        #Obtengo los colores de las posiciones de la barra
        if (treshold_viejo != treshold or umbral1_viejo!=umbral1 or umbral2_viejo!=umbral2
                or minLineLength_viejo!=minLineLength or maxLineGap_viejo!=maxLineGap):
            treshold_viejo=treshold
            umbral1_viejo=umbral1
            umbral2_viejo=umbral2
            minLineLength_viejo=minLineLength
            maxLineGap_viejo=maxLineGap
            treshold=cv.getTrackbarPos('treshold','image')
            umbral1= cv.getTrackbarPos('umbral1','image')
            umbral2= cv.getTrackbarPos('umbral2','image')
            minLineLength= cv.getTrackbarPos('minLineLength','image')
            maxLineGap= cv.getTrackbarPos('maxLineGap','image')
            img,lines=houghProb(imgOr,bin,umbral1,umbral2,treshold, minLineLength,maxLineGap)
            cv.imshow('image1',img)
            #cv.waitKey()

    cv.destroyAllWindows()
    return lines

def trackbarHoug(imgOr,bin):
    treshold_viejo=80
    treshold=80
    umbral1_viejo=100
    umbral1=100
    umbral2_viejo=150
    umbral2=150
    #Creo la imagen donde insertar los trackbars
    cv.namedWindow('image')
    # Creo trackbars para recibir un color
    cv.createTrackbar('umbral1','image',0,255,nothing)
    cv.createTrackbar('umbral2','image',0,255,nothing)
    cv.createTrackbar('treshold','image',0,500,nothing)
    while(1):
        cv.imshow('image',imgOr)
        #Para cerrar presionando escape
        #Agregar esto en cualquier loop que contenga un imshow
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break
        treshold=cv.getTrackbarPos('treshold','image')
        umbral1= cv.getTrackbarPos('umbral1','image')
        umbral2= cv.getTrackbarPos('umbral2','image')
        #Obtengo los colores de las posiciones de la barra
        if (treshold_viejo != treshold or umbral1_viejo!=umbral1 or umbral2_viejo!=umbral2):
            treshold_viejo=treshold
            umbral1_viejo=umbral1
            umbral2_viejo=umbral2
            treshold=cv.getTrackbarPos('treshold','image')
            umbral1= cv.getTrackbarPos('umbral1','image')
            umbral2= cv.getTrackbarPos('umbral2','image')
            img,lines=houghComun(imgOr,bin,umbral1,umbral2,treshold)
            cv.imshow('image1',img)
            #cv.waitKey()

    cv.destroyAllWindows()
    return lines,img


############## Codigos para rellenar con semillas mezcla de Franco con uno de internet###################
# Invoca a la función rellenar con un monton de cosas que no se que hacen
def rellenar(img,point,tolerancia):
    src = np.copy(img)

    connectivity = 4
    flags = connectivity
    flags |= cv.FLOODFILL_FIXED_RANGE

    cv.floodFill(src, None, point, (0, 255, 255), (tolerancia,) * 3, (tolerancia,) * 3, flags)
    cv.imshow('relleno', src)


click=[]

def capturar_puntoClick(event, x, y, flags, param):
    global imgn, click, mouseX,mouseY
    if event == cv.EVENT_LBUTTONDOWN:
            print("posicion elegida: (",x,",",y,"), presione 'a' o 'c' para confirmar.")
            mouseX,mouseY = x,y
            click.append((mouseX,mouseY))

#Una funcioón de trackbar para la tolerancia
def trackbar_value(value):
    global tolerancia
    tolerancia = value
#Funcion global
def rellenarConSemillas(img):
    global imgn
    global tolerancia
    tolerancia=1
    imgn=img
    cv.namedWindow("image")
    cv.setMouseCallback("image", capturar_punto)
    cv.namedWindow("image")
    cv.createTrackbar('tolerancia', "image", 1, 100, trackbar_value)

    while(1):
        # display the image and wait for a keypress
        cv.imshow("image", imgn)
        key = cv.waitKey(20) & 0xFF
        if key == 27:
            break
        elif key == ord('a'):
            rellenar(img,(mouseX,mouseY),tolerancia)
        elif key == ord('c'):
            rellenar(img,(mouseX,mouseY),tolerancia)
    cv.destroyAllWindows()
##########################################################################################################

##############################################Morfologia##################################################

def elementoEstructurante(tipo,size=3):
    if (tipo=="rec"):
        ele=cv.getStructuringElement(cv.MORPH_RECT,(size,size))
    elif (tipo=="cir"):
        ele=cv.getStructuringElement(cv.MORPH_ELLIPSE,(size,size))
    elif (tipo=="cruz"):
        ele=cv.getStructuringElement(cv.MORPH_CROSS,(size,size))
    else:
        ele=np.zeros((size,size))
    return ele
    
def erosionar(img,element,it):
    return cv.erode(img,element, iterations = it)

def dilatacion(img,element,it):
    return cv.dilate(img,element, iterations = it)

def apertura(img, elemen):
    #C = cv.dilate(A,B, iterations = it)
    ## func.graficar(C, 255, 0, 'gray', 'Dilatacion')
    #D = cv.erode(C,B, iterations = it)
    return cv.morphologyEx(img, cv.MORPH_OPEN, elemen)

def cierre(img, elemen):
    #C = cv.erode(A,B, iterations = it)
    ## func.graficar(C, 255, 0, 'gray', 'Erosion')
    #D = cv.dilate(C,B, iterations = it)
    return cv.morphologyEx(img, cv.MORPH_CLOSE, elemen)

def gradienteMorfologico(img, elemen):
    #It is the difference between dilation and erosion of an image.
    #The result will look like the outline of the object.
    return cv.morphologyEx(img, cv.MORPH_GRADIENT, elemen)

def morfologiaTopHat(img, elemen):
    #It is the difference between input image and Opening of the image.
    return cv.morphologyEx(img, cv.MORPH_TOPHAT, elemen)

def morfologiaBlackHat(img, elemen):
    #It is the difference between the closing of the input image and input image.
    return cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)

## Convex hull
def convexHull(img):
    # Imagen binaria se le debe pasar
    # Finding contours for the thresholded image
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # create hull array for convex hull points
    hull = []
    
    # calculate points for each contour
    for i in range(len(contours)):
        # creating convex hull object for each contour
        hull.append(cv.convexHull(contours[i], False))

    # create an empty black image
    drawing = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    
    # draw contours and hull points
    for i in range(len(contours)):
        color_contours = (0, 255, 0) # green - color for contours
        color = (255, 0, 0) # blue - color for convex hull
        # draw ith contour
        cv.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
        # draw ith convex hull object
        cv.drawContours(drawing, hull, i, color, 1, 8)
    return drawing
    
## Encontrar el esqueleto
def esquleto(imgBin):
    #Es necesatio que la imagen sea binaria con el fondo negro y los
    #objetos de interez blancos
    # Generate the data
    data=np.copy(imgBin)
    data=data / 255
    # Compute the medial axis (skeleton) and the distance transform
    skel, distance = medial_axis(data, return_distance=True)

    # Compare with other skeletonization algorithms
    skeleton = skeletonize(data)
    skeleton_lee = skeletonize(data)

    # Distance to the background for pixels of the skeleton
    dist_on_skel = distance * skel
    return dist_on_skel