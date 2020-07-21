from Parcial2019 import funciones
from Parcial2019 import pdifun
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def recortarImagen(imagen):
    h, w= imagen.shape
   # print(h,w)
    inicioColumna = 0
    finColumna = w
    inicioFila = 0
    finFila = h

    for i in range(h):
        #print(max(imagen[i,:]))
        if (max(imagen[i,:]) > 250):
            inicioFila = i
            break
    i = h-1
    while i>inicioFila:
       # print(max(imagen[i,:]))
        if (max(imagen[i,:]) > 250):
            finFila = i
            break
        i=i-1
        
    for i in range(w):
        if (max(imagen[:,i]) > 250):
            inicioColumna = i
            break
    i = w-1
    while i>inicioColumna:
        #print(max(imagen[:,i]))
        if (max(imagen[:,i]) > 250):
            finColumna = i
            break
        i=i-1
    #print(inicioFila, finFila, inicioColumna, finColumna)
    return inicioFila,finFila,inicioColumna,finColumna 

#nombreImagen = "train01.png" ## 24 - 4
#nombreImagen = "train02.png"  ## 19 - 4
#nombreImagen = "train03.png" ## 19 - 4
#nombreImagen = "train04.png" ## 24 - 4
#nombreImagen = "test01_19p_8L_3032.png" ## 19 - 4
#nombreImagen = "test02_24p_11L__2333.png" ## 24 - 4
#nombreImagen = "test03_19p_15L_5244.png" ## 19 - 4
#nombreImagen = "test04_19p_15L_5244.png" ## 19 - 4
#nombreImagen = "test05_19p_8L_332.png" ## 19 - 3

def resolucion(nombreImagen): 
    imagen = cv.imread('./Parcial2019/'+nombreImagen)

    #cv.imshow("Edificio", imagen)
    #cv.waitKey(0)

    #imagen_gray = funciones.pasarRGBtoGray(imagen)

    #funciones.obtenerRoiHSV(imagen)

    imagen_hsv = funciones.pasarHSV(imagen)

    bajo = np.array([100,15,240], dtype = "uint8")
    alto = np.array([120,20,245], dtype = "uint8")
    mask_edificio = funciones.mascaraRango(imagen_hsv, bajo, alto)
    ele=cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    mask_edificio = funciones.erosionar(mask_edificio, ele, 1)
    mask_edificio = funciones.filtroMediana(mask_edificio, 5)
    #cv.imshow("mask edificio", mask_edificio)
    #cv.waitKey(0)

    inicioFila,finFila,inicioColumna,finColumna = recortarImagen(mask_edificio)
    imagen_recortada = imagen[inicioFila:finFila, inicioColumna:finColumna]

    #cv.imshow("Edificio Recortado", imagen_recortada)
    #cv.waitKey(0)

    #plt.figure()
    #plt.imshow(imagen_recortada)
    #plt.show()

    ### CONTAR PISOS

    imagen_recortada_hsv = imagen_hsv[inicioFila:finFila, inicioColumna:finColumna]
    bajo = np.array([100,30,150], dtype = "uint8")
    alto = np.array([110,80,255], dtype = "uint8")
    mask_pisos = funciones.mascaraRango(imagen_recortada_hsv, bajo, alto)
    ele=cv.getStructuringElement(cv.MORPH_RECT,(7,1))
    mask_pisos = funciones.erosionar(mask_pisos, ele, 10)
    mask_pisos = funciones.filtroMediana(mask_pisos, 3)
    #cv.imshow("mask pisos", mask_pisos)
    #cv.waitKey(0)

    imagenOR_2, cont, centros_pisos = funciones.componentesConectadas(mask_pisos)

    ## desde el piso 1 hasta el ultimo (sin planta baja ni sum)
    cantidad_pisos = cont

    ### EDIFICIOS POR PISOS

    bajo = np.array([155,45,150], dtype = "uint8")
    alto = np.array([165,60,170], dtype = "uint8")
    mask_edificios = funciones.mascaraRango(imagen_recortada_hsv, bajo, alto)
    ele=cv.getStructuringElement(cv.MORPH_RECT,(1,7))
    mask_edificios = funciones.dilatacion(mask_edificios, ele, 5)
    #mask_edificios = funciones.filtroMediana(mask_edificios, 5)
    #cv.imshow("mask edificios", mask_edificios)
    #cv.waitKey(0)

    """
    h,w = mask_edificio
    division = 0
    for i in range(w):
        if(mask_edificio[int(h/2), i] > 250):
            division = division + 1
            i = i + 5
    """
    imagenOR_2, cont, centros_division = funciones.componentesConectadas(mask_edificios)

    ## desde el piso 1 hasta el ultimo (sin planta baja ni sum)
    division_edificio = cont
    cantidad_edificios = (division_edificio+1) * (cantidad_pisos-1)


    imagen_recortada_hsv = imagen_hsv[inicioFila:finFila, inicioColumna:finColumna]
    bajo = np.array([25,55,170], dtype = "uint8")
    alto = np.array([35,105,245], dtype = "uint8")
    mask_edificioI = funciones.mascaraRango(imagen_recortada_hsv, bajo, alto)
    #ele=cv.getStructuringElement(cv.MORPH_RECT,(7,1))
    #mask_edificioI = funciones.erosionar(mask_edificioI, ele, 10)
    #mask_edificioI = funciones.filtroMediana(mask_edificioI, 3)
    #cv.imshow("mask edificioI", mask_edificioI)
    #cv.waitKey(0)

    imagenOR_2, cont, centros = funciones.componentesConectadas(mask_edificioI)

    ## desde el piso 1 hasta el ultimo (sin planta baja ni sum)
    cantidad_edificioI = cont
    cantidad_edificioA = cantidad_edificios - cantidad_edificioI

    #sorted(centros_pisos[:]1])
    #print("Centro pisos {}".format(centros_pisos[:][:][1]))

    j=0
    pisos_x = np.zeros((cantidad_pisos, 1))
    for i in centros_pisos[:]:
        _, pisos_x[j] = i
        j = j+1

    sorted(pisos_x)
    j=0
    edificios_y = np.zeros((division_edificio, 1))
    for i in centros_division[:]:
        edificios_y[j], _ = i
        j = j+1
    sorted(edificios_y)
    #print("Centro pisos {}".format(pisos_x[0][0]))


    h, w = mask_edificioI.shape
    print(h,w)
    i=0
    piso=[]
    columna=[]
    print(centros)
    for c in centros:
        #print(c[0], c[1]) 
        for j in range(cantidad_pisos-1):
            
    #        print(pisos_x[j][0], pisos_x[j+1][0]) 
            if c[1] > pisos_x[j][0] and c[1] < pisos_x[j+1][0]:
    #            print("entre pisos_x {} {}".format(pisos_x[j][0], pisos_x[j+1,0]))
                print("piso append")
                piso.append(cantidad_pisos-(j+1)+1)
                if c[0] > 0 and c[0] < edificios_y[0][0]:
                    print("edificio append")
                #   print(k)
    #                print("entre edificios {} {}".format(0, edificios_y[0][0]))
                    columna.append(1)
                    break
                for k in range(division_edificio-1):        
                    if c[0] > edificios_y[k][0] and c[0] < edificios_y[k+1][0]:
                        print("entre edificios {} {}".format(edificios_y[k][0], edificios_y[k+1][0]))
                        print("edificio append")
                #      print(k)
                        columna.append(k+2)
                        salir = True
                        break
                if c[0] > edificios_y[-1][0] and c[0] < w:
                    print("edificio append")
                #  print(k)
    #                print("entre edificios {} {}".format(edificios_y[-1][0],w))
                    columna.append(division_edificio+1)
                    break
                else:
                    break   
        if c[1] > pisos_x[-1][0] and c[1] < h:
    #        print("piso fin append")
            piso.append(cantidad_pisos-(j+1)+1)
            salir = False
            if c[0] > 0 and c[0] < edificios_y[0][0]:
                columna.append(1)
                salir=True
            if salir==False:
                for k in range(division_edificio-1):
                    if c[0] > edificios_y[k][0] and c[0] < edificios_y[k+1][0]:
                        columna.append(k+2)
                        salir = True
                        break
            if salir==False:        
                if c[0] > edificios_y[-1][0] and c[0] < h:
                    columna.append(division_edificio+1)
                

    print("Pisos: {}".format(piso))
    print("Edifi: {}".format(columna))   
    ### RESULTADOS
    print("Cantidad de pisos (sin planta baja ni sum): {}".format(cantidad_pisos))
    print("Cantidad de edificios: {}".format(cantidad_edificios))
    print("Cantidad de edificios iluminados: {}".format(cantidad_edificioI))
    print("Cantidad de edificios apagados: {}".format(cantidad_edificioA))

    for i in range(cantidad_edificioI):
        print("Posicion ventana iluminada : piso {} departamento {}".format(piso[i], columna[i]))

    cv.imshow("Edificio Recortado", imagen_recortada)
    cv.waitKey(0)

    cv.destroyAllWindows()
    return "Cantidad de edificios apagados: {}".format(cantidad_edificioA) 
### Segmentar edificio
# H 100 - 120
# V 15 - 20
# S 240 - 245

### Segmentar Pisos
# H 100 - 110
# V 30 - 80
# S 150 - 255

### Segmentar Edificios
# H 155 - 165
# V 45 - 60
# S 150 - 170

### Segmentar Iluminado
# H 25 - 35
# V 55 - 105
# S 170 - 245