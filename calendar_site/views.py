from django.http import HttpResponse
from django.shortcuts import render

from .Parcial2019 import Ejercicio


def hello_world(request):
    return HttpResponse('Hola Mundo')

def calendar(request, data):
   # data = User.objects.get(data=data)
    result = Ejercicio.resolucion(data)
    return HttpResponse(result)
