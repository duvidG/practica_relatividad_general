#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 12:07:41 2020

@author: dgiron
"""
import matplotlib.pyplot as plt
import numpy as np
import math

def biseccion(f, a, b, e):
    # f es la función que define f(x)=0. a y b son dos puntos tales que el signo de f(a) y el de f(b) son distintos. Finalmente, "e" es el "epsilon" que marca cuándo consideramos que podemos parar
    fa = f(a)
    fb = f(b)
    if np.sign(fa) == np.sign(fb):
        print("Error: la función tiene igual signo en los extremos")
        return # La función no devuelve ningún resultado
    else:
        c = (a+b)/2
        fc = f(c)
        condicion_parada = (np.abs(fc) < e) or (np.abs(b-a)<e)
        while not(condicion_parada):
            if np.sign(fc) == np.sign(fa):
                a = c # El nuevo intervalo será [c,b], ahora se llama [a,b]
                fa = f(a)
                c = (a+b)/2 # El nuevo punto medio
                fc = f(c)
            else:
                b = c # El nuevo intervalo será [a,c], ahora se llama [a,b]
                fb = f(b)
                c = (a+b)/2 # El nuevo punto medio
                fc = f(c)
            condicion_parada = (np.abs(fc) < e) or (np.abs(b-a)<e)
    return c

def biseccion2D(f, a, b, e, par):
    # f es la función que define f(x)=0. a y b son dos puntos tales que el signo de f(a) y el de f(b) son distintos. Finalmente, "e" es el "epsilon" que marca cuándo consideramos que podemos parar
    fa = f(a, *par)
    fb = f(b, *par)
    if np.sign(fa) == np.sign(fb):
        print("Error: la función tiene igual signo en los extremos")
        return # La función no devuelve ningún resultado
    else:
        # print("a=Extremo_izdo\tc=Punto_Medio\tb=Extremo_dcho\t\tf(c)") # Escribiremos una tabla para ir mostrando los resultados
        c = (a+b)/2
        fc = f(c, *par)
        # print(f"{a:.5f}\t\t\t{c:.5f}\t\t\t{b:.5f}\t\t\t{fc:.5f}")
        condicion_parada = (np.abs(fc) < e) or (np.abs(b-a)<e)
        while not(condicion_parada):
            if np.sign(fc) == np.sign(fa):
                a = c # El nuevo intervalo será [c,b], ahora se llama [a,b]
                fa = f(a, *par)
                c = (a+b)/2 # El nuevo punto medio
                fc = f(c, *par)
            else:
                b = c # El nuevo intervalo será [a,c], ahora se llama [a,b]
                fb = f(b, *par)
                c = (a+b)/2 # El nuevo punto medio
                fc = f(c, *par)
            condicion_parada = (np.abs(fc) < e) or (np.abs(b-a)<e)
            # print(f"{a:.5f}\t\t\t{c:.5f}\t\t\t{b:.5f}\t\t\t{fc:.5f}")
        
    return c

def Newton(f, x, h, e):
    # f es la función que define f(X)=0. x es el valor inicial dado a la iteración. h es el parámetro usado para calcular la derivada de forma aproximada. "e" es el epsilon usado para decidir cuándo detener la iteración.
    fx = f(x)
    contador = 1 # Contaremos cuántas iteraciones hacemos
    condicion_parada = (np.abs(fx) < e) or (contador>100) # A lo sumo 100 iteraciones
    print("x=Punto_actual\t f(x)") # Escribiremos una tabla para ir mostrando los resultados
    print(f"{x:.5f}\t\t\t{fx:.5f}")
    while not(condicion_parada):
        denominador = f(x+h) - f(x-h) 
        if (denominador == 0) and (fx!=0): # Nota: "!=" significa "distinto de"
            print("Error: he encontrado una división por 0, intenta con otro punto inicial")
        else:
            x = x - 2*h*fx/denominador
            fx = f(x)
            contador = contador + 1
            print(f"{x:.5f}\t\t\t{fx:.5f}")
        condicion_parada = (np.abs(fx) < e) or (contador>100)
    if contador > 100:
        print("No se ha alcanzado convergencia, intenta con otro punto inicial")
        return
    else:
         print(f"La solución aproximada es x={x:.5f} que cumple f(x)={fx:.5f}")
    return x

def Newton2D(f, x, h, e, par):
    # f es la función que define f(X)=0. x es el valor inicial dado a la iteración. h es el parámetro usado para calcular la derivada de forma aproximada. "e" es el epsilon usado para decidir cuándo detener la iteración.
    fx = f(x, *par)
    contador = 1 # Contaremos cuántas iteraciones hacemos
    condicion_parada = (np.abs(fx) < e) or (contador>100) # A lo sumo 100 iteraciones
    print("x=Punto_actual\t f(x)") # Escribiremos una tabla para ir mostrando los resultados
    print(f"{x:.5f}\t\t\t{fx:.5f}")
    while not(condicion_parada):
        denominador = f(x+h, *par) - f(x-h, *par) 
        if (denominador == 0) and (fx!=0): # Nota: "!=" significa "distinto de"
            print("Error: he encontrado una división por 0, intenta con otro punto inicial")
        else:
            x = x - 2*h*fx/denominador
            fx = f(x, *par)
            contador = contador + 1
            print(f"{x:.5f}\t\t\t{fx:.5f}")
        condicion_parada = (np.abs(fx) < e) or (contador>100)
    if contador > 100:
        print("No se ha alcanzado convergencia, intenta con otro punto inicial")
        return
    else:
         print(f"La solución aproximada es x={x:.5f} que cumple f(x)={fx:.5f}")
    return x

