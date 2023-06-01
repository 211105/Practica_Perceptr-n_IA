import csv
import numpy as np
import random
import pandas as pd
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt

datos_x = []
datos_y = []

# Abre el archivo CSV
with open('Actividad/211105.CSV', 'r') as archivo_csv:
    # Lee el archivo CSV utilizando el lector de CSV
    lector_csv = csv.reader(archivo_csv, delimiter=';')

    # Omite la primera fila que contiene los encabezados
    next(lector_csv)

    # Itera sobre cada fila del archivo CSV
    for fila in lector_csv:
        # Obtiene los valores de x1, x2 y x3 de la fila actual
        valores_x = [float(i) for i in fila[1:4]]

        # Agrega un 1 al inicio de cada elemento de valores_x
        valores_x.insert(0, 1)

        # Obtiene el valor de Y de la última columna
        valor_y = float(fila[4])

        # Agrega los valores de x1, x2 y x3 a la lista de datos_x
        datos_x.append(valores_x)

        # Agrega el valor de Y a la lista de datos_y
        datos_y.append(valor_y)


def definir_peso_inicial(matriz, w):
    for i in range(int(len(matriz[0]))):
        w = np.append(w, random.randint(0, 20))
    return w


def obtener_u(matriz, w):
    # Realizar el producto punto entre la matriz y el vector
    u = np.dot(matriz, w)
    return u


def obtener_yc(u):
    yc = np.maximum(0, u)  # Función de activación rectificada (ReLU)
    return yc


def obtener_error(yd, yc):
    error = yd - yc
    return error


def obtener_deltaW(eta, error):
    error = np.delete(error, -1)
    resultado = eta * error
    return resultado


def obtener_nueva_w(delta_w, w):
    primeros_cuatro = delta_w[:4]
    delta_w = np.array(primeros_cuatro, dtype=float)
    resultado = w + delta_w
    return resultado


def mostrar_tabla(df):
    def abrir_grafica():
        mostrar_grafica(df['Tolerancia al Error'])

    window = tk.Tk()
    window.title("Tabla de Resultados")
    window.geometry("800x600")  # Establece el tamaño de la ventana

    tree = ttk.Treeview(window)
    tree["columns"] = tuple(df.columns)
    tree["show"] = "headings"
    for column in df.columns:
        tree.heading(column, text=column)

    # Agrega los datos a la tabla
    for index, row in df.iterrows():
        tree.insert("", "end", values=tuple(row))

    # Ajusta el tamaño de las columnas según su contenido
    for column in df.columns:
        tree.column(column, width=100)  # Establece un ancho inicial para las columnas

    scroll = ttk.Scrollbar(window, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scroll.set)
    scroll.pack(side="right", fill="y")

    tree.pack(fill="both", expand=True)  # Ajusta la tabla al tamaño de la ventana

    # Ajusta el tamaño de las columnas después de que la ventana se ha mostrado
    window.update()
    for column in df.columns:
        col_width = tree.column(column, width=None)
        tree.column(column, width=col_width)  # Ajusta el ancho de la columna según el contenido

    # Botón para abrir la gráfica
    button = ttk.Button(window, text="Abrir Gráfica", command=abrir_grafica)
    button.pack()

    window.mainloop()



def mostrar_grafica(tolerancia_de_error):
    # Obtener los valores de tolerancia_de_error
    iteration_numbers = range(1, len(tolerancia_de_error) + 1)

    # Graficar los valores de tolerancia_de_error
    plt.plot(iteration_numbers, tolerancia_de_error)
    plt.scatter(iteration_numbers, tolerancia_de_error, color='red')
    plt.xlabel('Iteraciones')
    plt.ylabel('Tolerancia al Error')
    plt.title('Gráfico de Tolerancia al Error por Iteración')
    plt.grid(True)
    plt.show()


def main():
    etas = []
    pesos_iniciales = []
    pesos_finales = []
    tolerancia_de_error = []
    iteraciones = []
    eta = 0.2
    matriz = np.array(datos_x, dtype=float)
    yd = np.array(datos_y)

    w = np.array([])
    w = definir_peso_inicial(matriz, w)

    # Inicia el bucle
    for i in range(120):
        iteraciones.append(i + 1)  # Agregar el número de iteración actual
        pesos_iniciales.append(w)
        u = obtener_u(matriz, w)
        yc = obtener_yc(u)
        error = obtener_error(yd, yc)
        # Guardar
        tolerancia_de_error.append(np.sum(np.abs(error)))
        delta_w = obtener_deltaW(eta, error)
        # Guardar
        etas.append(eta)
        w = obtener_nueva_w(delta_w, w)
        # Guardar
        pesos_finales.append(w)

        # Verificar si el margen de error es cero
        if np.sum(np.abs(error)) == 0:
            break

    # Create a DataFrame
    df = pd.DataFrame({
        'Iteración': iteraciones,
        'Pesos Iniciales': [str(w) for w in pesos_iniciales],
        'Eta': etas,
        'Pesos Finales': [str(w) for w in pesos_finales],
        'Tolerancia al Error': tolerancia_de_error
    })

    # Show the table in a window
    mostrar_tabla(df)


main()
