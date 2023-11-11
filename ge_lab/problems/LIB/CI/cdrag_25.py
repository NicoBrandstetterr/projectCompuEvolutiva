import random
from numpy import cos, sin
from sge.utilities.protected_math import _log_, _div_, _exp_, _inv_, _sqrt_, protdiv
from sge.engine import setup
import sge
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')


class SymbolicRegression:
    """
    Clase diseñada para realizar regresión simbólica enfocada en modelar el coeficiente de arrastre (cdrag) 
    como una función de las características de diferentes configuraciones de paquetes de baterías.
    Los datos se agrupan por la cantidad de celdas en el paquete de baterías.
    """
    
    def __init__(self, has_test_set=False, invalid_fitness=9999999):
        """
        Inicializa la clase SymbolicRegression con valores por defecto.

        :param has_test_set: No se usa actualmente. Indicaría si hay un conjunto de pruebas disponible.
        :param invalid_fitness: Valor que se usa para la aptitud cuando la evaluación de un individuo falla.
        """
        self.__invalid_fitness = invalid_fitness
        self.read_fit_cases()

    def read_fit_cases(self):
        """
        Lee y procesa los casos de ajuste de los archivos txt, correspondientes a los datos de coeficiente de arrastre 
        para paquetes de baterías con diferente número de celdas. Prepara los datos para el entrenamiento y la validación.
        """
        # Carga y muestreo aleatorio de los datos para un paquete de baterías de 25 celdas
        self.df_25 = pd.read_csv('resources/LIB/CI/df_cdrag_25.txt', sep=',').sample(n=1000, random_state=1)
        self.X_25 = self.df_25.values[:, :-1]
        self.Y_25 = self.df_25.values[:, -1]
        
        # Carga y muestreo aleatorio de los datos para un paquete de baterías de 53 celdas
        self.df_53 = pd.read_csv('resources/LIB/CI/df_cdrag_53.txt', sep=',').sample(n=1000, random_state=1)
        self.X_53 = self.df_53.values[:, :-1]
        self.Y_53 = self.df_53.values[:, -1]
        
        # Carga y muestreo aleatorio de los datos para un paquete de baterías de 74 celdas
        self.df_74 = pd.read_csv('resources/LIB/CI/df_cdrag_74.txt', sep=',').sample(n=1000, random_state=1)
        self.X_74 = self.df_74.values[:, :-1]
        self.Y_74 = self.df_74.values[:, -1]
        
        # Carga y muestreo aleatorio de los datos para un paquete de baterías de 102 celdas
        self.df_102 = pd.read_csv('resources/LIB/CI/df_cdrag_102.txt', sep=',').sample(n=1000, random_state=1)
        self.X_102 = self.df_102.values[:, :-1]
        self.Y_102 = self.df_102.values[:, -1]

    def get_error(self, individual, Y_train, dataset):
        """
        Evalúa un individuo (expresión matemática) calculando el error cuadrático medio entre las predicciones y 
        los valores reales del coeficiente de arrastre para un conjunto de datos específico de paquetes de baterías.

        :param individual: Expresión del individuo a evaluar como una cadena de texto.
        :param Y_train: Valores reales del coeficiente de arrastre (cdrag) para el conjunto de datos.
        :param dataset: Características del conjunto de datos (sin incluir el cdrag).
        :return: Error cuadrático medio o un valor de aptitud inválido si la evaluación falla.
        """
        try:
            print(f'get_error - individuo: {individual}')
            # Evaluar la expresión del individuo en el contexto de los datos de entrada
            Y_pred = list(map(lambda x: eval(individual), dataset))
            # Calcular el error cuadrático medio (MSE) para las predicciones
            error = mean_squared_error(Y_train, Y_pred, squared=False)
        except Exception as e:
            # Manejo de errores en la evaluación del individuo
            print(f"Error evaluating individual: {e}")
            error = self.__invalid_fitness
        
        # Asignar un valor de error inválido si el error es None
        if error is None:
            error = self.__invalid_fitness
        
        return error

    def evaluate(self, individual):
        """
        Evalúa la aptitud de un individuo en varios conjuntos de datos representando diferentes configuraciones de paquetes 
        de baterías y calcula la aptitud general basada en el error de predicción del coeficiente de arrastre.

        :param individual: La expresión matemática del individuo a evaluar.
        :return: Una tupla que contiene la aptitud para el conjunto de 25 celdas y la aptitud promedio para los conjuntos 
                 de 53, 74 y 102 celdas, junto con un diccionario que detalla la aptitud para cada configuración.
        """
        if individual is None:
            return self.__invalid_fitness
        print(f"cdrag.evaluate - se evaluara individuo: {individual}")
        # Calcular el error para cada configuración de paquete de baterías
        error_25 = self.get_error(individual, self.Y_25, self.X_25)
        error_53 = self.get_error(individual, self.Y_53, self.X_53)
        error_74 = self.get_error(individual, self.Y_74, self.X_74)
        error_102 = self.get_error(individual, self.Y_102, self.X_102)
        
        # Uso del error del conjunto de 25 celdas como aptitud de entrenamiento
        fitness_train = error_25
        # Calcular la aptitud promedio para los conjuntos de validación
        fitness_val = np.mean([error_53, error_74, error_102])
        
        # Devolver la aptitud de entrenamiento, de validación y un diccionario detallado
        return fitness_train, fitness_val, {
            'fitness 25': error_25,
            'fitness 53': error_53,
            'fitness 74': error_74,
            'fitness 102': error_102
        }
if __name__ == "__main__":
    import sge
    eval_func = SymbolicRegression()
    sge.evolutionary_algorithm(evaluation_function=eval_func)