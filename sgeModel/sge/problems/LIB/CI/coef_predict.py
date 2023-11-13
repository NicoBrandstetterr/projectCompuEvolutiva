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
    
    def __init__(self,segment,coef, has_test_set=False, invalid_fitness=9999999,path = 'resources/LIB/CI'):
        """
        Inicializa la clase SymbolicRegression con valores por defecto.

        :param has_test_set: No se usa actualmente. Indicaría si hay un conjunto de pruebas disponible.
        :param invalid_fitness: Valor que se usa para la aptitud cuando la evaluación de un individuo falla.
        """
        self.__invalid_fitness = invalid_fitness
        self.read_fit_cases(segment,coef,path)

    def read_fit_cases(self,segment,coef,path):
        """
        Lee y procesa los casos de ajuste de los archivos txt, correspondientes a los datos de coeficiente de arrastre 
        para paquetes de baterías con diferente número de celdas. Prepara los datos para el entrenamiento y la validación.
        """
        if segment == 0:
            # Carga de datos para un paquete de baterías de 25 celdas
            self.df_25 = pd.read_csv(f'{path}/df_{coef}_25.txt', sep=',').sample(n=1000, random_state=1)

            # Carga de datos para un paquete de baterías de 74 celdas
            self.df_74 = pd.read_csv(f'{path}/df_{coef}_74.txt', sep=',').sample(n=1000, random_state=1)


            # Muestreo aleatorio si los tamaños son diferentes
            if len(self.df_25) != len(self.df_74):
                # Determinar el tamaño mínimo entre los dos dataframes
                min_size = min(len(self.df_25), len(self.df_74))
                self.df_25 = self.df_25.sample(n=min_size, random_state=1)
                self.df_74 = self.df_74.sample(n=min_size, random_state=1)

            # Asignar valores de X e Y
            self.X_25 = self.df_25.values[:, :-1]
            self.Y_25 = self.df_25.values[:, -1]
            self.X_74 = self.df_74.values[:, :-1]
            self.Y_74 = self.df_74.values[:, -1]

            # Carga y muestreo aleatorio de los datos para un paquete de baterías de 53 celdas
            self.df_53 = pd.read_csv(f'{path}/df_{coef}_53.txt', sep=',').sample(n=1000, random_state=1)
            self.X_53 = self.df_53.values[:, :-1]
            self.Y_53 = self.df_53.values[:, -1]

            # Carga y muestreo aleatorio de los datos para un paquete de baterías de 102 celdas
            self.df_102 = pd.read_csv(f'{path}/df_{coef}_102.txt', sep=',').sample(n=1000, random_state=1)
            self.X_102 = self.df_102.values[:, :-1]
            self.Y_102 = self.df_102.values[:, -1]
        else:
            # Carga de datos para un paquete de baterías de 25 celdas
            self.df_25 = pd.read_csv(f'{path}/df_{coef}_25_{segment}.txt', sep=',')
            self.df_25 = self.df_25 if len(self.df_25) < 1000 else self.df_25.sample(n=1000, random_state=1)

            # Carga de datos para un paquete de baterías de 74 celdas
            self.df_74 = pd.read_csv(f'{path}/df_{coef}_74_{segment}.txt', sep=',')
            self.df_74 = self.df_74 if len(self.df_74) < 1000 else self.df_74.sample(n=1000, random_state=1)


            # Muestreo aleatorio si los tamaños son diferentes
            if len(self.df_25) != len(self.df_74):
                # Determinar el tamaño mínimo entre los dos dataframes
                min_size = min(len(self.df_25), len(self.df_74))
                self.df_25 = self.df_25.sample(n=min_size, random_state=1)
                self.df_74 = self.df_74.sample(n=min_size, random_state=1)

            # Asignar valores de X e Y
            self.X_25 = self.df_25.values[:, :-1]
            self.Y_25 = self.df_25.values[:, -1]
            self.X_74 = self.df_74.values[:, :-1]
            self.Y_74 = self.df_74.values[:, -1]

            # Carga y muestreo aleatorio de los datos para un paquete de baterías de 53 celdas
            self.df_53 = pd.read_csv(f'{path}/df_{coef}_53_{segment}.txt', sep=',')
            self.df_53 = self.df_53 if len(self.df_53) < 1000 else self.df_53.sample(n=1000, random_state=1)
            self.X_53 = self.df_53.values[:, :-1]
            self.Y_53 = self.df_53.values[:, -1]


            # Carga y muestreo aleatorio de los datos para un paquete de baterías de 102 celdas
            self.df_102 = pd.read_csv(f'{path}/df_{coef}_102_{segment}.txt', sep=',')
            self.df_102 = self.df_102 if len(self.df_102) < 1000 else self.df_102.sample(n=1000, random_state=1)
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
            # print(f'get_error - individuo: {individual}')
            # Evaluar la expresión del individuo en el contexto de los datos de entrada
            Y_pred = list(map(lambda x: eval(individual), dataset))
            # Calcular el error cuadrático medio (MSE) para las predicciones
            error = mean_squared_error(Y_train, Y_pred, squared=False)
        except Exception as e:
            # Manejo de errores en la evaluación del individuo
            # print(f"Error evaluating individual: {e}")
            # print(f'individuo que genera error: {individual}')
            error = self.__invalid_fitness
        
        # Asignar un valor de error inválido si el error es None
        if error is None:
            error = self.__invalid_fitness
        
        return error

    def evaluate(self, individual,last_gen):
        """
        Evalúa la aptitud de un individuo en varios conjuntos de datos representando diferentes configuraciones de paquetes 
        de baterías y calcula la aptitud general basada en el error de predicción del coeficiente de arrastre.

        :param individual: La expresión matemática del individuo a evaluar.
        :return: Una tupla que contiene la aptitud para el conjunto de 25 celdas y la aptitud promedio para los conjuntos 
                 de 53, 74 y 102 celdas, junto con un diccionario que detalla la aptitud para cada configuración.
        """
        if individual is None:
            return self.__invalid_fitness
        # print(f"cdrag.evaluate - se evaluara individuo: {individual}")
        # # Calcular el error para cada configuración de paquete de baterías
        # error_25 = self.get_error(individual, self.Y_25, self.X_25)
        # error_53 = self.get_error(individual, self.Y_53, self.X_53)
        # error_74 = self.get_error(individual, self.Y_74, self.X_74)
        # error_102 = self.get_error(individual, self.Y_102, self.X_102)
        
        # # Uso del error del conjunto de 25 celdas como aptitud de entrenamiento
        # fitness_train = error_25
        # # Calcular la aptitud promedio para los conjuntos de validación
        # fitness_val = np.mean([error_53, error_74, error_102])
        

        # Concatenar los conjuntos X e Y para el entrenamiento
        X_train = np.concatenate((eval_func.X_25, eval_func.X_74))
        Y_train = np.concatenate((eval_func.Y_25, eval_func.Y_74))

        # Calcular el error de entrenamiento con los conjuntos combinados
        error_train = self.get_error(individual, Y_train, X_train)

        # Calcular el error de validación con el conjunto X_102 y Y_102
        error_validation = self.get_error(individual, eval_func.Y_102, eval_func.X_102)
        # Calcular el error de prueba con el conjunto X_53 y Y_53
        error_test = self.get_error(individual, eval_func.Y_53, eval_func.X_53)


        # Devolver la aptitud de entrenamiento, de validación y un diccionario detallado
        # return error_train, error_validation, {
        #     'fitness 25': error_25,
        #     'fitness 53': error_53,
        #     'fitness 74': error_74,
        #     'fitness 102': error_102
        # }
        return error_train, error_validation, {"errortest": error_test,}
    

def parse_args():
    parser = argparse.ArgumentParser(description='Run SGE algorithm with custom settings.')
    parser.add_argument('--experiment_name', type=str, required=True, help='Name of the experiment.')
    parser.add_argument('--parameters', type=str, required=True, help='Path to the parameters file.')
    parser.add_argument('--algorithm', type=str, required=True, help='Name of the algorithm to use.')
    parser.add_argument('--seg', type=int, required=False, default=1, help='Segment number for Symbolic Regression.')
    parser.add_argument('--coef', type=str, required=False, default="cdrag", help='Coefficient for Symbolic Regression.')
    parser.add_argument('--choice', type=str, required=False, default="1", help='choice of db.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    import sge
    args = parse_args()
    path = 'resources/LIB/CI'
    if args.choice == "2":
        path = 'resources/LIB/CI2'
    eval_func = SymbolicRegression(segment=args.seg,coef=args.coef,path=path)
    sge.evolutionary_algorithm(evaluation_function=eval_func)