import random
from numpy import cos, sin
from sge.utilities.protected_math import _log_, _div_, _exp_, _inv_, _sqrt_, protdiv


# Función para generar un rango con pasos decimales
# SE DEBE CAMBIAR POR LOS DATASET DE LOS COEFICIENTES PAG 
def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step

# Clase para Regresión Simbólica
class SymbolicRegression():
    def __init__(self, function="quarticpolynomial", has_test_set=False, invalid_fitness=9999999):
        # Inicializar conjuntos de entrenamiento y prueba
        self.__train_set = []
        self.__test_set = None
        self.__number_of_variables = 1
        # Fitness inválido por defecto
        self.__invalid_fitness = invalid_fitness
        self.partition_rng = random.Random()
        self.function = function
        self.has_test_set = has_test_set
        # Leer polinomio y calcular denominadores RRSE
        self.readpolynomial()
        self.calculate_rrse_denominators()
        
    def calculate_rrse_denominators(self):
        # Calcular denominadores para el error RRSE en el conjunto de entrenamiento y prueba
        self.__RRSE_train_denominator = 0
        self.__RRSE_test_denominator = 0
        # Calcular la media de las salidas del conjunto de entrenamiento
        train_outputs = [entry[-1] for entry in self.__train_set]
        train_output_mean = float(sum(train_outputs)) / len(train_outputs)
        # Calcular el denominador RRSE para el conjunto de entrenamiento
        self.__RRSE_train_denominator = sum([(i - train_output_mean)**2 for i in train_outputs])
        # Si hay un conjunto de prueba, calcular el denominador RRSE para este
        if self.__test_set:
            test_outputs = [entry[-1] for entry in self.__test_set]
            test_output_mean = float(sum(test_outputs)) / len(test_outputs)
            self.__RRSE_test_denominator = sum([(i - test_output_mean)**2 for i in test_outputs])

    def read_fit_cases(self):
        # Leer casos de ajuste desde un archivo
        f_in = open(self.__file_problem,'r')
        data = f_in.readlines()
        f_in.close()
        # Convertir los datos leídos en valores flotantes y formar el conjunto de entrenamiento
        fit_cases_str = [ case[:-1].split() for case in data[1:]]
        self.__train_set = [[float(elem) for elem in case] for case in fit_cases_str]
        # Determinar el número de variables a partir del conjunto de entrenamiento
        self.__number_of_variables = len(self.__train_set[0]) - 1

    def readpolynomial(self):
        """
        Este método inicializa el conjunto de entrenamiento (__train_set) con casos de ajuste basados
        en una función matemática predefinida. El conjunto de entrenamiento se genera evaluando
        la función matemática escogida sobre un rango de valores. Si se indica que se requiere un
        conjunto de prueba (has_test_set), este método también inicializará el conjunto de prueba
        (__test_set) de manera similar.

        Las funciones matemáticas que se pueden utilizar son:
        - quarticpolynomial: Un polinomio de cuarto grado.
        - kozapolynomial: Un polinomio propuesto por Koza en su trabajo sobre programación genética.
        - pagiepolynomial: Un polinomio que toma dos variables como entrada.
        - keijzer6: Una serie matemática propuesta por Keijzer.
        - keijzer9: Una fórmula que involucra raíces cuadradas y logaritmos.

        La selección de la función se realiza a través del atributo 'function' de la instancia.
        """
        def quarticpolynomial(inp):
            return inp**4 + inp**3 + inp**2 + inp

        def kozapolynomial(inp):
            return inp**6 - 2*inp**4 + inp**2

        def pagiepolynomial(inp1, inp2):
            return 1.0 / (1 + inp1**-4) + 1.0 / (1 + inp2**-4)

        def keijzer6(inp):
            return sum([1.0/i for i in range(1, inp + 1)])

        def keijzer9(inp):
            return _log_(inp + (inp**2 + 1)**0.5)

        # Determinar el conjunto de entrenamiento basado en la función seleccionada.
        if self.function == "quarticpolynomial":
            self.__train_set = [[x, quarticpolynomial(x)] for x in drange(-1, 1.1, 0.1)]
        elif self.function == "kozapolynomial":
            self.__train_set = [[x, kozapolynomial(x)] for x in drange(-1, 1.1, 0.1)]
        elif self.function == "pagiepolynomial":
            self.__train_set = [[x, y, pagiepolynomial(x, y)] for x in drange(-5, 5.4, 0.4) for y in drange(-5, 5.4, 0.4)]
        elif self.function == "keijzer6":
            self.__train_set = [[x, keijzer6(x)] for x in drange(1, 51, 1)]
        elif self.function == "keijzer9":
            self.__train_set = [[x, keijzer9(x)] for x in drange(0, 101, 1)]
        else:
            raise ValueError(f"Función desconocida: {self.function}")
        
        # Actualizar el número de variables según el conjunto de entrenamiento generado.
        self.__number_of_variables = len(self.__train_set[0]) - 1

        # Si se ha indicado que hay un conjunto de prueba, generar el conjunto de prueba.
        if self.has_test_set:
            if self.function == "keijzer6":
                test_x = list(drange(51, 121, 1))
            elif self.function == "keijzer9":
                test_x = list(drange(0, 101, 0.1))
            else:  # Para el resto de funciones, reutilizamos el rango del conjunto de entrenamiento.
                test_x = list(drange(-1, 1.1, 0.1))

            # Evaluar la función en los puntos de prueba para obtener el conjunto de prueba.
            function = eval(self.function)
            test_y = list(map(function, test_x))
            self.__test_set = list(zip(test_x, test_y))
            self.test_set_size = len(self.__test_set)

    def get_error(self, individual, dataset):
        # Calcular el error de predicción para un individuo y un conjunto de datos
        pred_error = 0
        for fit_case in dataset:
            case_output = fit_case[-1]
            try:
                # Evaluar el individuo con los datos de entrada
                result = eval(individual, globals(), {"x": fit_case[:-1]})
                # Calcular el error cuadrático
                pred_error += (case_output - result)**2
            except (OverflowError, ValueError) as e:
                # En caso de error durante la evaluación, retornar fitness inválido
                return self.__invalid_fitness
        return pred_error

    def evaluate(self, individual):
        # Evaluar el individuo en los conjuntos de entrenamiento y prueba
        error = 0.0
        test_error = 0.0
        if individual is None:
            return None

        # Calcular el error en el conjunto de entrenamiento
        error = self.get_error(individual, self.__train_set)
        error = _sqrt_(error / self.__RRSE_train_denominator)

        if error is None:
            error = self.__invalid_fitness
        
        # Si hay un conjunto de prueba, calcular el error en este
        if self.__test_set is not None:
            test_error = self.get_error(individual, self.__test_set)
            test_error = _sqrt_(test_error / float(self.__RRSE_test_denominator))
        
        # Retornar el error y detalles adicionales de la evaluación
        return error, {'generation': 0, "evals": 1, "test_error": test_error}

if __name__ == "__main__":
    import sge
    eval_func = SymbolicRegression()
    sge.evolutionary_algorithm(evaluation_function=eval_func, parameters_file="parameters/standard.yml")
