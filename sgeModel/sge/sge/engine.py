import random
import sys
import sge.grammar as grammar
import sge.logger as logger
from datetime import datetime
from tqdm import tqdm
from sge.operators.recombination import crossover
from sge.operators.mutation import mutate
from sge.operators.selection import tournament
from sge.parameters import (
    params,
    set_parameters,
    load_parameters
)
from multiprocessing import Process, Queue
import time
import re
import numpy as np
import pandas as pd
def generate_random_individual():
    """
    Genera un individuo aleatorio para la población inicial del algoritmo evolutivo.
    
    El tipo de individuo generado depende del algoritmo especificado en los parámetros:
    - 'SGE' (Structured Grammatical Evolution): Crea un genotipo basado en una gramática estructurada.
    - 'GE' (Grammatical Evolution): Crea un genotipo con una secuencia de codones aleatorios.
    - 'PGE' (Probabilistic Grammatical Evolution): Crea un genotipo con valores aleatorios uniformes.
    
    :return: Un diccionario que representa al individuo, que contiene el genotipo, la aptitud (inicialmente None),
             y la profundidad del árbol si es aplicable.
    """
    # Crear una lista de listas para almacenar el genotipo.    
    genotype = [[] for key in grammar.get_non_terminals()]
    
    # Crear el genotipo de manera recursiva utilizando la regla inicial de la gramática.
    tree_depth = grammar.recursive_individual_creation(genotype, grammar.start_rule()[0], 0)
    
    # Devolver el individuo como un diccionario con el genotipo, la aptitud aún sin calcular, y la profundidad del árbol.
    return {'genotype': genotype, 'fitness': None, 'tree_depth' : tree_depth}

def make_initial_population():
    """
    Genera una población inicial para el algoritmo evolutivo.
    
    La población se compone de un número de individuos especificados por el parámetro 'POPSIZE' en los
    parámetros del algoritmo. Cada individuo es generado aleatoriamente.
    
    :yields: Un individuo generado aleatoriamente para la población inicial.
    """
    # Iterar para crear tantos individuos como el tamaño de la población especificado.
    for i in range(params['POPSIZE']):
        # Generar un nuevo individuo aleatoriamente.
        yield generate_random_individual()

def evaluate(ind, eval_func,last_gen):
    """
    Evalúa la calidad o aptitud ('fitness') de un individuo basado en una representación genotípica y 
    una función de evaluación dada.

    El proceso de evaluación implica mapear el genotipo a un fenotipo utilizando una gramática predefinida,
    y luego evaluar el fenotipo utilizando la función de evaluación proporcionada.

    Args:
        ind (dict): Un diccionario que representa un individuo en la población. Debe contener al menos
                    la clave 'genotype', que es una lista que representa la codificación genética del individuo.
        eval_func (object): Un objeto que tiene un método 'evaluate', el cual toma un fenotipo (una cadena de caracteres
                            que representa una solución decodificada) y devuelve una medida de calidad y otra información.
    
    La función realiza las siguientes tareas:
    1. Inicializa una lista de valores de mapeo con ceros, uno por cada gen en el genotipo.
    2. Utiliza la gramática para mapear el genotipo a un fenotipo, obteniendo también la profundidad del árbol
       generado en el proceso de mapeo.
    3. Evalúa el fenotipo con la función de evaluación proporcionada, obteniendo la calidad y otra información.
    4. Almacena el fenotipo, la calidad de la aptitud, la otra información, los valores de mapeo y la profundidad
       del árbol en el diccionario del individuo para su uso posterior.
    """
    # Se printea individuo a evaluar
    # print(f'Engine.evaluate: Individuo a evaluar: {ind}')
    # Inicializar valores de mapeo para la transformación de genotipo a fenotipo.
    mapping_values = [0 for i in ind['genotype']]

    # Mapear el genotipo a fenotipo usando una gramática y obtener la profundidad del árbol.
    ind['original_phenotype'], tree_depth = grammar.mapping(ind['genotype'], mapping_values)
    if "Constant" in ind['original_phenotype']:
            phen,opt_const = Get_phtnotype_time(ind['original_phenotype'],[],eval_func)
    else:
        phen = ind['original_phenotype']
    # Evaluar el fenotipo utilizando la función de evaluación para obtener la calidad y otra información relevante.
    quality, fitness_validation,other_info = eval_func.evaluate(phen,last_gen)

    # Almacenar el fenotipo y la información de evaluación en el individuo.
    ind['phenotype'] = phen  # Fenotipo resultante.
    ind['fitness'] = quality  # Calidad o aptitud del fenotipo.
    ind['fitness_validation'] = fitness_validation
    ind['other_info'] = other_info  # Otra información proporcionada por la función de evaluación.
    ind['mapping_values'] = mapping_values  # Valores utilizados en el mapeo genotipo-fenotipo.
    ind['tree_depth'] = tree_depth  # Profundidad del árbol generada por el mapeo.

def Get_phtnotype_time(phenotype, old_constants, fitness_function):
    try:
        q = Queue()
        p = Process(target=f, args=(phenotype, old_constants, fitness_function, q))
        max_time = 30
        t0 = time.time()

        p.start()
        while time.time() - t0 < max_time:
            p.join(timeout=1)
            if not p.is_alive():
                break

        if p.is_alive():
            #process didn't finish in time so we terminate it
            p.terminate()
            replace_phenotype, opt_const = Get_phenotype(phenotype, old_constants, fitness_function, False)
        else:
            replace_phenotype, opt_const = q.get()
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected, terminating the process...")
        if p.is_alive():
            p.terminate()
        raise  # Re-lanza la excepción KeyboardInterrupt
    return replace_phenotype, opt_const

def f(phenotype, old_constants, fitness_function, queue):
        res = Get_phenotype(phenotype, old_constants, fitness_function)
        queue.put(res)

def Get_phenotype(phenotype, old_constants, fitness_function):
    p = r"Constant"
    n_constants = len(re.findall(p, phenotype))

    replace_phenotype = phenotype
    for i in range(n_constants):
      replace_phenotype = replace_phenotype.replace('Constant', 'c[' + str(i) + ']',1)

    def eval_ind(c):
        aux = replace_phenotype
        for i in range(len(c)):
            aux = aux.replace('c[' + str(i) + ']', str(c[i]))
        return fitness_function.evaluate(aux)[0]

    if n_constants>0:
        if len(old_constants)==0:
            old_constants = np.random.rand(n_constants)
        opt_const = old_constants
        for index in range(n_constants):
            replace_phenotype = replace_phenotype.replace('c[' + str(index) + ']', str(opt_const[index]))
    return replace_phenotype, opt_const

def setup(parameters_file_path=None):
    """
    Configura el entorno de ejecución para un algoritmo genético basado en la gramática, 
    estableciendo los parámetros necesarios, inicializando la semilla para la generación
    de números aleatorios y preparando la gramática.

    Parameters:
        parameters_file_path (str, opcional): Ruta al archivo de parámetros de configuración. 
                                              Si se proporciona, carga los parámetros desde este archivo. 
                                              De lo contrario, los parámetros se toman de los argumentos del sistema.

    Raises:
        FileNotFoundError: Si se proporciona una ruta al archivo de parámetros y el archivo no existe.
        ValueError: Si los valores de los parámetros no son del tipo esperado o están fuera de los rangos permitidos.
        Exception: Para cualquier otro error que pueda ocurrir durante la carga y configuración de parámetros.
    """

    # Carga los parámetros desde un archivo si se proporciona la ruta.
    if parameters_file_path is not None:
        load_parameters(file_name=parameters_file_path)
    
    # Sobreescribe los parámetros con los argumentos de línea de comando si existen.
    set_parameters(sys.argv[1:])
    
    # Establece una semilla basada en el microsegundo actual si no se ha definido una.
    if params['SEED'] is None:
        params['SEED'] = int(datetime.now().microsecond)
    
    # Prepara la estructura para volcar información durante la ejecución.
    logger.prepare_dumps()
    
    # Inicializa la semilla para la generación de números aleatorios.
    random.seed(params['SEED'])
    # Para el algoritmo SGE, establece la ruta de la gramática, la lee y establece las profundidades máxima y mínima del árbol.
    grammar.set_path(params['GRAMMAR'])
    grammar.read_grammar()
    grammar.set_max_tree_depth(params['MAX_TREE_DEPTH'])
    grammar.set_min_init_tree_depth(params['MIN_TREE_DEPTH'])


def contains_variables(phenotype):
    """
    Verifica si el fenotipo contiene al menos una de las variables requeridas.

    Args:
        phenotype (str): El fenotipo a verificar.

    Returns:
        bool: True si contiene al menos una variable, False si solo tiene constantes.
    """
    # Lista de variables a buscar en el fenotipo.
    variables = ["x[0]", "x[1]", "x[2]"]
    return any(var in phenotype for var in variables)


def replace_with_valid_individuals(population, new_population, params):
    """
    Reemplaza individuos en new_population que solo contienen constantes con individuos válidos de population.

    Args:
        population (list): La lista completa de la población de individuos.
        new_population (list): La lista de la nueva población de individuos seleccionados para elitismo.
        params (dict): Diccionario de parámetros que contiene 'ELITISM'.

    Returns:
        list: La nueva población actualizada con todos los individuos válidos.
    """
    
    elite_count = params['ELITISM']
    # Índice para rastrear el próximo individuo a considerar para reemplazo en la población original.
    next_index = elite_count

    for i in range(len(new_population)):
        # bool individuo contiene variables
        var_ind = contains_variables(new_population[i]['original_phenotype'])
        
        if not var_ind:
            print(f"phenotipo del individuo no contiene variable, se procede a cambiar por otro")
            # Encuentra el siguiente individuo válido en la población.
            while next_index < len(population) and not var_ind:
                next_index += 1

            if next_index < len(population):
                # Reemplazar el individuo no válido con el próximo válido encontrado.
                new_population[i] = population[next_index]

    return new_population


def evolutionary_algorithm(evaluation_function=None, parameters_file=None):
    """
    Ejecuta un algoritmo evolutivo para optimizar una población de individuos.

    Args:
        evaluation_function (callable, optional): Función que evalúa la aptitud de un individuo.
            Debe ser una función que tome un individuo como argumento y devuelva un valor numérico
            que representa su aptitud. Si no se proporciona, se debe definir en el archivo de parámetros.
        parameters_file (str, optional): Ruta al archivo de parámetros que configura el algoritmo.
            Si se proporciona, el archivo será leído y los parámetros serán establecidos según su contenido.

    El algoritmo procede de la siguiente manera:
    1. Carga los parámetros del algoritmo desde un archivo si se proporciona.
    2. Inicializa una población de individuos.
    3. Evalúa la aptitud de cada individuo en la población.
    4. Ordena la población basada en la aptitud de cada individuo.
    5. Itera a través de un número de generaciones, realizando en cada una:
        a. Conservación de una fracción de la población como élite.
        b. Población de nuevos individuos hasta alcanzar el tamaño de población deseado, usando:
            i. Cruce entre individuos seleccionados por un torneo si un número aleatorio es menor que la probabilidad de cruce.
            ii. Mutación de un individuo seleccionado por el torneo con una probabilidad de mutación dada.
        c. Reemplazo de la población actual con la nueva población.
        d. Incremento del contador de generaciones.
    6. Registra el progreso del algoritmo.
    """
    # Configura los parámetros del algoritmo a partir del archivo proporcionado.
    setup(parameters_file_path=parameters_file)
    
    # Crea la población inicial a partir de una función que genera individuos aleatorios.
    population = list(make_initial_population())
    
    # Inicializa el contador de generaciones.
    it = 0
    last_gen=False
    # Número total de generaciones a ejecutar.
    generaciones = params['GENERATIONS']
    # Bucle principal del algoritmo evolutivo que se ejecuta por un número predefinido de generaciones.
    while it <= params['GENERATIONS']:
        print(f'Starting generation: {it+1}/{generaciones+1}')
        if it == params['GENERATIONS']:
            last_gen=True
        # Evalúa la aptitud de cada individuo en la población si aún no se ha evaluado.
        # Referencia a pag 30 donde se evalua un individuo hasta que esté correcto el fitness
        for i in tqdm(population):
            if i['fitness'] is None:
                evaluate(i, evaluation_function,last_gen)
            tries = 0
            while (i is None) or (i['fitness'] is None) or (i['fitness'] > 10**6):
                if tries == 5:
                    break
                tries+=1
                print(f"individuo es Nulo o con error muy alto, re generando... intento N°{tries}")
                # print(f"Individuo: {i}")
                i = generate_random_individual()
                evaluate(i, evaluation_function,last_gen)        
        # Ordena la población en función de su aptitud.
        population.sort(key=lambda x: x['fitness'])

        # Registra el progreso de la evolución.
        logger.evolution_progress(it, population)
        
        
        # Selecciona la élite de la población para pasar directamente a la siguiente generación.
        new_population = population[:params['ELITISM']]
        # Asegúrate de que todos los individuos de new_population sean válidos.
        new_population = replace_with_valid_individuals(population, new_population, params)
        # Completa la nueva población utilizando cruzamiento y mutación.
        while len(new_population) < params['POPSIZE']:
            # Decide si realiza un cruzamiento basado en la probabilidad de cruzamiento.
            if random.random() < params['PROB_CROSSOVER']:
                # Selecciona dos padres mediante torneo.
                # print('Performing crossover for new individual')
                p1 = tournament(population, params['TSIZE'])
                p2 = tournament(population, params['TSIZE'])
                # Crea un nuevo individuo por cruzamiento.
                ni = crossover(p1, p2)
            else:
                # Selecciona un individuo mediante torneo para ser clonado y mutado.
                ni = tournament(population, params['TSIZE'])
            # print('Performing mutation for new individual')
            # Muta el nuevo individuo basado en la probabilidad de mutación.
            ni = mutate(ni, params['PROB_MUTATION'])
            if last_gen:
                evaluate(ni,evaluation_function,last_gen)
            # Añade el nuevo individuo a la nueva población.
            new_population.append(ni)
        
        # La nueva población se convierte en la población actual para la siguiente generación.
        population = new_population
        
        # Incrementa el contador de generaciones.
        it += 1


