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


def evaluate(ind, eval_func):
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

    # Inicializar valores de mapeo para la transformación de genotipo a fenotipo.
    mapping_values = [0 for i in ind['genotype']]

    # Mapear el genotipo a fenotipo usando una gramática y obtener la profundidad del árbol.
    phen, tree_depth = grammar.mapping(ind['genotype'], mapping_values)

    # Evaluar el fenotipo utilizando la función de evaluación para obtener la calidad y otra información relevante.
    quality, other_info = eval_func.evaluate(phen)

    # Almacenar el fenotipo y la información de evaluación en el individuo.
    ind['phenotype'] = phen  # Fenotipo resultante.
    ind['fitness'] = quality  # Calidad o aptitud del fenotipo.
    ind['other_info'] = other_info  # Otra información proporcionada por la función de evaluación.
    ind['mapping_values'] = mapping_values  # Valores utilizados en el mapeo genotipo-fenotipo.
    ind['tree_depth'] = tree_depth  # Profundidad del árbol generada por el mapeo.



def setup(parameters_file_path = None):
    if parameters_file_path is not None:
        load_parameters(file_name=parameters_file_path)
    set_parameters(sys.argv[1:])
    if params['SEED'] is None:
        params['SEED'] = int(datetime.now().microsecond)
    logger.prepare_dumps()
    random.seed(params['SEED'])
    grammar.set_path(params['GRAMMAR'])
    grammar.read_grammar()
    grammar.set_max_tree_depth(params['MAX_TREE_DEPTH'])
    grammar.set_min_init_tree_depth(params['MIN_TREE_DEPTH'])


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
    # Número total de generaciones a ejecutar.
    generaciones = params['GENERATIONS']
    # Bucle principal del algoritmo evolutivo que se ejecuta por un número predefinido de generaciones.
    while it <= params['GENERATIONS']:
        print(f'Starting generation: {it+1}/{generaciones}')

        # Evalúa la aptitud de cada individuo en la población si aún no se ha evaluado.
        for i in tqdm(population):
            if i['fitness'] is None:
                evaluate(i, evaluation_function)
        
        # Ordena la población en función de su aptitud.
        population.sort(key=lambda x: x['fitness'])

        # Registra el progreso de la evolución.
        logger.evolution_progress(it, population)
        
        # Selecciona la élite de la población para pasar directamente a la siguiente generación.
        new_population = population[:params['ELITISM']]
        
        # Completa la nueva población utilizando cruzamiento y mutación.
        while len(new_population) < params['POPSIZE']:
            # Decide si realiza un cruzamiento basado en la probabilidad de cruzamiento.
            if random.random() < params['PROB_CROSSOVER']:
                # Selecciona dos padres mediante torneo.
                p1 = tournament(population, params['TSIZE'])
                p2 = tournament(population, params['TSIZE'])
                # Crea un nuevo individuo por cruzamiento.
                ni = crossover(p1, p2)
            else:
                # Selecciona un individuo mediante torneo para ser clonado y mutado.
                ni = tournament(population, params['TSIZE'])
            
            # Muta el nuevo individuo basado en la probabilidad de mutación.
            ni = mutate(ni, params['PROB_MUTATION'])
            
            # Añade el nuevo individuo a la nueva población.
            new_population.append(ni)
        
        # La nueva población se convierte en la población actual para la siguiente generación.
        population = new_population
        
        # Incrementa el contador de generaciones.
        it += 1


