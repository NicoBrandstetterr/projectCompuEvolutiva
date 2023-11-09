from ast import Continue
import random
import sys
from token import OP
import numpy as np
import pandas as pd
import sge.grammar_sge as grammar_sge
import sge.grammar_pge as grammar_pge
import sge.grammar_ge as grammar_ge
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
from random import randint, shuffle
from sge.mapper import mapper_GE, mapper_PGE
from sge.functions_pge import update_probs
import copy

import re
from scipy.optimize import minimize
from multiprocessing import Process, Queue
import time

from sge.utilities.stats.trackers import cache

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
    genotype, tree_depth = None, None

    # Si el algoritmo es Structured Grammatical Evolution (SGE)
    if params['ALGORITHM'] == 'SGE':
        # Crear una lista de listas para almacenar el genotipo.
        genotype = [[] for key in grammar_sge.get_non_terminals()]
        # Crear el genotipo de manera recursiva utilizando la regla inicial de la gramática.
        tree_depth = grammar_sge.recursive_individual_creation(genotype, grammar_sge.start_rule()[0], 0)
    
    # Si el algoritmo es Grammatical Evolution (GE)
    elif params['ALGORITHM'] == 'GE':
        # Crear el genotipo como una secuencia de números aleatorios dentro del tamaño de codón definido.
        genotype = [randint(0, params['CODON_SIZE']) for _ in range(params['SIZE_GENOTYPE'])]
    
    # Si el algoritmo es Probabilistic Grammatical Evolution (PGE)
    elif params['ALGORITHM'] == 'PGE':
        # Crear el genotipo como una secuencia de números aleatorios continuos entre 0 y 1.
        genotype = [np.random.uniform() for _ in range(params['SIZE_GENOTYPE'])]
    
    # Devolver el individuo como un diccionario con el genotipo, la aptitud aún sin calcular, y la profundidad del árbol.
    return {'genotype': genotype, 'fitness': None, 'tree_depth': tree_depth}

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

def evaluate(ind, eval_func, OPTIMIZE=False):
    inf_info = list(ind.keys())
    phen, tree_depth, other_info, quality, quality_val,opt_const = None, None, None, np.inf, np.inf, []
    if pd.isna(ind['fitness']):
        if params['ALGORITHM']=='SGE':
            #print('mapping')
            mapping_values = [0 for i in ind['genotype']]
            ind['original_phenotype'], tree_depth = grammar_sge.mapping(ind['genotype'], mapping_values)
            ind['mapping_values'] = mapping_values
        elif params['ALGORITHM']=='GE':
            ind['original_phenotype'], genome, tree, nodes, invalid, tree_depth, used_codonsmapper = mapper_GE(ind['genotype'])
        elif params['ALGORITHM']=='PGE':
            ind['original_phenotype'], ind['gram_counter'] = mapper_PGE(ind['genotype'])
        if "Constant" in ind['original_phenotype']:
            phen,opt_const = Get_phtnotype_time(ind['original_phenotype'],[],eval_func, OPTIMIZE)
        else:
            phen = ind['original_phenotype']
        if (params['CACHE']  and (phen not in cache.keys())) or not params['CACHE']:
            try:
                quality, quality_val, other_info = eval_func.evaluate(phen)
            except:
                pass
        if pd.isna(quality):
            quality = np.inf
        if params['CACHE']:
            cache[phen] = quality
        ind['phenotype'] = phen
        ind['opt_const'] = opt_const
        ind['fitness'] = quality
        ind['fitness val'] = quality_val
        ind['other_info'] = other_info
        ind['tree_depth'] = tree_depth
        ind['optimized'] = OPTIMIZE
    elif (not ind['optimized']) and OPTIMIZE and ("Constant" in ind['original_phenotype']):    
        if 'opt_const' in inf_info: 
            phen,opt_const = Get_phtnotype_time(ind['original_phenotype'],ind['opt_const'],eval_func, OPTIMIZE)
        else:
            phen,opt_const = Get_phtnotype_time(ind['original_phenotype'],[],eval_func, OPTIMIZE)
        if phen not in cache.keys():
            try:
                quality,quality_val, other_info = eval_func.evaluate(phen)
            except:
                pass
            if quality == None:
                quality = np.inf
            if params['CACHE']:
                cache[phen] = quality
        ind['opt_const'] = opt_const
        ind['phenotype'] = phen
        ind['fitness val'] = quality_val
        ind['fitness'] = quality
        ind['other_info'] = other_info
        ind['optimized'] = OPTIMIZE
    return ind

# Se dejo la función como global pues generaba un problema con pickle
def f(phenotype, old_constants, fitness_function, OPTIMIZE, queue):
        res = Get_phenotype(phenotype, old_constants, fitness_function, OPTIMIZE)
        queue.put(res)

def Get_phtnotype_time(phenotype, old_constants, fitness_function, OPTIMIZE):
    
    q = Queue()
    p = Process(target=f, args=(phenotype, old_constants, fitness_function, OPTIMIZE, q))
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
    return replace_phenotype, opt_const

def Get_phenotype(phenotype, old_constants, fitness_function, OPTIMIZE):
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
        if OPTIMIZE:
            try:
                fun = lambda x: eval_ind(x)
                res = minimize(fun, old_constants, method='SLSQP',jac=False)
                opt_const = res['x']
            except:
                print("Error - Hubo un problema al usar método de optimización")
                opt_const = old_constants
        else:
            opt_const = old_constants
        for index in range(n_constants):
            replace_phenotype = replace_phenotype.replace('c[' + str(index) + ']', str(opt_const[index]))
    return replace_phenotype, opt_const

def setup(parameters_file_path=None):
    """
    Configura el entorno de ejecución para un algoritmo genético basado en la gramática, 
    estableciendo los parámetros necesarios, inicializando la semilla para la generación
    de números aleatorios y preparando la gramática según el algoritmo seleccionado.

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
    
    # Configura la gramática y parámetros según el algoritmo genético seleccionado.
    if params['ALGORITHM'] == 'SGE':
        # Para el algoritmo SGE, establece la ruta de la gramática, la lee y establece las profundidades máxima y mínima del árbol.
        grammar_sge.set_path(params['GRAMMAR'])
        grammar_sge.read_grammar()
        grammar_sge.set_max_tree_depth(params['MAX_TREE_DEPTH'])
        grammar_sge.set_min_init_tree_depth(params['MIN_TREE_DEPTH'])
    elif params['ALGORITHM'] == 'GE':
        # Para el algoritmo GE, convierte la gramática a una forma compatible con el sistema.
        params['BNF_GRAMMAR'] = grammar_ge.grammar_ge(params['GRAMMAR'])
    elif params['ALGORITHM'] == 'PGE':
        # Para el algoritmo PGE, establece la ruta de la gramática y la lee.
        grammar_pge.set_path(params['GRAMMAR'])
        grammar_pge.read_grammar()


def evolutionary_algorithm(evaluation_function=None, parameters_file=None):
    """
    Implementa un algoritmo evolutivo genérico que opera en una población de individuos para 
    encontrar una solución óptima basada en una función de evaluación dada.
    
    :param evaluation_function: Instancia de SymbolicRegression que se utilizará para evaluar la aptitud de los individuos.
    :param parameters_file: Ruta al archivo de parámetros para configurar el algoritmo evolutivo.
    """
    
    # Configuración inicial del algoritmo con los parámetros proporcionados.
    setup(parameters_file_path=parameters_file)
    
    # Creación de la población inicial.
    population = list(make_initial_population())
    
    # Inicialización del contador de iteraciones.
    it = 0
    
    # Mejor individuo encontrado en todas las iteraciones.
    best_overall = {}
    
    # Bandera utilizada para controlar la alternancia en el proceso de actualización de probabilidades.
    flag = False
    
    # Número total de generaciones a ejecutar.
    generaciones = params['GENERATIONS']
    
    # Bucle principal del algoritmo evolutivo.
    while it <= params['GENERATIONS']:
        print(f'Starting generation: {it+1}/{generaciones}')
        
        # Limpieza de la caché en intervalos configurados si está habilitada.
        if params['CACHE'] and it % params['CLEAN_CACHE_EACH'] == 0 and it != 0:
            cache = {}
        
        # Evaluación de cada individuo de la población.
        for i in range(len(population)):
            print(f'Evaluating individual: {i+1}/{len(population)}')
            
            # Optimización de individuos en intervalos configurados.
            if params['OPTIMIZE'] and it % params['OPTIMIZE_EACH'] == 0 and it != 0:
                print('Optimizing individual:', i)
                population[i] = evaluate(population[i], evaluation_function, OPTIMIZE=True)
            else:
                population[i] = evaluate(population[i], evaluation_function, OPTIMIZE=False)
            
            # Si se requiere que todos los individuos sean válidos, se generan nuevos hasta cumplir el criterio.
            if params['ALL_VALID']:
                while (pd.isna(population[i]['fitness'])) or (population[i]['fitness'] > 10**6):
                    population[i] = generate_random_individual()
                    population[i] = evaluate(population[i], evaluation_function, OPTIMIZE=False)        
        
        # Ordenar la población en función de su aptitud.
        population.sort(key=lambda x: x['fitness'])
        
        # Proceso específico para el algoritmo 'PGE'.
        if params['ALGORITHM'] == 'PGE':
            # Actualización del mejor individuo si se encuentra uno con mejor aptitud.
            if population[0]['fitness'] <= best_overall.setdefault('fitness', np.inf):
                best_overall = copy.deepcopy(population[0])
            # Actualización de probabilidades alternando entre el mejor global y el de la generación.
            if not flag:
                update_probs(best_overall, params['LEARNING_FACTOR'])
            else:
                update_probs(best_generation, params['LEARNING_FACTOR'])
            flag = not flag
        
        # Registro del progreso de la evolución.
        logger.evolution_progress(it, population)
        
        # Creación de una nueva población manteniendo un porcentaje de élite.
        new_population = population[:params['ELITISM']]
        
        # Población restante generada a través de cruces y mutaciones.
        while len(new_population) < params['POPSIZE']:
            print('Performing crossover for new individual')
            if random.random() < params['PROB_CROSSOVER']:
                p1 = tournament(population, params['TSIZE'])
                p2 = tournament(population, params['TSIZE'])
                ni = crossover(p1, p2)
            else:
                ni = tournament(population, params['TSIZE'])
            
            print('Performing mutation for new individual')
            ni = mutate(ni, params['PROB_MUTATION'])
            new_population.append(ni)
        
        # La nueva población se convierte en la población actual para la siguiente generación.
        population = new_population
        
        # Ajustes adaptativos para el algoritmo 'PGE'.
        if params['ALGORITHM'] == 'PGE':
            best_generation = copy.deepcopy(new_population[0])
            if params['ADAPTIVE']:
                params['LEARNING_FACTOR'] += params['ADAPTIVE_INCREMENT']
        
        # Incremento del contador de iteraciones.
        it += 1

