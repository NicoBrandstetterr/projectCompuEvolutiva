import re
import random
from sge.utilities import ordered_set


class grammar_sge:
    """
    Esta clase representa una gramática para ser usada en Evolución Gramatical Estructurada (SGE).
    La gramática trabaja con la notación prefija y permite la generación y mapeo de individuos
    según las reglas definidas en un archivo de gramática.
    """
    NT = "NT"
    T = "T"
    NT_PATTERN = "(<.+?>)"
    RULE_SEPARATOR = "::="
    PRODUCTION_SEPARATOR = "|"

    def __init__(self):
        """
        Inicializa la clase con los valores por defecto para representar una gramática.
        """
        self.grammar_file = None
        self.grammar = {}
        self.productions_labels = {}
        self.non_terminals, self.terminals = set(), set()
        self.ordered_non_terminals = ordered_set.OrderedSet()
        self.non_recursive_options = {}
        self.number_of_options_by_non_terminal = None
        self.start_rule = None
        self.max_depth = None
        self.max_init_depth = None

    def set_path(self, grammar_path):
        """
        Define la ruta al archivo de la gramática.
        :param grammar_path: Ruta del archivo de gramática.
        """
        self.grammar_file = grammar_path

    def get_non_recursive_options(self):
        """
        Obtiene las opciones no recursivas de la gramática.
        :return: Diccionario con las opciones no recursivas.
        """
        return self.non_recursive_options

    def set_min_init_tree_depth(self, min_tree_depth):
        """
        Establece la profundidad mínima del árbol al inicializar la gramática.
        :param min_tree_depth: Profundidad mínima del árbol.
        """
        self.max_init_depth = min_tree_depth

    def set_max_tree_depth(self, max_tree_depth):
        """
        Establece la profundidad máxima del árbol para la gramática.
        :param max_tree_depth: Profundidad máxima del árbol.
        """
        self.max_depth = max_tree_depth

    def get_max_depth(self):
        """
        Obtiene la profundidad máxima del árbol de derivación.
        :return: Profundidad máxima del árbol.
        """
        return self.max_depth

    def read_grammar(self):
        """
        Lee la gramática desde un archivo en formato BNF y la convierte a un diccionario de Python.
        El método asume que la primera regla del archivo es el axioma de la gramática.
        Este método fue adaptado de PonyGE versión 0.1.3 por Erik Hemberg y James McDermott.
        """
        if self.grammar_file is None:
            raise Exception("You need to specify the path of the grammar file")

        with open(self.grammar_file, "r") as f:
            for line in f:
                if not line.startswith("#") and line.strip() != "":
                    if line.find(self.PRODUCTION_SEPARATOR):
                        left_side, productions = line.split(self.RULE_SEPARATOR)
                        left_side = left_side.strip()
                        if not re.search(self.NT_PATTERN, left_side):
                            raise ValueError("Left side not a non-terminal!")
                        self.non_terminals.add(left_side)
                        self.ordered_non_terminals.add(left_side)
                        # assumes that the first rule in the file is the axiom
                        if self.start_rule is None:
                            self.start_rule = (left_side, self.NT)
                        temp_productions = []
                        for production in [production.strip() for production in productions.split(self.PRODUCTION_SEPARATOR)]:
                            temp_production = []
                            if not re.search(self.NT_PATTERN, production):
                                if production == "None":
                                    production = ""
                                self.terminals.add(production)
                                temp_production.append((production, self.T))
                            else:
                                for value in re.findall("<.+?>|[^<>]*", production):
                                    if value != "":
                                        if re.search(self.NT_PATTERN, value) is None:
                                            sym = (value, self.T)
                                            self.terminals.add(value)
                                        else:
                                            sym = (value, self.NT)
                                        temp_production.append(sym)
                            temp_productions.append(temp_production)
                        if left_side not in self.grammar:
                            self.grammar[left_side] = temp_productions
        self.compute_non_recursive_options()

    def get_non_terminals(self):
        """
        Devuelve el conjunto ordenado de símbolos no terminales de la gramática.

        Los símbolos no terminales son aquellos que pueden ser reemplazados por uno o más símbolos
        (terminales o no terminales) mediante las reglas de producción de la gramática. Estos símbolos
        son fundamentales en el proceso de derivación de nuevas cadenas o estructuras a partir de la
        regla inicial o axioma.

        El conjunto ordenado asegura que el orden de los símbolos no terminales se mantiene consistente,
        lo cual es importante para ciertos procesos algorítmicos que pueden depender de la consistencia
        del orden para funcionar correctamente, como el mapeo de individuos en algoritmos genéticos.

        Returns:
            OrderedSet: Un conjunto ordenado de símbolos no terminales de la gramática.
        """
        return self.ordered_non_terminals

    def count_number_of_options_in_production(self):
        if self.number_of_options_by_non_terminal is None:
            self.number_of_options_by_non_terminal = {}
            for nt in self.ordered_non_terminals:
                self.number_of_options_by_non_terminal.setdefault(nt, len(self.grammar[nt]))
        return self.number_of_options_by_non_terminal

    def compute_non_recursive_options(self):
        self.non_recursive_options = {}
        for nt in self.ordered_non_terminals:
            choices = []
            for nrp in self.list_non_recursive_productions(nt):
                choices.append(self.grammar[nt].index(nrp))
            self.non_recursive_options[nt] = choices

    def list_non_recursive_productions(self, nt):
        non_recursive_elements = []
        for options in self.grammar[nt]:
            for option in options:
                if option[1] == self.NT and option[0] == nt:
                    break
            else:
                non_recursive_elements += [options]
        return non_recursive_elements

    def recursive_individual_creation(self, genome, symbol, current_depth):
        """
        Crea un individuo de manera recursiva basado en la gramática y las reglas definidas,
        expandiendo los símbolos no terminales hasta alcanzar la profundidad máxima inicial o
        hasta que no haya más símbolos no terminales que expandir.

        La creación del individuo sigue un proceso en el que se expande cada símbolo no terminal 
        seleccionando aleatoriamente una de las posibles producciones definidas en la gramática.
        Esto se realiza de forma recursiva hasta que se alcanza un símbolo terminal o la profundidad
        máxima permitida para la inicialización.

        Parameters:
            genome (list): El genoma del individuo a ser creado. El genoma es una representación 
                        del árbol de derivación en forma de lista, donde cada sublista contiene 
                        las elecciones de producción para un símbolo no terminal dado.
            symbol (str): El símbolo actual desde donde se expandirá la gramática. Este símbolo es
                        normalmente un símbolo no terminal que se reemplaza en cada paso de la 
                        expansión.
            current_depth (int): La profundidad actual del árbol de derivación durante la creación
                                del individuo. Sirve como un contador para asegurar que no se 
                                exceda la profundidad máxima de inicialización.

        Returns:
            int: La profundidad máxima alcanzada durante la creación del individuo. Esto es útil
                para mantener un control sobre la complejidad del árbol generado y puede ser usado 
                posteriormente para evaluar la aptitud del individuo o para aplicar ciertas políticas
                de poda.

        Raises:
            Exception: Si se intenta expandir un símbolo más allá de la profundidad máxima de 
                    inicialización sin encontrar opciones de expansión no recursivas, se podría
                    generar una excepción. Esta situación debería ser manejada adecuadamente
                    en el contexto en el que se utilice este método.
        """
        # Si la profundidad actual supera la profundidad máxima de inicialización, 
        # se busca una producción no recursiva para evitar ciclos infinitos.
        if current_depth > self.max_init_depth:
            possibilities = [index for index, option in enumerate(self.grammar[symbol])
                            if not any(s[0] == symbol for s in option)]
            expansion_possibility = random.choice(possibilities) if possibilities else None
            if expansion_possibility is None:
                raise Exception(f"No non-recursive expansion possibilities found for symbol '{symbol}'.")

        # Si no se supera la profundidad máxima, se selecciona cualquier producción de manera aleatoria.
        else:
            expansion_possibility = random.randint(0, self.count_number_of_options_in_production()[symbol] - 1)

        # La selección se añade al genoma y se expande el símbolo actual.
        genome[self.get_non_terminals().index(symbol)].append(expansion_possibility)
        expansion_symbols = self.grammar[symbol][expansion_possibility]
        
        # Se inicializa la lista de profundidades, empezando con la profundidad actual.
        depths = [current_depth]
        
        # Se itera a través de los símbolos de la producción seleccionada para expandirlos recursivamente.
        for sym in expansion_symbols:
            if sym[1] != self.T:
                # Para cada símbolo no terminal, se llama a la función recursivamente y se actualiza la lista de profundidades.
                depths.append(self.recursive_individual_creation(genome, sym[0], current_depth + 1))
        
        # Se retorna la profundidad máxima alcanzada en esta rama de la expansión.
        return max(depths)


    def mapping(self, mapping_rules, positions_to_map=None, needs_python_filter=False):
        """
        Mapea un conjunto de reglas de mapeo a una representación de individuo.
        :param mapping_rules: Las reglas de mapeo que se utilizarán para generar el individuo.
        :param positions_to_map: Posiciones del genoma que se mapearán.
        :param needs_python_filter: Si se necesita aplicar un filtro para corregir la sintaxis de Python.
        :return: Una tupla con la representación del individuo y la profundidad máxima del árbol de mapeo.
        """
        if positions_to_map is None:
            positions_to_map = [0] * len(self.ordered_non_terminals)
        output = []
        max_depth = self._recursive_mapping(mapping_rules, positions_to_map, self.start_rule, 0, output)
        output = "".join(output)
        if self.grammar_file.endswith("pybnf"):
            output = self.python_filter(output)
        return output, max_depth

    def _recursive_mapping(self, mapping_rules, positions_to_map, current_sym, current_depth, output):
        depths = [current_depth]
        if current_sym[1] == self.T:
            output.append(current_sym[0])
        else:
            current_sym_pos = self.ordered_non_terminals.index(current_sym[0])
            choices = self.grammar[current_sym[0]]
            size_of_gene = self.count_number_of_options_in_production()
            if positions_to_map[current_sym_pos] >= len(mapping_rules[current_sym_pos]):
                if current_depth > self.max_depth:
                    # print "True"
                    possibilities = []
                    for index, option in enumerate(self.grammar[current_sym[0]]):
                        for s in option:
                            if s[0] == current_sym[0]:
                                break
                        else:
                            possibilities.append(index)
                    expansion_possibility = random.choice(possibilities)
                else:
                    expansion_possibility = random.randint(0, size_of_gene[current_sym[0]] - 1)
                mapping_rules[current_sym_pos].append(expansion_possibility)
            current_production = mapping_rules[current_sym_pos][positions_to_map[current_sym_pos]]
            positions_to_map[current_sym_pos] += 1
            next_to_expand = choices[current_production]
            for next_sym in next_to_expand:
                depths.append(
                    self._recursive_mapping(mapping_rules, positions_to_map, next_sym, current_depth + 1, output))
        return max(depths)

    @staticmethod
    def python_filter(txt):
        """
        Crea la sintaxis correcta de Python utilizando símbolos especiales para
        representar la indentación de forma que sea compatible con las reglas BNF.
        :param txt: Texto de entrada a filtrar.
        :return: Texto con la sintaxis de Python corregida.
        """
        txt = txt.replace("\le", "<=")
        txt = txt.replace("\ge", ">=")
        txt = txt.replace("\l", "<")
        txt = txt.replace("\g", ">")
        txt = txt.replace("\eb", "|")
        indent_level = 0
        tmp = txt[:]
        i = 0
        while i < len(tmp):
            tok = tmp[i:i+2]
            if tok == "{:":
                indent_level += 1
            elif tok == ":}":
                indent_level -= 1
            tabstr = "\n" + "  " * indent_level
            if tok == "{:" or tok == ":}" or tok == "\\n":
                tmp = tmp.replace(tok, tabstr, 1)
            i += 1
            # Strip superfluous blank lines.
            txt = "\n".join([line for line in tmp.split("\n") if line.strip() != ""])
        return txt

    def get_start_rule(self):
        return self.start_rule

    def __str__(self):
        """
        Representación en cadena de la gramática definida en la clase.
        :return: La gramática en formato de cadena de texto.
        """
        grammar = self.grammar
        text = ""
        for key in self.ordered_non_terminals:
            text += key + " ::= "
            for options in grammar[key]:
                for option in options:
                    text += option[0]
                if options != grammar[key][-1]:
                    text += " | "
            text += "\n"
        return text

# Create one instance and export its methods as module-level functions.
# The functions share state across all uses
# (both in the user's code and in the Python libraries), but that's fine
# for most programs and is easier for the casual user


_inst = grammar_sge()
set_path = _inst.set_path
read_grammar = _inst.read_grammar
get_non_terminals = _inst.get_non_terminals
count_number_of_options_in_production = _inst.count_number_of_options_in_production
compute_non_recursive_options = _inst.compute_non_recursive_options
list_non_recursive_productions = _inst.list_non_recursive_productions
recursive_individual_creation = _inst.recursive_individual_creation
mapping = _inst.mapping
start_rule = _inst.get_start_rule
set_max_tree_depth = _inst.set_max_tree_depth
set_min_init_tree_depth = _inst.set_min_init_tree_depth
get_max_depth = _inst.get_max_depth
get_non_recursive_options = _inst.get_non_recursive_options



