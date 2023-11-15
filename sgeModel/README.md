# sge3: And Implementation of Dynamic Structured Grammatical Evolution in Python 3

Structured Grammatical Evolution (SGE) is a recent Grammatical Evolution (GE) variant that aims at addressing some of its locality and redundancy issues. The SGE distinctive feature is having a one-to-one correspondence between genes and non-terminals of the grammar being used. If you use this code, a reference to the following work would be greatly appreciated:

```
@article{Lourenco2016,
 title={Unveiling the properties of structured grammatical evolution},
  author={Louren{\c{c}}o, Nuno and Pereira, Francisco B and Costa, Ernesto},
  journal={Genetic Programming and Evolvable Machines},
  volume={17},
  number={3},
  pages={251--289},
  year={2016},
  publisher={Springer}
}

@incollection{lourencco2018structured,
  title={Structured grammatical evolution: a dynamic approach},
  author={Louren{\c{c}}o, Nuno and Assun{\c{c}}{\~a}o, Filipe and Pereira, Francisco B and Costa, Ernesto and Machado, Penousal},
  booktitle={Handbook of Grammatical Evolution},
  pages={137--161},
  year={2018},
  publisher={Springer}
}
```

This project corresponds to a new implementation of the SGE engine. SGE has been criticised for the fact that we need to specify the maximum levels of recursion in order to remove it from the grammar beforehand. In this new version we specify the maximum tree depth (similarly to what happens in standard tree-based GP), and the algorithm adds the mapping numbers as required during the evolutionary search. Thus, we do not need to pre-process the grammar to remove the recursive productions. Additionally, we provide mechanisms and operators to ensure that the generated trees are always within the allowed limits.


As in for the SGE framework we provide the implementations of some problems that we used to test the DSGE. Extending it to your own needs should be fairly easy. 


When running the framework a folder called *dumps* will be created together with an additional one that corresponds
 to the experience. Inside, there will be directories for each run. Each run folder contains snapshots of the
  population at a given generation, and a file called *progress_report.csv*,  which is updated during the
   evolutionary run
  . By default we take snapshots
   of the population every iteration (SAVESTEP parameter in the configuration file). This can be changed, together with
    all
    the numeric values in the
    *configs
   * folder.

### Requirements
Currently this codebase only works with python 3. 

### Installation

To be completed

### Execution

You can run the Symbolic Regression example like this:

```python -m examples.symreg --experiment_name dumps/example --seed 791021 --parameters parameters/standard.yml```

# caso original
python -m problems.LIB.CI.coef_predict --experiment_name=results/cdrag_original --parameters=parameters/LIB/CI/cdrag.yml --algorithm=SGE --seg=0 --coef=cdrag

# caso 1
python -m problems.LIB.CI.coef_predict --experiment_name=results/cdrag_1 --parameters=parameters/LIB/CI/cdrag.yml --algorithm=SGE --seg=1 --coef=cdrag
python -m problems.LIB.CI.coef_predict --experiment_name=results/cdrag_2 --parameters=parameters/LIB/CI/cdrag.yml --algorithm=SGE --seg=2 --coef=cdrag
python -m problems.LIB.CI.coef_predict --experiment_name=results/cdrag_3 --parameters=parameters/LIB/CI/cdrag.yml --algorithm=SGE --seg=3 --coef=cdrag

# caso 2
python -m problems.LIB.CI.coef_predict --experiment_name=results/cdrag_laminar_1 --parameters=parameters/LIB/CI/cdrag.yml --algorithm=SGE --seg=1 --coef=cdrag --choice=2
python -m problems.LIB.CI.coef_predict --experiment_name=results/cdrag_turbulento_2 --parameters=parameters/LIB/CI/cdrag.yml --algorithm=SGE --seg=2 --coef=cdrag --choice=2

# caso 3
python -m problems.LIB.CI.coef_predict --experiment_name=results/cdrag_inspeccion_1 --parameters=parameters/LIB/CI/cdrag.yml --algorithm=SGE --seg=1 --coef=cdrag --choice=3
python -m problems.LIB.CI.coef_predict --experiment_name=results/cdrag_inspeccion_2 --parameters=parameters/LIB/CI/cdrag.yml --algorithm=SGE --seg=2 --coef=cdrag --choice=3


# caso original
python -m problems.LIB.CI.coef_predict --experiment_name=results/ff_original --parameters=parameters/LIB/CI/ff.yml --algorithm=SGE --seg=0 --coef=ff

# caso 1
python -m problems.LIB.CI.coef_predict --experiment_name=results/ff_1 --parameters=parameters/LIB/CI/ff.yml --algorithm=SGE --seg=1 --coef=ff
python -m problems.LIB.CI.coef_predict --experiment_name=results/ff_2 --parameters=parameters/LIB/CI/ff.yml --algorithm=SGE --seg=2 --coef=ff
python -m problems.LIB.CI.coef_predict --experiment_name=results/ff_3 --parameters=parameters/LIB/CI/ff.yml --algorithm=SGE --seg=3 --coef=ff

# caso 2
python -m problems.LIB.CI.coef_predict --experiment_name=results/ff_laminar_1 --parameters=parameters/LIB/CI/ff.yml --algorithm=SGE --seg=1 --coef=ff --choice=2
python -m problems.LIB.CI.coef_predict --experiment_name=results/ff_turbulento_2 --parameters=parameters/LIB/CI/ff.yml --algorithm=SGE --seg=2 --coef=ff --choice=2

# caso 3
python -m problems.LIB.CI.coef_predict --experiment_name=results/ff_inspeccion_1 --parameters=parameters/LIB/CI/cdrag.yml --algorithm=SGE --seg=1 --coef=cdrag --choice=3
python -m problems.LIB.CI.coef_predict --experiment_name=results/ff_inspeccion_2 --parameters=parameters/LIB/CI/cdrag.yml --algorithm=SGE --seg=2 --coef=cdrag --choice=3


# caso original
python -m problems.LIB.CI.coef_predict --experiment_name=results/n_original --parameters=parameters/LIB/CI/n.yml --algorithm=SGE --seg=0 --coef=n

# caso 1
python -m problems.LIB.CI.coef_predict --experiment_name=results/n_1 --parameters=parameters/LIB/CI/n.yml --algorithm=SGE --seg=1 --coef=n
python -m problems.LIB.CI.coef_predict --experiment_name=results/n_2 --parameters=parameters/LIB/CI/n.yml --algorithm=SGE --seg=2 --coef=n
python -m problems.LIB.CI.coef_predict --experiment_name=results/n_3 --parameters=parameters/LIB/CI/n.yml --algorithm=SGE --seg=3 --coef=n

# caso 2
python -m problems.LIB.CI.coef_predict --experiment_name=results/n_laminar_1 --parameters=parameters/LIB/CI/n.yml --algorithm=SGE --seg=1 --coef=n --choice=2
python -m problems.LIB.CI.coef_predict --experiment_name=results/n_turbulento_2 --parameters=parameters/LIB/CI/n.yml --algorithm=SGE --seg=2 --coef=n --choice=2

# caso 3
python -m problems.LIB.CI.coef_predict --experiment_name=results/n_inspeccion_1 --parameters=parameters/LIB/CI/cdrag.yml --algorithm=SGE --seg=1 --coef=cdrag --choice=3
python -m problems.LIB.CI.coef_predict --experiment_name=results/n_inspeccion_2 --parameters=parameters/LIB/CI/cdrag.yml --algorithm=SGE --seg=2 --coef=cdrag --choice=3


### Support

Any questions, comments or suggestion should be directed to Nuno Lourenço ([naml@dei.uc.pt](mailto:naml@dei.uc.pt))

### Acknowledgments

I am grateful to my advisors Francisco B. Pereira and Ernesto Costa for their guidance during my PhD. I am also grateful to Filipe Assunção and Joaquim Ferrer for their help and comments on the development of this framework. 
