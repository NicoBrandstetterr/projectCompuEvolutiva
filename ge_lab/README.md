# GE_LAB

## **Clone**

```bash
git https://github.com/rafadls/GE_LAB.git
cd GE_LAB
```

## **RUN**

```bash
python -m <problem module> --experiment_name=<name output folder> --parameters=<pareameters file> --algorithm=<GE algorithm>
```

## **Li-Ion Battery: equilibrium**
### **Drag coefficient**

```bash
python -m problems.LIB.CI.cdrag --experiment_name='results/cdrag' --parameters='parameters/LIB/CI/cdrag.yml' --algorithm='SGE'
```
```bash
python -m problems.LIB.CI.cdrag_25 --experiment_name=results/cdrag --parameters=parameters/LIB/CI/cdrag.yml --algorithm=SGE
```




### **Friction factor**

```bash
python -m problems.LIB.CI.ff --experiment_name='results/ff' --parameters='parameters/LIB/CI/ff.yml' --algorithm='SGE'
```

### **Nusselt number**

```bash
python -m problems.LIB.CI.n --experiment_name='results/n' --parameters='parameters/LIB/CI/n.yml' --algorithm='SGE'
```