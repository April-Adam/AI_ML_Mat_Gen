Readme v 1.0

# Graph neural network based application on  Quantum Chemistry
Inspired from https://arxiv.org/pdf/1704.01212

## DataSet QM9
http://quantum-machine.org/datasets/
https://www.nature.com/articles/sdata201422

QM9 is a dataset of 134,000 molecules consisting of 9 heavy atoms drawn from the elements C, H, O, N, F. The features are the xyz coordinates  and elements of the molecule. The coordinates are determined from B3LYP/6-31G(2df,p) level DFT geometry optimization. There are multiple labels (see table below), including the energy of formation (Enthalpy at 298.15 K).

The labels include:
| Index | Name | Units | Description | 
| ------|------- | ----|--------- |
| 0 | index | - | Consecutive, 1-based integer identifier of molecule |
| 1| A | GHz | Rotational constant A |
| 2| B | GHz | Rotational constant B |
| 3| C | GHz | Rotational constant C |
| 4| mu | Debye | Dipole moment |
| 5| aplha | Bohr^3 | Isotropic polarizability |
| 6| homo | Hartree | Energy of Highest occupied molecular orbital (HOMO) |
| 7| lumo | Hartree | Energy of Lowest unoccupied molecular orbital (LUMO) |
| 8| gap | Hartree | Gap, difference between LUMO and HOMO |
| 9| r2 | Bohr^2 | Electronic spatial extent |
| 10| zpve | Hartree | Zero point vibrational energy |
| 11| Uo | Hartree | Internal energy at 0 K |
| 12| U | Hartree | Internal energy at 298.15 K |
| 13| H | Hartree | Enthalpy at 298.15 K |
| 14| G | Hartree | Free energy at 298.15 K |
| 15| Cv | cal/(mol K) | Heat capacity at 298.15 K |

Following explanatino come from https://arxiv.org/pdf/1704.01212 
"First, we have four properties related to how tightly bound together the atoms in a molecule
are. These measure the energy required to break up the molecule at different temperatures and pressures. These include the atomization energy at 0K, U0 (eV), atomization energy at room temperature, U (eV), enthalpy of atomization at room temperature, H (eV), and free energy of atomization, G (eV).
Next there are properties related to fundamental vibrations of the molecule, including the highest fundamental vibrational frequency ω1 (cm−1) and the zero point vibrational energy (ZPVE) (eV). Additionally, there are a number of properties that concern the states of the electrons in the molecule. They include the energy of the electron in the highest occupied molecular orbital (HOMO) εHOMO (eV), the energy of the lowest unoccupied molecular orbital (LUMO) εLUMO (eV), and the electron energy gap (∆ε (eV)). The electron energy gap is simply the difference εHOMO − εLUMO.
Finally, there are several measures of the spatial distribution of electrons in the molecule. These include the electronic spatial extent hR2 i (Bohr2), the norm of the dipole
moment µ (Debye), and the norm of static polarizability α (Bohr3)."

## Problem statement
To desgin AI Neural network to prdicte QM9 material properites (indicated by label values in above table) based its atoms and corrdinations.

![image info](./docs/Pred.png)

## Neural Network 
### Convolution Neural Network 
This network consists of a fully connected layer. Atom information is encoded using one-hot encoding mechnaism. For materials in QM9, there are maximum 32 atoms of 16 different elements. After encoding, the input is in format of 32x16.
Notes: atom corrdination information is not used.

![image info](./docs/fc_net.png)

The model construction is done as following:
```
optimizer = keras.optimizers.RMSprop(learning_rate=0.0001)
def simple_keras_model():
    model = keras.Sequential()
    model.add(keras.layers.Lambda(lambda x: keras.backend.sum(x, axis=1,keepdims=False)))
    model.add(keras.layers.Dense(16, use_bias=True))
    model.compile(loss='mean_squared_error', optimizer = optimizer, metrics=['mae', r2_keras])
    return model
```

![image info](./docs/fc_net.png)

Training:

MAE is measured on 100 epochs

![image info](./docs/fc_qm9_mae.png)


```
Epoch 100/100
157/157 [==============================] - 0s 687us/step - loss: 0.1879 - mae: 0.1002 - r2_keras: 0.8083
```
[7.412908554077148, 0.10898412764072418, 0.6476361155509949]

### Graph Neural Network 

![image info](./docs/gnn_net.png)

epoch 100/100
157/157 [==============================] - 1s 5ms/step - loss: 0.1255 - mae: 0.1145 - r2_keras: 0.8714


[0.12728539109230042, 0.11673972755670547, 0.7171451449394226]



![image info](./docs/gnn_qm9_mae.png)
