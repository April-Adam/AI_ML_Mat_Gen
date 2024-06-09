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
### convolution neural network 


### graph neural network 