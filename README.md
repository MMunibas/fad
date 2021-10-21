# PhysNet

Tensorflow implementation of PhysNet (see https://arxiv.org/abs/1902.08408) for details


## Requirements

To run this software, you need:

- python3 (tested with version 3.6 and higher)
- TensorFlow2 (tested with version 2.2 and higher)


## How to use - Training

Edit the config.txt file to specify hyperparameters, dataset location, training/validation set size etc.
(see "train.py" for a list of all options)

Then, simply run

```
python3 train.py 
```

in a terminal to start training. 

## How to use - ASE Calculator

Import the ASE calculator PhysNet from physnet.py and link the path to the checkpoint files for the neural network parameters and the config.txt file for the PhysNet architecture. 

```
from physnet import PhysNet
calc = PhysNet(
    atoms=fad,					# Atoms object
    charge=0,					# Total charge of system
    checkpoint="./Final_Fit/best/best_model",	# Define just the name tag without .data-?????-of-????? prefix
    config="./Final_Fit/config.txt")
```

The PhysNet calculator contains a PointChargePotential class for QMMM calculation using the EIQMMM class of ASE. The electrostatic potential is calculated via shifted electrostatic Coulomb interactions.

Example for QM system in Water, see https://wiki.fysik.dtu.dk/ase/tutorials/qmmm/qmmm.html for detailed instructions:

```
qm_calc = PhysNet(...)            # QM calulcator as defined above
mm_calc = TIP3P(rc=10.0)                    # MM calculator TIP3P with cutoff range of 10.0
interaction = LJInteractions(...)           # Interaction potential 

qmmm_calc = EIQMMM(
    qm_index,
    qm_calc,
    mm_calc,
    interaction)
```


## How to cite

If you find this software useful, please cite:

```
Unke, O. T. and Meuwly, M. "PhysNet: A Neural Network for Predicting Energies, Forces, Dipole Moments and Partial Charges", JCTC, 15, 3678-3693 (2019).
```


