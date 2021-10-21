# PhysNet

Tensorflow implementation of PhysNet (see https://arxiv.org/abs/1902.08408 (OLD)) for details


## Requirements

To run this software, you need:

- python3 (tested with version 3.5.2 and higher)
- TensorFlow2 (tested with version 2.2 and higher)


## How to use - Training

Edit the config.txt file to specify hyperparameters, dataset location, training/validation set size etc.
(see "train.py" for a list of all options)

Then, simply run

```
python3 train.py 
```

in a terminal to start training. 

The included "config.txt" assumes that the dataset "sn2_reactions.npz" is present. It can be downloaded from: https://zenodo.org/record/2605341. In order to use a different dataset, it needs to be formatted in the same way as this example ("sn2_reactions.npz"). Please refer to the README file of the dataset (available from https://zenodo.org/record/2605341) for details.


## How to use - ASE Calculator

Import the ASE calculator PhysNetCalculator from PhysNet.py and link the path to the checkpoint files for the neural network parameters and the config.txt file for the PhysNet architecture. 

```
from physnet import PhysNet
calc = PhysNet(
    atoms=fad,					# Atoms object
    charge=0,					# Total charge of system
    checkpoint="./Final_Fit/best/best_model",	# Define just the name tag without .data-?????-of-????? prefix
    config="./Final_Fit/config.txt")
```

PhysNetCalculator contains a PointChargePotential class for QMMM calculation using the EIQMMM class of ASE. The electrostatic interactions are calculated as in the TIP3P model in ASE with the cutoff radii equivalent to the cutoff radii defined in PhysNet (default: width of 1.0 for the switch function to disable electrostatic interactions in the range 9.0 to 10.0)

Example for QM system (PhysNet) in Water (TIP3P), see https://wiki.fysik.dtu.dk/ase/tutorials/qmmm/qmmm.html for detailed instructions:

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
Unke, O. T. and Meuwly, M. "PhysNet: A Neural Network for Predicting Energies, Forces, Dipole Moments and Partial Charges" arxiv:1902.08408 (2019).
```


