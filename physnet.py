import tensorflow as tf
import numpy as np
import argparse

import ase
from ase.calculators.calculator import Calculator
from ase.neighborlist import neighbor_list
import ase.units as units

from .Physnet_v2.NeuralNetwork import *
from .Physnet_v2.activation_fn import *

'''
Calculator for the atomic simulation environment (ASE)
that evaluates energies and forces using a neural network
'''
class PhysNet(Calculator):
    
    # Properties provided by calculator
    implemented_properties = ['energy', 'forces', 'charges']
    
    def __init__(self,
                 # ASE atoms object
                 atoms,
                 # Checkpoint file to restore the model 
                 # (can also be a list for ensembles)
                 checkpoint,
                 # Respective config file for PhysNet architecture
                 config,
                 # System charge
                 charge=0,
                 # Checkpoint file fron PhysNet v1?
                 v1=False,
                 # Cutoff distance for long range interactions 
                 # (default: no cutoff)
                 lr_cut = None,
                 # Activate QM/MM mode
                 qmmm=False,
                 # Number of MM charges
                 Nmmcharges=None,
                 # Fixed charges of QM atoms for interaction potential
                 qmcharges=None,
                 # Interaction range for PointChargePotential
                 rc = None,
                 # Cutoff width
                 width=1.0,
                 # Umbrella mode
                 umbrella=False,
                 # Umbrella mode: harmonic force constant
                 umb_k=None,
                 # Umbrella mode: reactive coordinate minimum position
                 umb_r=None,
                 # Activation function
                 activation_fn=shifted_softplus,
                 # Single or double precision
                 dtype=tf.float32):
        
        # Read config file to ensure same PhysNet architecture as during fit
        # Initiate parser
        parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
        
        # Add arguments
        parser.add_argument("--restart", type=str, default=None)
        parser.add_argument("--num_features", default=128, type=int)
        parser.add_argument("--num_basis", default=64, type=int)
        parser.add_argument("--num_blocks", default=5, type=int)
        parser.add_argument("--num_residual_atomic", default=2, type=int)
        parser.add_argument("--num_residual_interaction", default=3, type=int)
        parser.add_argument("--num_residual_output", default=1, type=int)
        parser.add_argument("--cutoff", default=10.0, type=float)
        parser.add_argument("--use_electrostatic", default=1, type=int)
        parser.add_argument("--use_dispersion", default=1, type=int)
        parser.add_argument("--grimme_s6", default=None, type=float)
        parser.add_argument("--grimme_s8", default=None, type=float)
        parser.add_argument("--grimme_a1", default=None, type=float)
        parser.add_argument("--grimme_a2", default=None, type=float)
        parser.add_argument("--dataset", type=str)
        parser.add_argument("--num_train", type=int)
        parser.add_argument("--num_valid", type=int)
        parser.add_argument("--seed", default=None, type=int)
        parser.add_argument("--max_steps", default=10000, type=int)
        parser.add_argument("--learning_rate", default=0.001, type=float)
        parser.add_argument("--max_norm", default=1000.0, type=float)
        parser.add_argument("--ema_decay", default=0.999, type=float)
        parser.add_argument("--rate", default=0.0, type=float)
        parser.add_argument("--l2lambda", default=0.0, type=float)
        parser.add_argument("--nhlambda", default=0.1, type=float)
        parser.add_argument("--decay_steps", default=1000, type=int)
        parser.add_argument("--decay_rate", default=0.1, type=float)
        parser.add_argument("--batch_size", type=int)
        parser.add_argument("--valid_batch_size", type=int)
        parser.add_argument("--force_weight", default=52.91772105638412)
        parser.add_argument("--charge_weight", default=14.399645351950548,
                            type=float)
        parser.add_argument("--dipole_weight", default=27.211386024367243,
                            type=float)
        parser.add_argument("--summary_interval", default=5, type=int)
        parser.add_argument("--validation_interval", default=5, type=int)
        parser.add_argument("--show_progress", default=True, type=bool)
        parser.add_argument("--save_interval", default=5, type=int)
        parser.add_argument("--record_run_metadata", default=0, type=int)
        
        # Read config file
        args = parser.parse_args(["@" + config])
        
        # Create neighborlist
        if lr_cut is None:
            self.sr_cutoff = args.cutoff
            self.lr_cutoff = None
            self.use_neighborlist = False
        else:
            self.sr_cutoff = args.cutoff
            self.lr_cutoff = lr_cut
            self.use_neighborlist = True
        
        # QM/MM mode
        self.qmmm = qmmm
        
        # Periodic boundary conditions
        self.pbc = atoms.pbc
        self.cell = atoms.cell.diagonal()
        
        # Electrostatic cutoff -> half cell edge lengths if None
        if rc is None:
            self.rc = np.min(self.cell)/2.
        else:
            self.rc = rc
        # ... and width
        self.width = width
        
        # Umbrella mode
        self.umbrella = umbrella
        self.umb_k = umb_k
        self.umb_r = umb_r
        
        # Float data type
        self.dtype = dtype
        
        # Initiate calculator
        Calculator.__init__(self)
        
        # Set checkpoint file(s)
        self.checkpoint = checkpoint
        
        # Create PhysNet model
        self.model = PhysNet_v2(
            F=args.num_features,
            K=args.num_basis,
            sr_cut=self.sr_cutoff,
            lr_cut=self.lr_cutoff,
            num_blocks=args.num_blocks,                   
            num_residual_atomic=args.num_residual_atomic,          
            num_residual_interaction=args.num_residual_interaction,     
            num_residual_output=args.num_residual_output,          
            use_electrostatic=(args.use_electrostatic==1),
            use_dispersion=(args.use_dispersion==1),
            use_qmmm=self.qmmm,
            r_qmmm=self.rc,
            width_qmmm=self.width,
            cell=self.cell,
            pbc=self.pbc,
            qmcharges=qmcharges,
            s6=args.grimme_s6,
            s8=args.grimme_s8,
            a1=args.grimme_a1,
            a2=args.grimme_a2,
            activation_fn=activation_fn, 
            dtype=dtype, 
            name="PhysNet")
        
        # Initiate variables
        self.Z = tf.constant(
            atoms.get_atomic_numbers(), dtype=tf.int32, name="Z")
        self.R = tf.Variable(
            initial_value=atoms.get_positions(), trainable=False,
            name="R", dtype=dtype)
        self.Q_tot = tf.constant(
            [charge], dtype=dtype, name="Z")
        self.idx_i, self.idx_j = self.get_indices(atoms)
        if self.qmmm:
            self.idx_qm, self.idx_mm = self.get_qmmm_indices(atoms, Nmmcharges)
            
        # Initiate Embedded flag
        self.pcpot=None
        
        # Load neural network parameter
        if type(self.checkpoint) is not list:
            if v1:
                self.model.restore_v1(self.checkpoint)
            else:
                self.model.restore(self.checkpoint)
        
        # Calculate properties once to initialize everything
        self.last_atoms = None
        self.calculate(atoms)
        # Set last_atoms to None if pcpot get enabled later and recalculation 
        # becomes necessary again
        self.last_atoms = None
        
        Calculator.__init__(self)
        
    def get_indices(self, atoms):
        
        # Number of atoms
        N = len(atoms)
        
        idx = tf.range(N, dtype=tf.int32)
        # Indices for atom pairs ij - Atom i
        idx_i = tf.repeat(idx, N - 1, name="idx_i")
        # Indices for atom pairs ij - Atom j
        idx_j = tf.roll(idx, -1, axis=0, name="idx_j")
        if tf.math.greater_equal(N, 2):
            for Na in tf.range(2, N):
                idx_j = tf.concat(
                    [idx_j, tf.roll(idx, -Na, axis=0, name="idx_j")], 
                    axis=0)
        
        return idx_i, idx_j
    
    def get_qmmm_indices(self, atoms, Nmm):
        
        # Number of QM atoms
        Nqm = len(atoms)
        
        # Indices for QM/MM pair - qm atom
        idx_qm = tf.repeat(tf.range(Nqm, dtype=tf.int32), Nmm, name="idx_qm")
        # Indices for QM/MM pair - mm atom
        idx_mm = tf.tile(tf.range(Nmm, dtype=tf.int32), [Nqm], name="idx_qm")
        
        return idx_qm, idx_mm
        
    def calculation_required(self, atoms):
        
        # Check positions, atomic numbers, unit cell and pbc
        if self.last_atoms is None:
            return True
        else:
            return atoms != self.last_atoms
    
    def check_state(self, atoms, tol=1e-15):
        
        return self.calculation_required(atoms)
                
    def calculate(self, atoms, properties=None, system_changes=None):
        
        # Find neighbors and offsets
        if self.use_neighborlist:# or any(atoms.get_pbc()):
            
            idx_i, idx_j, S = neighbor_list('ijS', atoms, self.lr_cutoff)
            offsets = np.dot(S, atoms.get_cell())
            sr_idx_i, sr_idx_j, sr_S = neighbor_list(
                'ijS', atoms, self.sr_cutoff)
            sr_offsets = np.dot(sr_S, atoms.get_cell())
            
        else:
            
            idx_i = self.idx_i
            idx_j = self.idx_j
            offsets = None
            sr_idx_i = None
            sr_idx_j = None
            sr_offsets = None
            
        # Assign positions
        self.R.assign(atoms.get_positions())
        
        # Calculate energy, forces and atomic charges
        # (in case multiple NNs are used as ensemble, take the average)
        if(type(self.checkpoint) is not list):
            # Only one NN
            if self.qmmm and self.pcpot is not None:
                self.last_energy, self.last_forces, self.last_mmforces, \
                self.last_charges = \
                    self.model.energy_and_forces_and_charges(
                        self.Z, self.R, idx_i, idx_j, Q_tot=self.Q_tot,
                        batch_seg=None, offsets=offsets, 
                        sr_idx_i=sr_idx_i, sr_idx_j=sr_idx_j, 
                        sr_offsets=sr_offsets, Rmm=self.pcpot.mmpositions,
                        Qmm=self.pcpot.mmcharges, idx_qm=self.idx_qm, 
                        idx_mm=self.idx_mm)
            else:    
                self.last_energy, self.last_forces, self.last_charges = \
                    self.model.energy_and_forces_and_charges(
                        self.Z, self.R, idx_i, idx_j, Q_tot=self.Q_tot,
                        batch_seg=None, offsets=offsets, 
                        sr_idx_i=sr_idx_i, sr_idx_j=sr_idx_j, 
                        sr_offsets=sr_offsets)
                
            self._energy_stdev = 0
            
        else: 
            # Ensemble
            # Not done yet
            for i in range(len(self.checkpoint)):
                
                self.model.restore(self.checkpoint[i])
                
                self.last_energy, self.last_forces, self.last_charges = \
                    self.model.energy_and_forces_and_charges(
                        self.Z, self.R, idx_i, idx_j, Q_tot=self.Q_tot,
                        batch_seg=None, offsets=offsets, 
                        sr_idx_i=sr_idx_i, sr_idx_j=sr_idx_j, 
                        sr_offsets=sr_offsets)
                
                if i == 0:
                    
                    self.last_energy  = energy
                    self.last_forces  = forces
                    self.last_charges = charges
                    self.energy_stdev = 0
                    
                else:
                    
                    n = i + 1
                    delta = energy - self.last_energy
                    self.last_energy += delta/n
                    self.energy_stdev += delta*(energy - self.last_energy)
                    # Loop over atoms
                    for a in range(np.shape(charges)[0]):
                        self.last_charges[a] += (
                            charges[a] - self.last_charges[a])/n
                        for b in range(3):
                            self.last_forces[a,b] += (
                                forces[a,b] - self.last_forces[a,b])/n 
            
            if(len(self.checkpoint) > 1):
                
                self._energy_stdev = np.sqrt(
                    self.energy_stdev/len(self.checkpoint))
                    
        # Convert results to numpy array
        self.last_energy = self.last_energy.numpy()
        self.last_forces = tf.convert_to_tensor(self.last_forces).numpy()
        self.last_charges = self.last_charges.numpy()
        if self.qmmm and self.pcpot:
            self.last_mmforces = tf.convert_to_tensor(
                self.last_mmforces).numpy()
        
        # Add umbrella potential
        if self.umbrella:
            
            #      rHO11 rHO12 
            #      O- - H - -O
            #     /           \
            # H--C             C--H
            #     \           /
            #      O- - H - -O
            #      rHO21 rHO22 
            
            # Atom indices
            iH1 = 4
            iO11 = 2
            iO12 = 8
            iH2 = 9
            iO21 = 3
            iO22 = 7
            
            # Atom positions
            xH1 = atoms.positions[iH1]
            xO11 = atoms.positions[iO11]
            xO12 = atoms.positions[iO12]
            
            xH2 = atoms.positions[iH2]
            xO21 = atoms.positions[iO21]
            xO22 = atoms.positions[iO22]
            
            # Bond vectors
            vHO11 = xO11 - xH1
            vHO12 = xO12 - xH1
            vHO21 = xO21 - xH2
            vHO22 = xO22 - xH2
            
            # Bond lengths
            rHO11 = np.sqrt(np.sum(vHO11**2))
            rHO12 = np.sqrt(np.sum(vHO12**2))
            rHO21 = np.sqrt(np.sum(vHO21**2))
            rHO22 = np.sqrt(np.sum(vHO22**2))
            
            # Reactive coordinates
            reacor1 = rHO11 - rHO12
            reacor2 = rHO21 - rHO22
            
            # Umbrella potential
            # E = 0.5 * k * (r - r0)**2
            umb_E1 = 0.5*self.umb_k*(reacor1 - self.umb_r)**2
            umb_E2 = 0.5*self.umb_k*(reacor2 - self.umb_r)**2
            
            # Add umbrella potential
            self.last_energy += umb_E1 + umb_E2
            
            # Umbrella force
            # Fxi = -dEdxi = -(dEdr * drdxi)
            # dEdr = k * (r - r0)
            dE1dr = self.umb_k*(reacor1 - self.umb_r)
            dE2dr = self.umb_k*(reacor2 - self.umb_r)
            # drdxH = drHO1dxH - drHO2dxH
            # drHOjdxH = -vHOj/rHOj
            dr1dxH1 = (-vHO11/rHO11) - (-vHO12/rHO12)
            dr2dxH2 = (-vHO21/rHO21) - (-vHO22/rHO22)
            # drdxOj = vHOj/rHOj
            dr1dxO11 = vHO11/rHO11
            dr1dxO12 = -vHO12/rHO12
            dr2dxO21 = vHO21/rHO21
            dr2dxO22 = -vHO22/rHO22
            # Fxi
            umb_F1H1 = -dE1dr*dr1dxH1
            umb_F1O11 = -dE1dr*dr1dxO11
            umb_F1O12 = -dE1dr*dr1dxO12
            umb_F2H2 = -dE2dr*dr2dxH2
            umb_F2O21 = -dE2dr*dr2dxO21
            umb_F2O22 = -dE2dr*dr2dxO22
            
            # Add umbrella force
            self.last_forces[iH1] += umb_F1H1
            self.last_forces[iO11] += umb_F1O11
            self.last_forces[iO12] += umb_F1O12
            self.last_forces[iH2] += umb_F2H2
            self.last_forces[iO21] += umb_F2O21
            self.last_forces[iO22] += umb_F2O22
            
        # Store a copy of the atoms object
        self.last_atoms = atoms.copy()
    
        # Save properties
        self.results['energy'] = self.last_energy
        self.results['forces'] = self.last_forces
        self.results['charges'] = self.last_charges
        if self.qmmm and self.pcpot:
            self.results['mmforces'] = self.last_mmforces
        
    def get_potential_energy(self, atoms, force_consistent=False):
        
        if self.calculation_required(atoms):
            self.calculate(atoms)
        
        return self.results['energy']
    
    def get_forces(self, atoms):
        
        if self.calculation_required(atoms):
            self.calculate(atoms)
        
        return self.results['forces']
    
    def get_charges(self, atoms):
        
        if self.calculation_required(atoms):
            self.calculate(atoms)
        
        return self.results['charges']
    
    def get_forces_on_point_charges(self):
        
        return self.results['mmforces']
    
    def embed(self, charges):
        """Embed atoms in point-charges."""
        
        self.pcpot = PointChargePotential(charges, dtype=self.dtype)
        
        return self.pcpot
    
class PointChargePotential:
    """Point-charge potential for PhysNet"""
    
    def __init__(self, mmcharges, mmpositions=None, dtype=tf.float32):
        """Initiate parameters"""
        self.mmcharges = tf.constant(mmcharges, dtype=dtype)
        if mmpositions is not None:
            self.mmpositions = tf.Variable(
                mmpositions, trainable=False, name="Rmm", dtype=dtype)
        else:
            self.mmpositions = None
        self.mmforces = None
        self.dtype = dtype
        
    def set_positions(self, mmpositions):
        """Set the positions of point charges"""
        if self.mmpositions is None:
            self.mmpositions = tf.Variable(
                initial_value=mmpositions, trainable=False,
                name="Rmm", dtype=self.dtype)
        else:
            self.mmpositions.assign(tf.constant(mmpositions, dtype=self.dtype))
        
    def get_forces(self, calc):
        """Forces acting on point charges"""
        return calc.get_forces_on_point_charges()
