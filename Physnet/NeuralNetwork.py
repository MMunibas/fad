import tensorflow as tf
from .layers.RBFLayer import *
from .layers.InteractionBlock import *
from .layers.OutputBlock      import *
from .activation_fn import *
from .grimme_d3.grimme_d3 import *

def softplus_inverse(x):
    '''numerically stable inverse of softplus transform'''
    return x + np.log(-np.expm1(-x))

class PhysNet_v2(tf.keras.Model):
    def __str__(self):
        return "Neural Network"

    def __init__(self,
                 # Dimensionality of feature vector
                 F,
                 # Number of radial basis functions
                 K,
                 # Cutoff distance for short range interactions
                 sr_cut,
                 # Cutoff distance for long range interactions 
                 # (default: no cutoff)
                 lr_cut = None,
                 # Number of building blocks to be stacked
                 num_blocks=3,
                 # Number of residual layers for atomic refinements of 
                 # feature vector
                 num_residual_atomic=2,
                 # Number of residual layers for refinement of message vector
                 num_residual_interaction=2,
                 # Number of residual layers for the output blocks
                 num_residual_output=1,
                 # Adds electrostatic contributions to atomic energy
                 use_electrostatic=True,
                 # Adds dispersion contributions to atomic energy
                 use_dispersion=True,
                 # Adds QM/MM electrostatic contribution
                 use_qmmm=False,
                 # Cutoff distance for QM/MM electrostatic interactions
                 r_qmmm=None,
                 # Cutoff width for QM/MM electrostatic interactions
                 width_qmmm=None,
                 # QM/MM orthorhombic cell parameter
                 cell=None,
                 # QM/MM cell periodic boundary conditions
                 pbc=None,
                 # Fixed QM charges
                 qmcharges=None,
                 # s6 coefficient for d3 dispersion, by default is learned
                 s6=None,
                 # s8 coefficient for d3 dispersion, by default is learned
                 s8=None,
                 # a1 coefficient for d3 dispersion, by default is learned
                 a1=None,
                 # a2 coefficient for d3 dispersion, by default is learned   
                 a2=None,
                 # Initial value for output energy shift 
                 # (makes convergence faster)
                 Eshift=0.0,
                 # Initial value for output energy scale 
                 # (makes convergence faster)
                 Escale=1.0,
                 # Initial value for output charge shift 
                 Qshift=0.0,
                 # Initial value for output charge scale 
                 Qscale=1.0,
                 # Half (else double counting) of the Coulomb constant 
                 # (default is in units e=1, eV=1, A=1)
                 kehalf=7.199822675975274,
                 # Activation function
                 activation_fn=shifted_softplus, 
                 # Single or double precision
                 dtype=tf.float32, 
                 # Random seed
                 seed=None,
                 # Model name
                 name="PhysNet",
                 # Further Keras variables
                 **kwargs):
        
        super(PhysNet_v2, self).__init__(name=name, **kwargs)
        
        assert(num_blocks > 0)
        self._num_blocks = num_blocks
        self._dtype = dtype
        self._kehalf = kehalf
        self._F = F
        self._K = K
        self._sr_cut = sr_cut #cutoff for neural network interactions
        self._lr_cut = lr_cut #cutoff for long-range interactions
        self._use_electrostatic = use_electrostatic
        self._use_dispersion = use_dispersion
        self._use_qmmm = use_qmmm
        self._activation_fn = activation_fn
        
        # Probability for dropout regularization
        self._rate = tf.constant(0.0, shape=[], name="rate")

        # Atom embeddings (we go up to Pu(94): 95 - 1 ( for index 0))
        self._embeddings = tf.Variable(
            tf.random.uniform(
                [95, self.F], minval=-np.sqrt(3), maxval=np.sqrt(3), 
                seed=seed, dtype=dtype), 
            trainable=True, name="embeddings", dtype=dtype)
        
        tf.summary.histogram("embeddings", self.embeddings)

        # Radial basis function expansion layer
        self._rbf_layer = RBFLayer(K, sr_cut, scope="rbf_layer")
        
        # Initialize variables for d3 dispersion (the way this is done, 
        # positive values are guaranteed)
        if s6 is None:
            self._s6 = tf.nn.softplus(tf.Variable(
                initial_value=softplus_inverse(d3_s6), trainable=True, 
                name="s6", dtype=dtype))
        else:
            self._s6 = tf.Variable(
                initial_value=s6, trainable=True, 
                name="s6", dtype=dtype)
        tf.summary.scalar("d3-s6", self.s6)
        
        if s8 is None:
            self._s8 = tf.nn.softplus(tf.Variable(
                initial_value=softplus_inverse(d3_s8), trainable=True, 
                name="s8", dtype=dtype))
        else:
            self._s8 = tf.Variable(
                initial_value=s8, trainable=True, 
                name="s8", dtype=dtype)
        tf.summary.scalar("d3-s8", self.s8)
        
        if a1 is None:
            self._a1 = tf.nn.softplus(tf.Variable(
                initial_value=softplus_inverse(d3_a1), trainable=True, 
                name="a1", dtype=dtype))
        else:
            self._a1 = tf.Variable(
                initial_value=a1, trainable=True, 
                name="a1", dtype=dtype)
        tf.summary.scalar("d3-a1", self.a1)
        
        if a2 is None:
            self._a2 = tf.nn.softplus(tf.Variable(
                softplus_inverse(d3_a2), trainable=True, 
                name="a2", dtype=dtype))
        else:
            self._a2 = tf.Variable(
                initial_value=a2, trainable=True, 
                name="a2", dtype=dtype)
        tf.summary.scalar("d3-a2", self.a2)
        
        # QM/MM parameters
        if self.use_qmmm:
            # Electrostatic interaction range
            if r_qmmm is not None:
                self._r_qmmm = tf.constant(r_qmmm, dtype=dtype)
            else:
                if self.lr_cut is not None:
                    self._r_qmmm = tf.constant(self.lr_cut, dtype=dtype)
                else:
                    self._r_qmmm = tf.constant(self.sr_cut, dtype=dtype)
            # Cutoff width
            if width_qmmm is not None:
                self._width_qmmm = tf.constant(width_qmmm, dtype=dtype)
            else:
                self._width_qmmm = tf.constant(1.0, dtype=dtype)
            # Periodic boundary conditions
            if pbc is not None:
                pbc_once = False
                cell_then = np.zeros(3)
                for ip, periodic in enumerate(pbc):
                    if periodic:
                        pbc_once = True
                        cell_then[ip] = cell[ip]
                    else:
                        cell_then[ip] = 100000.0
                self._pbc = tf.constant(pbc_once, dtype=tf.bool)
                self._cell = tf.constant(cell_then, dtype=dtype)
            else:
                self._pbc = tf.constant(False, dtype=tf.bool)
            # Fixed QM charges
            if qmcharges is not None:
                self._qmcharges = tf.constant(qmcharges, dtype=dtype)
            else:
                self._qmcharges = None
            
        # Initialize output scale/shift variables
        self._Eshift = tf.Variable(
            initial_value=tf.constant(Eshift, shape=[95], dtype=dtype),
            name="Eshift", dtype=dtype)
        self._Escale = tf.Variable(
            initial_value=tf.constant(Escale, shape=[95], dtype=dtype),
            name="Escale", dtype=dtype)
        self._Qshift = tf.Variable(
            initial_value=tf.constant(Qshift, shape=[95], dtype=dtype), 
            name="Qshift", dtype=dtype)
        self._Qscale = tf.Variable(
            initial_value=tf.constant(Qscale, shape=[95], dtype=dtype), 
            name="Qscale", dtype=dtype)

        # Embedding blocks and output layers
        self._interaction_block = []
        self._output_block = []
        
        for i in range(num_blocks):
            
            self.interaction_block.append(
                InteractionBlock(
                    F, K, num_residual_atomic, num_residual_interaction,
                    activation_fn=activation_fn, seed=seed, 
                    rate=self.rate, scope="interaction_block" + str(i), dtype=dtype))
            
            self.output_block.append(
                OutputBlock(
                    F, num_residual_output, activation_fn=activation_fn, 
                    seed=seed, rate=self.rate, scope="output_block" + str(i),
                    dtype=dtype))
                            
        # Save checkpoint to save/restore the model
        self._saver = tf.train.Checkpoint(model=self)
        
    #@tf.function
    def calculate_interatomic_distances(self, R, idx_i, idx_j, offsets=None):
        ''' Calculate interatomic distances '''
        
        Ri = tf.gather(R, idx_i)
        Rj = tf.gather(R, idx_j)
        if offsets is not None:
            Rj = Rj + offsets
        Dij = tf.sqrt(tf.nn.relu(tf.reduce_sum((Ri - Rj)**2, -1))) 
        # ReLU: y = max(0, x), prevent negative sqrt
        return Dij

    #@tf.function
    def calculate_qmmm_distances(self, Rqm, Rmm, idx_qm, idx_mm):
        ''' Calculate QM/MM distances '''
        
        Ri = tf.gather(Rqm, idx_qm)
        Rj = tf.gather(Rmm, idx_mm)
        
        dR = Ri - Rj
        
        #index_R = []
        #iexample = 0
        #for iR, dRi in enumerate(dR):
            #if tf.reduce_sum(tf.math.abs(tf.math.round(dRi/self.cell))) > 0.0:
                #iexample = iexample + 1
                #if iexample > 10:
                    #continue
                #print(dRi)
                #index_R.append(iR)
        
        if self.pbc:
            dR = dR - self.cell*tf.math.round(dR/self.cell)
        
        #for iR in index_R:
            #print(dR[iR])
        
        Dqmmm = tf.sqrt(tf.nn.relu(tf.reduce_sum(dR**2, -1)))
        # ReLU: y = max(0, x), prevent negative sqrt
        
        return Dqmmm

    #@tf.function
    def atomic_properties(
        self, Z, R, idx_i, idx_j, offsets=None, sr_idx_i=None, sr_idx_j=None,
        sr_offsets=None, Rmm=None, idx_qm=None, idx_mm=None):
        ''' Calculates the atomic energies, charges and distances 
            (needed if unscaled charges are wanted e.g. for loss function) '''
        
        with tf.name_scope("atomic_properties"):
        
            # Calculate distances (for long range interaction)
            Dij_lr = self.calculate_interatomic_distances(
                R, idx_i, idx_j, offsets=offsets)
            
            # Optionally, it is possible to calculate separate distances 
            # for short range interactions (computational efficiency)
            if sr_idx_i is not None and sr_idx_j is not None:
                Dij_sr = self.calculate_interatomic_distances(
                    R, sr_idx_i, sr_idx_j, offsets=sr_offsets)
            else:
                sr_idx_i = idx_i
                sr_idx_j = idx_j
                Dij_sr = Dij_lr
            
            # Calculate radial basis function expansion
            rbf = self.rbf_layer(Dij_sr)
            
            # Initialize feature vectors according to embeddings for 
            # nuclear charges
            x = tf.gather(self.embeddings, Z)
            
            # Apply blocks
            Ea = 0 #atomic energy 
            Qa = 0 #atomic charge
            nhloss = 0 #non-hierarchicality loss
            
            for i in range(self.num_blocks):
                x = self.interaction_block[i](x, rbf, sr_idx_i, sr_idx_j)
                out = self.output_block[i](x)
                Ea = Ea + out[:,0]
                Qa = Qa + out[:,1]
                # Compute non-hierarchicality loss
                out2 = out**2
                if i > 0:
                    nhloss = nhloss +tf.reduce_mean(
                        out2/(out2 + lastout2 + 1e-7))
                lastout2 = out2
        
            # Apply scaling/shifting
            Ea = (tf.gather(self.Escale, Z) * Ea + tf.gather(self.Eshift, Z))
                #+ 0*tf.reduce_sum(R, -1))
            # Last term necessary to guarantee no "None" in force evaluation
            Qa = tf.gather(self.Qscale, Z) * Qa + tf.gather(self.Qshift, Z)
            
            # Calculate atoms - point charge distances
            if self.use_qmmm and Rmm is not None:
                Dqmmm = self.calculate_qmmm_distances(
                    R, Rmm, idx_qm, idx_mm)
            else:
                Dqmmm = None
        
        return Ea, Qa, Dij_lr, nhloss, Dqmmm
    
    #@tf.function
    def energy_from_scaled_atomic_properties(
        self, Ea, Qa, Dij, Z, idx_i, idx_j, batch_seg,
        Dqmmm, Qmm, idx_qm, idx_mm):
        ''' Calculates the energy given the scaled atomic properties (in order 
            to prevent recomputation if atomic properties are calculated) '''
    
        with tf.name_scope("energy_from_atomic_properties"):
            
            if batch_seg is None:
                batch_seg = tf.zeros_like(Z)
            
            # Add electrostatic and dispersion contribution to atomic energy
            if self.use_electrostatic:
                Ea = Ea + self.electrostatic_energy_per_atom(
                    Dij, Qa, idx_i, idx_j)
            if self.use_dispersion:
                if self.lr_cut is not None:   
                    Ea = Ea + d3_autoev*edisp(
                        Z, Dij/d3_autoang, idx_i, idx_j,
                        s6=self.s6, s8=self.s8, a1=self.a1, a2=self.a2,
                        cutoff=self.lr_cut/d3_autoang)
                else:
                    Ea = Ea + d3_autoev*edisp(
                        Z, Dij/d3_autoang, idx_i, idx_j,
                        s6=self.s6, s8=self.s8, a1=self.a1, a2=self.a2)
            # Add QM/MM contribution to atomic energy
            if self.use_qmmm and Dqmmm is not None:
                Ea = Ea + self.electrostatic_energy_per_atom_to_point_charge(
                    Dqmmm, Qa, Qmm, idx_qm, idx_mm)
            
        
        return tf.squeeze(tf.math.segment_sum(Ea, batch_seg))

    #@tf.function
    def energy_from_atomic_properties(
        self, Ea, Qa, Dij, Z, idx_i, idx_j, Q_tot, batch_seg,
        Dqmmm, Qmm, idx_qm, idx_mm):
        ''' Calculates the energy given the atomic properties (in order to 
            prevent recomputation if atomic properties are calculated) '''
    
        with tf.name_scope("energy_from_atomic_properties"):
            
            if batch_seg is None:
                batch_seg = tf.zeros_like(Z)
                
            # Scale charges such that they have the desired total charge
            Qa = self.scaled_charges(Z, Qa, Q_tot, batch_seg)
                
        return self.energy_from_scaled_atomic_properties(
            Ea, Qa, Dij, Z, idx_i, idx_j, batch_seg, 
            Dqmmm, Qmm, idx_qm, idx_mm)

    #@tf.function
    def energy_and_forces_from_scaled_atomic_properties(
        self, Ea, Qa, Dij, Z, R, idx_i, idx_j, batch_seg,
        Dqmmm, Qmm, idx_qm, idx_mm):
        ''' Calculates the energy and forces given the scaled atomic atomic  
            properties (in order to prevent recomputation if atomic properties 
            are calculated '''
    
        with tf.name_scope("energy_and_forces_from_atomic_properties"):
            
            with tf.GradientTape() as tape:
                
                tape.watch(R)
                
                # Calculate distances again for force evaluation
                Dij = self.calculate_interatomic_distances(
                    R, idx_i, idx_j, offsets=offsets)
                
                energy = self.energy_from_scaled_atomic_properties(
                    Ea, Qa, Dij, Z, idx_i, idx_j, batch_seg)
                
                reduced_energy = tf.reduce_sum(energy)
                
            forces = -tape.gradient(reduced_energy, R)
            
        return energy, forces

    #@tf.function
    def energy_and_forces_from_atomic_properties(
        self, Ea, Qa, Dij, Z, R, idx_i, idx_j, Q_tot=None, batch_seg=None):
        ''' Calculates the energy and force given the atomic properties 
            (in order to prevent recomputation if atomic properties are
            calculated) '''
        
        with tf.name_scope("energy_and_forces_from_atomic_properties"):
            
            with tf.GradientTape() as tape:
                
                tape.watch(R)
                
                # Calculate distances again for force evaluation
                Dij = self.calculate_interatomic_distances(
                    R, idx_i, idx_j, offsets=offsets)
                
                energy = self.energy_from_atomic_properties(
                    Ea, Qa, Dij, Z, idx_i, idx_j, Q_tot, batch_seg)
                
                reduced_energy = tf.reduce_sum(energy)
            
            forces = -tape.gradient(reduced_energy, R)
        
        return energy, forces

    @tf.function
    def energy(
        self, Z, R, idx_i, idx_j, Q_tot=None, batch_seg=None, offsets=None,
        sr_idx_i=None, sr_idx_j=None, sr_offsets=None, 
        Rmm=None, Qmm=None, idx_qm=None, idx_mm=None):
        ''' Calculates the total energy (including electrostatic 
            interactions) '''
        
        with tf.name_scope("energy"):
            
            Ea, Qa, Dij, _, Dqmmm = self.atomic_properties(
                Z, R, idx_i, idx_j, offsets, sr_idx_i, sr_idx_j, sr_offsets,
                Rmm, idx_qm, idx_mm)
            
            energy = self.energy_from_atomic_properties(
                Ea, Qa, Dij, Z, idx_i, idx_j, Q_tot, batch_seg,
                Dqmmm, Qmm, idx_qm, idx_mm)
                
        return energy 

    @tf.function
    def energy_and_forces(
        self, Z, R, idx_i, idx_j, Q_tot=None, batch_seg=None, offsets=None,
        sr_idx_i=None, sr_idx_j=None, sr_offsets=None,
        Rmm=None, Qmm=None, idx_qm=None, idx_mm=None):
        ''' Calculates the total energy and forces (including electrostatic 
            interactions)'''
        
        with tf.name_scope("energy_and_forces"):
            
            with tf.GradientTape() as tape:
                
                if self.use_qmmm and Rmm is not None:
                    tape.watch([R, Rmm])
                else:
                    tape.watch(R)
                
                Ea, Qa, Dij, _, Dqmmm = self.atomic_properties(
                    Z, R, idx_i, idx_j, offsets, 
                    sr_idx_i, sr_idx_j, sr_offsets,
                    Rmm, idx_qm, idx_mm)
                
                energy = self.energy_from_atomic_properties(
                    Ea, Qa, Dij, Z, idx_i, idx_j, Q_tot, batch_seg,
                    Dqmmm, Qmm, idx_qm, idx_mm)
                
                reduced_energy = tf.reduce_sum(energy)
                
            if self.use_qmmm and Rmm is not None:
                
                forces, forces_mm = -tape.gradient(reduced_energy, [R, Rmm])
                return energy, forces, forces_mm
                
            else:
                
                forces = -tape.gradient(reduced_energy, R)
                return energy, forces
    
    @tf.function
    def energy_and_forces_and_atomic_properties(
        self, Z, R, idx_i, idx_j, Q_tot=None, batch_seg=None, offsets=None, 
        sr_idx_i=None, sr_idx_j=None, sr_offsets=None,
        Rmm=None, Qmm=None, idx_qm=None, idx_mm=None):
        ''' Calculates the total energy and forces (including electrostatic 
            interactions)'''
        
        with tf.name_scope("energy_and_forces"):
            
            with tf.GradientTape() as tape:
                
                if self.use_qmmm and Rmm is not None:
                    tape.watch([R, Rmm])
                else:
                    tape.watch(R)
                
                Ea, Qa, Dij, nhloss, Dqmmm = self.atomic_properties(
                    Z, R, idx_i, idx_j, offsets, 
                    sr_idx_i, sr_idx_j, sr_offsets,
                    Rmm, idx_qm, idx_mm)
                
                energy = self.energy_from_atomic_properties(
                    Ea, Qa, Dij, Z, idx_i, idx_j, Q_tot, batch_seg,
                    Dqmmm, Qmm, idx_qm, idx_mm)
                
                reduced_energy = tf.reduce_sum(energy)
                
            forces = -tape.gradient(reduced_energy, R)
            
            if self.use_qmmm and Rmm is not None:
                
                forces, forces_mm = -tape.gradient(reduced_energy, [R, Rmm])
                return energy, forces, forces_mm, Ea, Qa, nhloss
                
            else:
                
                forces = -tape.gradient(reduced_energy, R)
                return energy, forces, Ea, Qa, nhloss
        
    @tf.function
    def energy_and_forces_and_charges(
        self, Z, R, idx_i, idx_j, Q_tot=None, batch_seg=None, offsets=None, 
        sr_idx_i=None, sr_idx_j=None, sr_offsets=None,
        Rmm=None, Qmm=None, idx_qm=None, idx_mm=None):
        ''' Calculates the total energy and forces (including electrostatic 
            interactions)'''
        
        with tf.name_scope("energy_and_forces"):
            
            with tf.GradientTape() as tape:
                
                if self.use_qmmm and Rmm is not None:
                    tape.watch([R, Rmm])
                else:
                    tape.watch(R)
                
                Ea, Qa, Dij, nhloss, Dqmmm = self.atomic_properties(
                    Z, R, idx_i, idx_j, offsets, 
                    sr_idx_i, sr_idx_j, sr_offsets,
                    Rmm, idx_qm, idx_mm)
                
                energy = self.energy_from_atomic_properties(
                    Ea, Qa, Dij, Z, idx_i, idx_j, Q_tot, batch_seg,
                    Dqmmm, Qmm, idx_qm, idx_mm)
                
                reduced_energy = tf.reduce_sum(energy)
                
            if self.use_qmmm and Rmm is not None:
                
                [forces, forces_mm] = tape.gradient(reduced_energy, [R, Rmm])
                return energy, -forces, -forces_mm, Qa
                
            else:
                
                forces = -tape.gradient(reduced_energy, R)
                return energy, forces, Qa
            
    def scaled_charges(self, Z, Qa, Q_tot=None, batch_seg=None):
        ''' Returns scaled charges such that the sum of the partial atomic 
            charges equals Q_tot (defaults to 0) '''
        
        with tf.name_scope("scaled_charges"):
            
            if batch_seg is None:
                batch_seg = tf.zeros_like(Z)
            
            # Number of atoms per batch (needed for charge scaling)
            Na_per_batch = tf.math.segment_sum(
                tf.ones_like(batch_seg, dtype=self.dtype), batch_seg)
            
            if Q_tot is None: # Assume desired total charge zero if not given
                Q_tot = tf.zeros_like(Na_per_batch, dtype=self.dtype)
        
        # Return scaled charges (such that they have the desired total charge)
        return Qa + tf.gather(
            ((Q_tot-tf.math.segment_sum(Qa, batch_seg))/Na_per_batch), 
            batch_seg)

    def _switch(self, Dij):
        ''' Switch function for electrostatic interaction (switches between
            shielded and unshielded electrostatic interaction) '''
    
        cut = self.sr_cut/2
        x  = Dij/cut
        x3 = x*x*x
        x4 = x3*x
        x5 = x4*x
        
        return tf.where(Dij < cut, 6*x5-15*x4+10*x3, tf.ones_like(Dij))

    def electrostatic_energy_per_atom(self, Dij, Qa, idx_i, idx_j):
        ''' Calculates the electrostatic energy per atom for very small 
            distances, the 1/r law is shielded to avoid singularities '''
    
        # Gather charges
        Qi = tf.gather(Qa, idx_i)
        Qj = tf.gather(Qa, idx_j)
        
        # Calculate variants of Dij which we need to calculate
        # the various shileded/non-shielded potentials
        DijS = tf.sqrt(Dij*Dij + 1.0) #shielded distance
        
        # Calculate value of switching function
        switch = self._switch(Dij) #normal switch
        cswitch = 1.0-switch #complementary switch
        
        # Calculate shielded/non-shielded potentials
        if self.lr_cut is None: #no non-bonded cutoff
            
            Eele_ordinary = 1.0/Dij   #ordinary electrostatic energy
            Eele_shielded = 1.0/DijS  #shielded electrostatic energy
            
            # Combine shielded and ordinary interactions and apply prefactors 
            Eele = self.kehalf*Qi*Qj*(
                cswitch*Eele_shielded + switch*Eele_ordinary)
            
        else: #with non-bonded cutoff
            
            cut   = self.lr_cut
            cut2  = self.lr_cut*self.lr_cut
            
            Eele_ordinary = 1.0/Dij  +  Dij/cut2 - 2.0/cut
            Eele_shielded = 1.0/DijS + DijS/cut2 - 2.0/cut
            
            # Combine shielded and ordinary interactions and apply prefactors 
            Eele = self.kehalf*Qi*Qj*(
                cswitch*Eele_shielded + switch*Eele_ordinary)
            Eele = tf.where(Dij <= cut, Eele, tf.zeros_like(Eele))
            
        return tf.math.segment_sum(Eele, idx_i) 
    
    def _cutoff(self, Dqmmm):
        ''' Switch function for electrostatic interaction (switches between
            shielded and unshielded electrostatic interaction) '''
        
        x  = (Dqmmm - self.r_qmmm + self.width_qmmm)/self.width_qmmm
        x3 = x*x*x
        x4 = x3*x
        x5 = x4*x
        
        cutoff = tf.where(
            Dqmmm < self.r_qmmm, tf.ones_like(Dqmmm), tf.zeros_like(Dqmmm))
        
        cutoff = tf.where(
            tf.logical_and(
                Dqmmm > self.r_qmmm - self.width_qmmm, Dqmmm < self.r_qmmm), 
            1-6*x5+15*x4-10*x3, cutoff)
        
        return cutoff

    def electrostatic_energy_per_atom_to_point_charge(
        self, Dqmmm, Qa, Qmm, idx_qm, idx_mm):
        ''' Calculate electrostatic interaction between QM atom charge and 
            MM point charge based on shifted Coulomb potential scheme'''
        
        # Atomic and point charges
        if self.qmcharges is None:
            Qi = tf.gather(Qa, idx_qm)
        else:
            Qi = tf.gather(self.qmcharges, idx_qm)
        Qj = tf.gather(Qmm, idx_mm)
        
        # Cutoff weighted reciprocal distance
        cutoff = self._cutoff(Dqmmm)
        rec_d = cutoff/Dqmmm
        
        # Shifted Coulomb energy
        QQ = 2.0*self.kehalf*Qi*Qj
        Eqmmm = QQ/Dqmmm - QQ/self.r_qmmm*(2.0 - Dqmmm/self.r_qmmm)
        
        return tf.math.segment_sum(cutoff*Eqmmm, idx_qm) 

    def save(self, save_file):
        ''' Save the current model '''
        self.saver.write(save_file)
        
    def restore(self, load_file):
        ''' Load a model '''
        self.saver.read(load_file).assert_consumed()
    
    def restore_v1(self, load_file):
        ''' Load a model v1 '''
        
        checkpoint = tf.train.load_checkpoint(load_file)
        chkvars = tf.train.list_variables(load_file)
        chkvarnames = [var[0] for var in chkvars]
        ii = 0
        iiall = len(chkvarnames)
        for var in self.trainable_variables:
            
            varname = var.name
            var.assign(checkpoint.get_tensor(varname))
            
            if varname in chkvarnames:
                #print(True)
                ii += 1
                chkvarnames.remove(varname)
           
        varname = "rbf_layer/centers:0"
        if varname in chkvarnames:
            
            self._rbf_layer._centers = tf.nn.softplus(
                checkpoint.get_tensor(varname))
            ii += 1
            chkvarnames.remove(varname)
                
        varname = "rbf_layer/widths:0"
        if varname in chkvarnames:
            
            self._rbf_layer._widths = tf.nn.softplus(
                checkpoint.get_tensor(varname))
            ii += 1
            chkvarnames.remove(varname)
                
        #print(ii, "/", iiall)
        #print("Not loaded:", chkvarnames)
        
            
    @property
    def rate(self):
        return self._rate
    
    @property
    def num_blocks(self):
        return self._num_blocks

    @property
    def dtype(self):
        return self._dtype

    @property
    def saver(self):
        return self._saver

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def Eshift(self):
        return self._Eshift

    @property
    def Escale(self):
        return self._Escale
  
    @property
    def Qshift(self):
        return self._Qshift

    @property
    def Qscale(self):
        return self._Qscale

    @property
    def s6(self):
        return self._s6

    @property
    def s8(self):
        return self._s8
    
    @property
    def a1(self):
        return self._a1

    @property
    def a2(self):
        return self._a2

    @property
    def use_electrostatic(self):
        return self._use_electrostatic

    @property
    def use_dispersion(self):
        return self._use_dispersion

    @property
    def use_qmmm(self):
        return self._use_qmmm

    @property
    def r_qmmm(self):
        return self._r_qmmm

    @property
    def width_qmmm(self):
        return self._width_qmmm

    @property
    def cell(self):
        return self._cell

    @property
    def pbc(self):
        return self._pbc

    @property
    def qmcharges(self):
        return self._qmcharges

    @property
    def kehalf(self):
        return self._kehalf

    @property
    def F(self):
        return self._F

    @property
    def K(self):
        return self._K

    @property
    def sr_cut(self):
        return self._sr_cut

    @property
    def lr_cut(self):
        return self._lr_cut
    
    @property
    def activation_fn(self):
        return self._activation_fn
    
    @property
    def rbf_layer(self):
        return self._rbf_layer

    @property
    def interaction_block(self):
        return self._interaction_block

    @property
    def output_block(self):
        return self._output_block
