import numpy  as np
import tensorflow  as tf

class DataContainer:
    
    def __repr__(self):
         return "DataContainer"
    
    def __init__(
        self, filename, ntrain, nvalid, batch_size=1, valid_batch_size=1, 
        seed=None, dtype=tf.float32):
        
        # Read in data
        dictionary = np.load(filename)
        
        # Number of atoms
        if 'N' in dictionary: 
            self._N = lambda idx: tf.constant(
                dictionary['N'][idx], dtype=tf.int32)
            self._ndata = dictionary['N'].shape[0]
        else:
            raise IOError(
                'The information for the Atom Numbers N are essential')
        
        # Atomic numbers/nuclear charges
        if 'Z' in dictionary: 
            self._Z = lambda idx: tf.constant(
                dictionary['Z'][idx], dtype=tf.int32)
        else:
            raise IOError(
                'The information about the Atomic numbers Z are essential')
        
        # Positions (cartesian coordinates)
        if 'R' in dictionary:     
            self._R = lambda idx: tf.constant(
                dictionary['R'][idx], dtype=dtype)
        else:
            raise IOError(
                'The information about the Atomic positions R are essential')
        
        # Reference energy
        if 'E' in dictionary:
            self._E = lambda idx: tf.constant(
                dictionary['E'][idx], dtype=dtype)
            self._include_E = True
        else:
            self._E = lambda idx: None
            self._include_E = False
            
        # Reference atomic energies
        if 'Ea' in dictionary:
            self._Ea = lambda idx: tf.constant(
                dictionary['Ea'][idx], dtype=dtype)
            self._include_Ea = True
        else:
            self._Ea = lambda idx: None
            self._include_Ea = False
        
        # Reference forces
        if 'F' in dictionary:
            self._F = lambda idx: tf.constant(
                dictionary['F'][idx], dtype=dtype)
            self._include_F = True
        else:
            self._F = lambda idx: None
            self._include_F = False
        
        # Reference total charge
        if 'Q' in dictionary: 
            self._Q = lambda idx: tf.constant(
                dictionary['Q'][idx], dtype=dtype)
            self._include_Q = True
        else:
            self._Q = lambda idx: None
            self._include_Q = False
        
        # Reference atomic charges
        if 'Qa' in dictionary: 
            self._Qa = lambda idx: tf.constant(
                dictionary['Qa'][idx], dtype=dtype)
            self._include_Qa = True
        else:
            self._Qa = lambda idx: None
            self._include_Qa = False
        
        # Reference dipole moment vector
        if 'D' in dictionary: 
            self._D = lambda idx: tf.constant(
                dictionary['D'][idx], dtype=dtype)
            self._include_D = True
        else:
            self._D = lambda idx: None
            self._include_D = False
        
        # Assign parameters
        #self._ndata = self._N.shape[0]
        self._ntrain = ntrain
        self._nvalid = nvalid
        self._ntest = self.ndata - self.ntrain - self.nvalid
        self._batch_size = batch_size
        self._valid_batch_size = valid_batch_size
        self._dtype = dtype
        
        # Random state parameter for reproducible random operations
        self._random_state = np.random.RandomState(seed=seed)
        
        # Create shuffled list of indices
        idx = self._random_state.permutation(np.arange(self.ndata))
        
        # Store indices of training, validation and test data
        self._idx_train = idx[0:self.ntrain]
        self._idx_valid = idx[self.ntrain:self.ntrain+self.nvalid]
        self._idx_test  = idx[self.ntrain+self.nvalid:]
        
        # Initialize mean/stdev of properties
        self._EperA_mean  = None
        self._EperA_stdev = None
        self._FperA_mean  = None
        self._FperA_stdev = None
        self._DperA_mean  = None
        self._DperA_stdev = None
        
        # Create DataSet for training and valid data
        self._train_data = tf.data.Dataset.from_tensor_slices((
            self.N(self.idx_train), self.Z(self.idx_train), 
            self.R(self.idx_train), self.E(self.idx_train), 
            self.Ea(self.idx_train), self.F(self.idx_train), 
            self.Q(self.idx_train), self.Qa(self.idx_train), 
            self.D(self.idx_train)))
        
        self._valid_data = tf.data.Dataset.from_tensor_slices((
            self.N(self.idx_valid), self.Z(self.idx_valid), 
            self.R(self.idx_valid), self.E(self.idx_valid), 
            self.Ea(self.idx_valid), self.F(self.idx_valid), 
            self.Q(self.idx_valid), self.Qa(self.idx_valid), 
            self.D(self.idx_valid)))
        
    def get_train_batches(self, batch_size=None, seed=None):
        
        # Set batch custom size
        if batch_size is None:
            batch_size = self.batch_size
        
        # Shuffle training data and divide in batches
        train_batches = \
            self.train_data.shuffle(self.ntrain, seed=seed).batch(
                batch_size, drop_remainder=False)
        
        # Get number of batches
        N_train_batches = int(np.ceil(self.ntrain/batch_size))
        
        return train_batches, N_train_batches
    
    def get_valid_batches(self, batch_size=None):
        
        # Set batch custom size
        if batch_size is None:
            batch_size = self.valid_batch_size
        
        # Divide validation data into batches
        valid_batches = self.valid_data.batch(batch_size)
        
        return valid_batches
    
    def _compute_E_statistics(self):
        x = self.E(self.idx_train)/tf.cast(self.N(self.idx_train), self.dtype)
        self._EperA_mean = tf.reduce_sum(x, axis=0)/self.ntrain
        self._EperA_stdev = tf.reduce_sum((x - self.EperA_mean)**2, axis=0)
        self._EperA_stdev = tf.sqrt(self.EperA_stdev/self.ntrain)
        return
    
    def _compute_F_statistics(self):
        self._FperA_mean  = 0.0
        self._FperA_stdev = 0.0
        for i in range(self.ntrain):
            F = self.F(i)
            x = 0.0
            for j in range(self.N(i)):
                x = x + tf.sqrt(F[j][0]**2 + F[j][1]**2 + F[j][2]**2)
            m_prev = self.FperA_mean
            x = x/tf.cast(self.N(i), self.dtype)
            self._FperA_mean = (
                self.FperA_mean + (x - self.FperA_mean)/(i + 1))
            self._FperA_stdev = (
                self.FperA_stdev + (x - self.FperA_mean)*(x - m_prev))
        self._FperA_stdev = tf.sqrt(self.FperA_stdev/self.ntrain)
        return
    
    def _compute_D_statistics(self):
        self._DperA_mean  = 0.0
        self._DperA_stdev = 0.0
        for i in range(self.ntrain):
            D = self.D(i)
            x = tf.sqrt(D[0]**2 + D[1]**2 + D[2]**2)
            m_prev = self.DperA_mean
            self._DperA_mean = (
                self.DperA_mean + (x - self.DperA_mean)/(i + 1))
            self._DperA_stdev = (
                self.DperA_stdev + (x - self.DperA_mean)*(x - m_prev))
        self._DperA_stdev = tf.sqrt(self.DperA_stdev/self.ntrain)
        return
    
    @property 
    def EperA_mean(self):
        ''' Mean energy per atom in the training set '''
        if self._EperA_mean is None:
            self._compute_E_statistics()
        return self._EperA_mean

    @property
    def EperA_stdev(self): 
        ''' stdev of energy per atom in the training set '''
        if self._EperA_stdev is None:
            self._compute_E_statistics()
        return self._EperA_stdev
    
    @property 
    def FperA_mean(self): 
        ''' Mean force magnitude per atom in the training set '''
        if self._FperA_mean is None:
            self._compute_F_statistics()
        return self._FperA_mean

    @property
    def FperA_stdev(self): 
        ''' stdev of force magnitude per atom in the training set '''
        if self._FperA_stdev is None:
            self._compute_F_statistics()
        return self._FperA_stdev
    
    @property
    def DperA_mean(self): 
        ''' Mean partial charge per atom in the training set '''
        if self._DperA_mean is None:
            self._compute_D_statistics()
        return self._DperA_mean

    @property
    def DperA_stdev(self): 
        ''' stdev of partial charge per atom in the training set '''
        if self._DperA_stdev is None:
            self._compute_D_statistics()
        return self._DperA_stdev
        
    @property
    def N(self):
        return self._N

    @property
    def Z(self):
        return self._Z

    @property
    def R(self):
        return self._R

    @property
    def E(self):
        return self._E

    @property
    def include_E(self):
        return self._include_E

    @property
    def Ea(self):
        return self._Ea

    @property
    def include_Ea(self):
        return self._include_Ea

    @property
    def F(self):
        return self._F
    
    @property
    def include_F(self):
        return self._include_F

    @property
    def Q(self):
        return self._Q
    
    @property
    def include_Q(self):
        return self._include_Q

    @property
    def Qa(self):
        return self._Qa
    
    @property
    def include_Qa(self):
        return self._include_Qa

    @property
    def D(self):
        return self._D

    @property
    def include_D(self):
        return self._include_D

    @property
    def ndata(self):
        return self._ndata

    @property
    def ntrain(self):
        return self._ntrain
    
    @property
    def nvalid(self):
        return self._nvalid
    
    @property
    def ntest(self):
        return self._ntest

    @property
    def random_state(self):
        return self._random_state

    @property
    def idx_train(self):
        return self._idx_train

    @property
    def idx_valid(self):
        return self._idx_valid   

    @property
    def idx_test(self):
        return self._idx_test

    @property
    def valid_idx(self):
        return self._valid_idx

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def valid_batch_size(self):
        return self._valid_batch_size

    @property
    def dtype(self):
        return self._dtype
    
    @property
    def dtype(self):
        return self._dtype

    @property
    def train_data(self):
        return self._train_data
    
    @property
    def valid_data(self):
        return self._valid_data
    
