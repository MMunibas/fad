import tensorflow as tf
from .NeuronLayer      import *
from .InteractionLayer import *
from .ResidualLayer    import *

class InteractionBlock(NeuronLayer):
    
    def __str__(self):
        return "interaction_block"+super().__str__()

    def __init__(self, K, F, num_residual_atomic, num_residual_interaction,
                 activation_fn=None, seed=None, rate=0.0, dtype=tf.float32):
        
        super().__init__(K, F, activation_fn)
         
        # Interaction layer
        self._interaction = InteractionLayer(
            K, F, num_residual_interaction, activation_fn=activation_fn, 
            seed=seed, rate=rate, dtype=dtype)

        # Residual layers
        self._residual_layer = []
        for i in range(num_residual_atomic):
            self._residual_layer.append(ResidualLayer(
                F, F, activation_fn, seed=seed, rate=rate, dtype=dtype))

    @property
    def interaction(self):
        return self._interaction
    
    @property
    def residual_layer(self):
        return self._residual_layer

    def __call__(self, x, rbf, idx_i, idx_j):
        
        x = self.interaction(x, rbf, idx_i, idx_j)
        
        for i in range(len(self.residual_layer)):
            x = self.residual_layer[i](x)
        
        return x
