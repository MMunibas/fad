#!/usr/bin/env python3

import os
import sys
import numpy as np
import tensorflow as tf
import argparse
import logging
import string
import random

from shutil import copyfile
from datetime import datetime

from neural_network.NeuralNetwork import PhysNet
from neural_network.activation_fn import *

from DataContainer import DataContainer

from time import time

# Configure logging environment
logging.basicConfig(filename='train.log',level=logging.DEBUG)

#------------------------------------------------------------------------------
# Command line arguments
#------------------------------------------------------------------------------

# Initiate parser
parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

# Add arguments
parser.add_argument("--restart", type=str, default=None, 
                    help="Restart training from a specific folder")
parser.add_argument("--num_features", default=128, type=int, 
                    help="Dimensionality of feature vectors")
parser.add_argument("--num_basis", default=64, type=int, 
                    help="Number of radial basis functions")
parser.add_argument("--num_blocks", default=5, type=int, 
                    help="Number of interaction blocks")
parser.add_argument("--num_residual_atomic", default=2, type=int, 
                    help="Number of residual layers for atomic refinements")
parser.add_argument("--num_residual_interaction", default=3, type=int, 
                    help="Number of residual layers for the message phase")
parser.add_argument("--num_residual_output", default=1, type=int, 
                    help="Number of residual layers for the output blocks")
parser.add_argument("--cutoff", default=10.0, type=float, 
                    help="Cutoff distance for short range interactions")
parser.add_argument("--use_electrostatic", default=1, type=int, 
                    help="Use electrostatics in energy prediction (0/1)")
parser.add_argument("--use_dispersion", default=1, type=int, 
                    help="Use dispersion in energy prediction (0/1)")
parser.add_argument("--grimme_s6", default=None, type=float, 
                    help="Grimme s6 dispersion coefficient")
parser.add_argument("--grimme_s8", default=None, type=float, 
                    help="Grimme s8 dispersion coefficient")
parser.add_argument("--grimme_a1", default=None, type=float, 
                    help="Grimme a1 dispersion coefficient")
parser.add_argument("--grimme_a2", default=None, type=float, 
                    help="Grimme a2 dispersion coefficient")
parser.add_argument("--dataset", type=str, 
                    help="File path to dataset")
parser.add_argument("--num_train", type=int, 
                    help="Number of training samples")
parser.add_argument("--num_valid", type=int, 
                    help="Number of validation samples")
parser.add_argument("--batch_size", type=int, 
                    help="Batch size used per training step")
parser.add_argument("--valid_batch_size", type=int, 
                    help="Batch size used for going through validation_set")
parser.add_argument("--seed", default=np.random.randint(1000000), type=int, 
                    help="Seed for splitting dataset into " + \
                         "training/validation/test")
parser.add_argument("--max_steps", default=10000, type=int, 
                    help="Maximum number of training steps")
parser.add_argument("--learning_rate", default=0.001, type=float, 
                    help="Learning rate used by the optimizer")
parser.add_argument("--decay_steps", default=1000, type=int, 
                    help="Decay the learning rate every N steps by decay_rate")
parser.add_argument("--decay_rate", default=0.1, type=float, 
                    help="Factor with which the learning rate gets " + \
                         "multiplied by every decay_steps steps")
parser.add_argument("--max_norm", default=1000.0, type=float, 
                    help="Max norm for gradient clipping")
parser.add_argument("--ema_decay", default=0.999, type=float, 
                    help="Exponential moving average decay used by the " + \
                         "trainer")
parser.add_argument("--rate", default=0.0, type=float, 
                    help="Rate probability for dropout regularization of " + \
                         "rbf layer")
parser.add_argument("--l2lambda", default=0.0, type=float, 
                    help="Lambda multiplier for l2 loss (regularization)")
parser.add_argument("--nhlambda", default=0.1, type=float, 
                    help="Lambda multiplier for non-hierarchicality " + \
                         "loss (regularization)")
parser.add_argument('--force_weight', default=52.91772105638412, type=float,
                    help="This defines the force contribution to the loss " + \
                         "function relative to the energy contribution (to" + \
                         " take into account the different numerical range)")
parser.add_argument('--charge_weight', default=14.399645351950548, type=float,
                    help="This defines the charge contribution to the " + \
                         "loss function relative to the energy " + \
                         "contribution (to take into account the " + \
                         "different  numerical range)")
parser.add_argument('--dipole_weight', default=27.211386024367243, type=float,
                    help="This defines the dipole contribution to the " + \
                         "loss function relative to the energy " + \
                         "contribution (to take into account the " + \
                         "different numerical range)")
parser.add_argument('--summary_interval', default=5, type=int, 
                    help="Write a summary every N steps")
parser.add_argument('--validation_interval', default=5, type=int, 
                    help="Check performance on validation set every N steps")
parser.add_argument('--show_progress', default=True, type=bool, 
                    help="Show progress of the epoch")
parser.add_argument('--save_interval', default=5, type=int, 
                    help="Save progress every N steps")
parser.add_argument('--record_run_metadata', default=0, type=int, 
                    help="Records metadata like memory consumption etc.")

#------------------------------------------------------------------------------
# Read Parameters and define output files
#------------------------------------------------------------------------------

# Generate an (almost) unique id for the training session
def id_generator(size=8, 
                 chars=(string.ascii_uppercase 
                        + string.ascii_lowercase 
                        + string.digits)):
    
    return ''.join(random.SystemRandom().choice(chars) for _ in range(size))

# Read config file if no arguments are given
config_file='config.txt'
if len(sys.argv) == 1:
    if os.path.isfile(config_file):
        args = parser.parse_args(["@" + config_file])
    else:
        args = parser.parse_args(["--help"])
else:
    args = parser.parse_args()

# Create output directory for training session and 
# load config file arguments if restart
if args.restart is None:
    directory = (
        datetime.utcnow().strftime("%Y%m%d%H%M%S") 
        + "_" + id_generator() +"_F"+str(args.num_features)
        +"K"+str(args.num_basis)+"b"+str(args.num_blocks)
        +"a"+str(args.num_residual_atomic)
        +"i"+str(args.num_residual_interaction)
        +"o"+str(args.num_residual_output)+"cut"+str(args.cutoff)
        +"e"+str(args.use_electrostatic)+"d"+str(args.use_dispersion)
        +"l2"+str(args.l2lambda)+"nh"+str(args.nhlambda)
        +"rate"+str(args.rate))
else:
    directory=args.restart
    args = parser.parse_args(["@" + os.path.join(args.restart, config_file)])

# Create sub directories
logging.info("Creating directories...")

if not os.path.exists(directory):
    os.makedirs(directory)
    
best_dir = os.path.join(directory, 'best')
if not os.path.exists(best_dir):
    os.makedirs(best_dir)
    
log_dir = os.path.join(directory, 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Define output files
best_loss_file  = os.path.join(best_dir, 'best_loss.npz')
best_checkpoint = os.path.join(best_dir, 'best_model')

# Write config file of current training session
logging.info("Writing args to file...")

with open(os.path.join(directory, config_file), 'w') as f:
    for arg in vars(args):
        f.write('--'+ arg + '=' + str(getattr(args, arg)) + "\n")

#------------------------------------------------------------------------------
# Define utility functions
#------------------------------------------------------------------------------

def calculate_errors(val1, val2, weights=1.0):
    ''' Calculate error values and loss function '''
    
    # Value difference
    delta = tf.abs(val1 - val2)
    delta2 = delta**2
    
    # Mean absolute error
    mae = tf.reduce_mean(delta)
    # Mean squared error
    mse = tf.reduce_mean(delta2)
    
    # Loss function: mean absolute error
    loss = mae 
    
    return loss, mse, mae

def calculate_null(val1, val2, weights=1.0):
    ''' Return zero for error and loss values '''
    
    null = tf.constant(0.0, dtype=tf.float32)
    
    return null, null, null

def create_summary(dictionary):
    ''' Creates a summary from key-value pairs given by a dictionary '''
    
    # Initiate summary
    summary = tf.Summary()
    
    # Write key-value pairs to summary
    for key, value in dictionary.items():
        summary.value.add(tag=key, simple_value=value)
    
    return summary

def reset_averages():
    ''' Reset counter and average values '''
    
    null_float = tf.constant(
        0.0, name='averages', dtype=tf.float32)
    
    return null_float, null_float, null_float, null_float, null_float, \
        null_float, null_float, null_float, null_float, null_float
    
def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '#', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print("\r{0} |{1}| {2}% {3}".format(
        prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

#------------------------------------------------------------------------------
# Load data and initiate PhysNet model
#------------------------------------------------------------------------------

# Load dataset
logging.info("Loading dataset...")

data = DataContainer(
    args.dataset, args.num_train, args.num_valid,
    args.batch_size, args.valid_batch_size, seed=args.seed)

# Initiate PhysNet model
logging.info("Creating PhysNet model...")

model = PhysNet(
    F=args.num_features,
    K=args.num_basis,
    sr_cut=args.cutoff,
    num_blocks=args.num_blocks,
    num_residual_atomic=args.num_residual_atomic,
    num_residual_interaction=args.num_residual_interaction,
    num_residual_output=args.num_residual_output,
    use_electrostatic=(args.use_electrostatic==1),
    use_dispersion=(args.use_dispersion==1),
    s6=args.grimme_s6,
    s8=args.grimme_s8,
    a1=args.grimme_a1,
    a2=args.grimme_a2,
    Eshift=data.EperA_mean,
    Escale=data.EperA_stdev,
    activation_fn=shifted_softplus,
    seed=None)

#------------------------------------------------------------------------------
# Prepare loss evaluation and model trainer
#------------------------------------------------------------------------------

logging.info("prepare training...")

# Set evaluation function loss and error values if reference data are 
# available (return zero otherwise) for ...
# Total energy
if data.include_E:
    e_eval = calculate_errors
else:
    e_eval = calculate_null
# Atomic energy
if data.include_Ea:
    ea_eval = calculate_errors
else:
    ea_eval = calculate_null
# Forces
if data.include_F:
    f_eval = calculate_errors
else:
    f_eval = calculate_null
# Total charge
if data.include_Q:
    q_eval = calculate_errors
else:
    q_eval = calculate_null
# Atomic charges
if data.include_Qa:
    qa_eval = calculate_errors
else:
    qa_eval = calculate_null
# Dipole moment
if data.include_D:
    d_eval = calculate_errors
else:
    d_eval = calculate_null
    
# Load best recorded loss if available
if os.path.isfile(best_loss_file):
    loss_file = np.load(best_loss_file)
    best_loss = loss_file["loss"].item()
    best_emae = loss_file["emae"].item()
    best_ermse = loss_file["ermse"].item()
    best_fmae = loss_file["fmae"].item()
    best_frmse = loss_file["frmse"].item()
    best_qmae = loss_file["qmae"].item()
    best_qrmse = loss_file["qrmse"].item()
    best_dmae = loss_file["dmae"].item()
    best_drmse = loss_file["drmse"].item()
    best_epoch = loss_file["epoch"].item()
else:
    best_loss = np.Inf
    best_emae = np.Inf
    best_ermse = np.Inf
    best_fmae = np.Inf
    best_frmse = np.Inf
    best_qmae = np.Inf
    best_qrmse = np.Inf
    best_dmae = np.Inf
    best_drmse = np.Inf
    best_epoch = 0.
    np.savez(
        best_loss_file, loss=best_loss, emae=best_emae, ermse=best_ermse, 
        fmae=best_fmae, frmse=best_frmse, qmae=best_qmae, qrmse=best_qrmse,
        dmae=best_dmae, drmse=best_drmse, epoch=best_epoch)


#------------------------------------------------------------------------------
# Define training step
#------------------------------------------------------------------------------

#@tf.function(experimental_relax_shapes=False)
def get_indices(Nref):
    
    # Get indices pointing to batch image
    batch_seg = tf.repeat(tf.range(tf.shape(Nref)[0]), Nref)
    
    # Initiate auxiliary parameter
    Nref_tot = tf.constant(0, dtype=tf.int32)
    
    # Indices pointing to atom at each batch image
    idx = tf.range(Nref[0], dtype=tf.int32)
    # Indices for atom pairs ij - Atom i
    idx_i = tf.repeat(idx, Nref[0] - 1) + Nref_tot
    # Indices for atom pairs ij - Atom j
    idx_j = tf.roll(idx, -1, axis=0) + Nref_tot
    for Na in tf.range(2, Nref[0]):
        idx_j = tf.concat(
            [idx_j, tf.roll(idx, -Na, axis=0) + Nref_tot], 
            axis=0)
        
    # Increment auxiliary parameter
    Nref_tot = Nref_tot + Nref[0]
        
    # Complete indices arrays
    for Nref_a in Nref[1:]:
        
        rng_a = tf.range(Nref_a)
        
        idx = tf.concat(
            [idx, rng_a], axis=0)
        idx_i = tf.concat(
            [idx_i, tf.repeat(rng_a, Nref_a - 1) + Nref_tot], 
            axis=0)
        for Na in tf.range(1, Nref_a):
            idx_j = tf.concat(
                [idx_j, tf.roll(rng_a, -Na, axis=0) + Nref_tot], 
                axis=0)
        
        # Increment auxiliary parameter
        Nref_tot = Nref_tot + Nref_a
        
    # Combine indices for batch image and respective atoms
    idx = tf.stack([batch_seg, idx], axis=1)
    
    return idx, idx_i, idx_j, batch_seg

@tf.function(experimental_relax_shapes=False)
def train_step(
    batch, num_t, loss_avg_t, emse_avg_t, emae_avg_t, fmse_avg_t, fmae_avg_t, 
    qmse_avg_t, qmae_avg_t, dmse_avg_t, dmae_avg_t):
    
    # Decompose data
    N_t, Z_t, R_t, Eref_t, Earef_t, Fref_t, Qref_t, Qaref_t, Dref_t = batch
    
    # Get indices
    idx_t, idx_i_t, idx_j_t, batch_seg_t = get_indices(N_t)
    
    # Gather data
    Z_t = tf.gather_nd(Z_t, idx_t)
    R_t = tf.gather_nd(R_t, idx_t)
    if Earef_t is not None:
        Earef_t = tf.gather_nd(Earef_t, idx_t)
    if Fref_t is not None:
        Fref_t = tf.gather_nd(Fref_t, idx_t)
    if Qaref_t is not None:
        Qaref_t = tf.gather_nd(Qaref_t, idx_t)
    
    # Evaluate model
    with tf.GradientTape() as tape_t:
    
        # Calculate quantities
        energy_t, forces_t, Ea_t, Qa_t, nhloss_t = \
            model.energy_and_forces_and_atomic_properties(
                Z_t, R_t, idx_i_t, idx_j_t, Qref_t, batch_seg_t)

        # Get total charge
        Qtot_t = tf.math.segment_sum(Qa_t, batch_seg_t)

        # Get dipole moment vector
        QR_t = tf.stack([Qa_t*R_t[:,0], Qa_t*R_t[:,1], Qa_t*R_t[:,2]], 1)
        D_t = tf.math.segment_sum(QR_t, batch_seg_t)
        
        # Evaluate error and losses for ...
        # Total energy
        eloss_t, emse_t, emae_t = e_eval(Eref_t, energy_t)
        # Atomic energy
        ealoss_t, eamse_t, eamae_t = ea_eval(Earef_t, Ea_t)
        # Forces
        floss_t, fmse_t, fmae_t = f_eval(Fref_t, forces_t)
        # Total charge
        qloss_t, qmse_t, qmae_t = q_eval(Qref_t, Qtot_t)
        # Atomic charges
        qaloss_t, qamse_t, qamae_t = qa_eval(Qaref_t, Qa_t)
        # Dipole moment
        dloss_t, dmse_t, dmae_t = d_eval(Dref_t, D_t)
        # L2 regularization loss
        l2loss_t = tf.reduce_mean(tf.compat.v1.get_collection("l2loss"))
        
        # Evaluate total loss
        loss_t = (
            eloss_t + ealoss_t 
            + args.force_weight*floss_t
            + args.charge_weight*qloss_t
            + args.dipole_weight*dloss_t
            + args.nhlambda*nhloss_t
            + args.l2lambda*l2loss_t)
        
    # Get gradients of trainable variables
    gradients = tape_t.gradient(loss_t, model.trainable_variables)
    tf.summary.scalar(
        "global_gradient_norm", tf.linalg.global_norm(gradients))
    
    # Clip gradients by given maximal norm
    gradients, _ = tf.clip_by_global_norm(gradients, args.max_norm)

    # Apply gradient to trainable variables by optimizer
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Employing an exponential decay to moving averages
    ema.apply(model.trainable_variables)
    
    # Update averages
    f = num_t/(num_t + tf.cast(tf.shape(N_t)[0], tf.float32))
    loss_avg_t = f*loss_avg_t + (1.0 - f)*loss_t
    emse_avg_t = f*emse_avg_t + (1.0 - f)*emse_t
    emae_avg_t = f*emae_avg_t + (1.0 - f)*emae_t
    fmse_avg_t = f*fmse_avg_t + (1.0 - f)*fmse_t
    fmae_avg_t = f*fmae_avg_t + (1.0 - f)*fmae_t
    qmse_avg_t = f*qmse_avg_t + (1.0 - f)*qmse_t
    qmae_avg_t = f*qmae_avg_t + (1.0 - f)*qmae_t
    dmse_avg_t = f*dmse_avg_t + (1.0 - f)*dmse_t
    dmae_avg_t = f*dmae_avg_t + (1.0 - f)*dmae_t
    num_t = num_t + tf.cast(tf.shape(N_t)[0], tf.float32)
    
    return num_t, loss_avg_t, emse_avg_t, emae_avg_t, fmse_avg_t, fmae_avg_t, \
        qmse_avg_t, qmae_avg_t, dmse_avg_t, dmae_avg_t

@tf.function(experimental_relax_shapes=False)
def valid_step(
    batch, num_v, loss_avg_v, emse_avg_v, emae_avg_v, fmse_avg_v, fmae_avg_v, 
    qmse_avg_v, qmae_avg_v, dmse_avg_v, dmae_avg_v):
    
    # Decompose data
    N_v, Z_v, R_v, Eref_v, Earef_v, Fref_v, Qref_v, Qaref_v, Dref_v = batch
    
    # Get indices
    idx_v, idx_i_v, idx_j_v, batch_seg_v = get_indices(N_v)
    
    # Gather data
    Z_v = tf.gather_nd(Z_v, idx_v)
    R_v = tf.gather_nd(R_v, idx_v)
    if Earef_v is not None:
        Earef_v = tf.gather_nd(Earef_v, idx_v)
    if Fref_v is not None:
        Fref_v = tf.gather_nd(Fref_v, idx_v)
    if Qaref_v is not None:
        Qaref_v = tf.gather_nd(Qaref_v, idx_v)
    
    # Calculate quantities
    energy_v, forces_v, Ea_v, Qa_v, nhloss_v = \
        model.energy_and_forces_and_atomic_properties(
            Z_v, R_v, idx_i_v, idx_j_v, Qref_v, batch_seg_v)

    # Get total charge
    Qtot_v = tf.math.segment_sum(Qa_v, batch_seg_v)

    # Get dipole moment vector
    QR_v = tf.stack([Qa_v*R_v[:,0], Qa_v*R_v[:,1], Qa_v*R_v[:,2]], 1)
    D_v = tf.math.segment_sum(QR_v, batch_seg_v)
    
    # Evaluate error and losses for ...
    # Total energy
    eloss_v, emse_v, emae_v = e_eval(Eref_v, energy_v)
    # Atomic energy
    ealoss_v, eamse_v, eamae_v = ea_eval(Earef_v, Ea_v)
    # Forces
    floss_v, fmse_v, fmae_v = f_eval(Fref_v, forces_v)
    # Total charge
    qloss_v, qmse_v, qmae_v = q_eval(Qref_v, Qtot_v)
    # Atomic charges
    qaloss_v, qamse_v, qamae_v = qa_eval(Qaref_v, Qa_v)
    # Dipole moment
    dloss_v, dmse_v, dmae_v = d_eval(Dref_v, D_v)
    # L2 regularization loss
    l2loss_v = tf.reduce_mean(tf.compat.v1.get_collection("l2loss"))
    
    # Evaluate total loss
    loss_v = (
        eloss_v + ealoss_v 
        + args.force_weight*floss_v
        + args.charge_weight*qloss_v
        + args.dipole_weight*dloss_v
        + args.nhlambda*nhloss_v
        + args.l2lambda*l2loss_v)
    
    # Update averages
    f = num_v/(num_v + tf.cast(tf.shape(N_v)[0], tf.float32))
    loss_avg_v = f*loss_avg_v + (1.0 - f)*loss_v
    emse_avg_v = f*emse_avg_v + (1.0 - f)*emse_v
    emae_avg_v = f*emae_avg_v + (1.0 - f)*emae_v
    fmse_avg_v = f*fmse_avg_v + (1.0 - f)*fmse_v
    fmae_avg_v = f*fmae_avg_v + (1.0 - f)*fmae_v
    qmse_avg_v = f*qmse_avg_v + (1.0 - f)*qmse_v
    qmae_avg_v = f*qmae_avg_v + (1.0 - f)*qmae_v
    dmse_avg_v = f*dmse_avg_v + (1.0 - f)*dmse_v
    dmae_avg_v = f*dmae_avg_v + (1.0 - f)*dmae_v
    num_v = num_v + tf.cast(tf.shape(N_v)[0], tf.float32)
    
    return num_v, loss_avg_v, emse_avg_v, emae_avg_v, fmse_avg_v, fmae_avg_v, \
        qmse_avg_v, qmae_avg_v, dmse_avg_v, dmae_avg_v

#------------------------------------------------------------------------------
# Train PhysNet model
#------------------------------------------------------------------------------

logging.info("starting training...")

# Define Optimizer
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    args.learning_rate,
    decay_steps=args.decay_steps,
    decay_rate=args.decay_rate,
    staircase=False,
    name="learning_rate_decay")
optimizer = tf.keras.optimizers.Adam(
    lr=args.learning_rate, amsgrad=True)

# Initiate epoch and step counter
epoch = tf.Variable(
    initial_value=1, trainable=False, 
    name="global_epoch", dtype=tf.int64)
step = tf.Variable(
    initial_value=1, trainable=False, 
    name="global_step", dtype=tf.int64)

# Initiate summary writer
summary_writer = tf.summary.create_file_writer(
    log_dir, name="train_summary")

# Initiate checkpoints and load latest checkpoint
ckpt = tf.train.Checkpoint(
    model=model, optimizer=optimizer, epoch=epoch, step=step)

ckpt_manager = tf.train.CheckpointManager(
    ckpt, log_dir, max_to_keep=10, checkpoint_name="model")

latest_ckpt = ckpt_manager.latest_checkpoint
if latest_ckpt is not None:
    epoch = tf.Variable(
        initial_value=int(latest_ckpt.split('-')[-1]), trainable=False,
        name="global_epoch", dtype=tf.int64)
    step = tf.Variable(
        initial_value=int(latest_ckpt.split('-')[-1]), trainable=False,
        name="global_step", dtype=tf.int64)
    ckpt.restore(latest_ckpt)
else:
    epoch = tf.Variable(
        initial_value=1, trainable=False, 
        name="global_epoch", dtype=tf.int64)
    step = tf.Variable(
        initial_value=1, trainable=False, 
        name="global_step", dtype=tf.int64)

# Initiate an exponential decay to moving averages
ema = tf.train.ExponentialMovingAverage(args.ema_decay, step)

# Create validation batches
valid_batches = data.get_valid_batches()

# Initialize counter for estimated time per epoch
time_train_estimation = np.nan
time_train = 0.0

# Training loop
# Terminate training when maximum number of iterations is reached
while epoch <= args.max_steps:
    
    # Reset error averages
    num_t, loss_avg_t, emse_avg_t, emae_avg_t, fmse_avg_t, fmae_avg_t, \
        qmse_avg_t, qmae_avg_t, dmse_avg_t, dmae_avg_t = reset_averages()
    
    # Create train batches
    train_batches, N_train_batches = data.get_train_batches()
    
    # Start train timer
    train_start = time()
    
    # Iterate over batches
    for ib, batch in enumerate(train_batches):
        
        # Start batch timer
        batch_start = time()
        
        # Show progress bar
        if args.show_progress:
            printProgressBar(
                ib, N_train_batches, prefix="Epoch {0: 5d}".format(
                    epoch.numpy()), 
                suffix=("Complete - Remaining Epoch Time: " 
                    + "{0: 4.1f} s     ".format(time_train_estimation)),
                length=42)
            
        # Training step
        num_t, loss_avg_t, emse_avg_t, emae_avg_t, fmse_avg_t, fmae_avg_t, \
        qmse_avg_t, qmae_avg_t, dmse_avg_t, dmae_avg_t = train_step(
            batch, num_t, loss_avg_t, emse_avg_t, emae_avg_t, 
            fmse_avg_t, fmae_avg_t, qmse_avg_t, qmae_avg_t, 
            dmse_avg_t, dmae_avg_t)
        
        # Stop batch timer
        batch_end = time()
        
        # Actualize time estimation
        if args.show_progress:
            if ib==0:
                time_train_estimation = (
                    (batch_end - batch_start)*(N_train_batches - 1))
            else:
                time_train_estimation = (
                    0.5*(time_train_estimation - (batch_end - batch_start))
                    + 0.5*(batch_end - batch_start)*(N_train_batches - ib - 1))
                
        # Increment step number
        step.assign_add(1)
    
    # Stop train timer
    train_end = time()
    time_train = train_end - train_start
    
    # Show final progress bar and time
    if args.show_progress:
        printProgressBar(
            N_train_batches, N_train_batches, prefix="Epoch {0: 5d}".format(
                epoch.numpy()), 
            suffix=("Done - Epoch Time: " 
                + "{0: 4.1f} s, Average Loss: {1: 4.4f}   ".format(
                    time_train, loss_avg_t)),
            length=42)
        
    # Save progress
    if (epoch % args.save_interval == 0):
        ckpt_manager.save(checkpoint_number=epoch)
    
    # Check performance on the validation set
    if (epoch % args.validation_interval == 0):
        
        # Update training results
        results_t = {}            
        results_t["loss_train"] = loss_avg_t.numpy()
        results_t["time_train"] = time_train
        if data.include_E:
            results_t["energy_mae_train"]  = emae_avg_t.numpy()
            results_t["energy_rmse_train"] = np.sqrt(emse_avg_t.numpy())
        if data.include_F:
            results_t["forces_mae_train"]  = fmae_avg_t.numpy()
            results_t["forces_rmse_train"] = np.sqrt(fmse_avg_t.numpy())
        if data.include_Q:
            results_t["charge_mae_train"]  = qmae_avg_t.numpy()
            results_t["charge_rmse_train"] = np.sqrt(qmse_avg_t.numpy())
        if data.include_D:
            results_t["dipole_mae_train"]  = dmae_avg_t.numpy()
            results_t["dipole_rmse_train"] = np.sqrt(dmse_avg_t.numpy())
        
        with summary_writer.as_default():
            for key, value in results_t.items():
                tf.summary.scalar(key, value, step=epoch)
            summary_writer.flush()
        
        # Backup variables and assign EMA variables
        backup_vars = [tf.identity(var) for var in model.trainable_variables]
        for var in model.trainable_variables:
            var.assign(ema.average(var))
        
        # Reset error averages
        num_v, loss_avg_v, emse_avg_v, emae_avg_v, fmse_avg_v, fmae_avg_v, \
        qmse_avg_v, qmae_avg_v, dmse_avg_v, dmae_avg_v = reset_averages()
        
        # Start valid timer
        valid_start = time()
        
        for batch in valid_batches:
            
            num_v, loss_avg_v, emse_avg_v, emae_avg_v, \
            fmse_avg_v, fmae_avg_v, qmse_avg_v, qmae_avg_v, \
            dmse_avg_v, dmae_avg_v = valid_step(
                batch, num_v, loss_avg_v, emse_avg_v, emae_avg_v, 
                fmse_avg_v, fmae_avg_v, qmse_avg_v, qmae_avg_v, 
                dmse_avg_v, dmae_avg_v)
        
        # Stop valid timer
        valid_end = time()
        time_valid = valid_end - valid_end
            
        # Update validation results
        results_v = {}
        results_v["loss_valid"] = loss_avg_v.numpy()
        results_t["time_valid"] = time_valid
        if data.include_E:
            results_v["energy_mae_valid"] = emae_avg_v.numpy()
            results_v["energy_rmse_valid"] = np.sqrt(emse_avg_v.numpy())
        if data.include_F:
            results_v["forces_mae_valid"] = fmae_avg_v.numpy()
            results_v["forces_rmse_valid"] = np.sqrt(fmse_avg_v.numpy())
        if data.include_Q:
            results_v["charge_mae_valid"] = qmae_avg_v.numpy()
            results_v["charge_rmse_valid"] = np.sqrt(qmse_avg_v.numpy())
        if data.include_D:
            results_v["dipole_mae_valid"] = dmae_avg_v.numpy()
            results_v["dipole_rmse_valid"] = np.sqrt(dmse_avg_v.numpy())
            
        with summary_writer.as_default():
            for key, value in results_v.items():
                tf.summary.scalar(key, value, step=epoch)
            summary_writer.flush()
        
        if results_v["loss_valid"] < best_loss:
            
            # Assign results of best validation
            best_loss = results_v["loss_valid"]
            if data.include_E:
                best_emae = results_v["energy_mae_valid"]
                best_ermse = results_v["energy_rmse_valid"]
            else:
                best_emae = np.Inf
                best_ermse = np.Inf
            if data.include_F:
                best_fmae = results_v["forces_mae_valid"]
                best_frmse = results_v["forces_rmse_valid"]
            else:
                best_fmae = np.Inf
                best_frmse = np.Inf
            if data.include_Q:
                best_qmae = results_v["charge_mae_valid"]
                best_qrmse = results_v["charge_rmse_valid"]
            else:
                best_qmae = np.Inf
                best_qrmse = np.Inf
            if data.include_D:
                best_dmae = results_v["dipole_mae_valid"]
                best_drmse = results_v["dipole_rmse_valid"]
            else:
                best_dmae = np.Inf
                best_drmse = np.Inf
            best_epoch = epoch.numpy()
            
            # Save best results
            np.savez(
                best_loss_file, loss=best_loss, 
                emae=best_emae, ermse=best_ermse, 
                fmae=best_fmae, frmse=best_frmse, 
                qmae=best_qmae, qrmse=best_qrmse, 
                dmae=best_dmae, drmse=best_drmse, 
                epoch=best_epoch)
            
            # Save best model variables
            model.write(best_checkpoint)
            
        # Update best results
        results_b = {}
        results_b["loss_best"] = best_loss
        if data.include_E:
            results_b["energy_mae_best"] = best_emae
            results_b["energy_rmse_best"] = best_ermse
        if data.include_F:
            results_b["forces_mae_best"] = best_fmae
            results_b["forces_rmse_best"] = best_frmse
        if data.include_Q:
            results_b["charge_mae_best"] = best_qmae
            results_b["charge_rmse_best"] = best_qrmse
        if data.include_D:
            results_b["dipole_mae_best"] = best_dmae
            results_b["dipole_rmse_best"] = best_drmse
        
        with summary_writer.as_default():
            for key, value in results_b.items():
                tf.summary.scalar(key, value, step=epoch)
            summary_writer.flush()
        
        for var, bck in zip(model.trainable_variables, backup_vars):
            var.assign(bck)
        
    # Generate summaries
    if ((epoch % args.summary_interval == 0) 
        and (epoch >= args.validation_interval)): 
        
        if data.include_E:
            print(
                "Summary Epoch: " + \
                str(epoch.numpy()) + '/' + str(args.max_steps), 
                "\n    Loss   train/valid: {0: 1.3e}/{1: 1.3e}, ".format(
                    results_t["loss_train"], 
                    results_v["loss_valid"]), 
                " Best valid loss:   {0: 1.3e}, ".format(
                    results_b["loss_best"]),
                "\n    MAE(E) train/valid: {0: 1.3e}/{1: 1.3e}, ".format(
                    results_t["energy_mae_train"],
                    results_v["energy_mae_valid"]), 
                " Best valid MAE(E): {0: 1.3e}, ".format(
                    results_b["energy_mae_best"]))
    
    # Increment epoch number
    epoch.assign_add(1)
