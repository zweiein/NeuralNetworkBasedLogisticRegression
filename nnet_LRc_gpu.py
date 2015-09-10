# -*- coding: utf-8 -*-

#The current division (/) operator has an ambiguous meaning for
#    numerical arguments: it returns the floor of the mathematical
#    result of division if the arguments are ints or longs, but it
#    returns a reasonable approximation of the division result if the
#    arguments are floats or complex.  This makes expressions expecting
#    float or complex results error-prone when integers are not
#    expected but possible as inputs.
from __future__ import division
#1.0 / 2.0 --> 0.5                            1 / 2 --> 0.5
#1.0 / 2   --> 0.5        after import        4 / 2 --> 2.0
#1 / 2.0   --> 0.5          division          1 // 2 --> 0
#1 / 2     --> 0              ===>            4 // 2 --> 2


import numpy as np
from pycuda import driver, compiler, gpuarray, tools

from Cheetah.Template import Template
from os import path
import pycuda.autoinit

import prep_read_LLPLLR_features as pr

# -- default parameters
DEFAULT_BLOCK_SIZE = 16
DEFAULT_WORK_SIZE = 1
DEFAULT_UNROLL = 0
DEFAULT_SPILL = False
DEFAULT_PREFETCH = False


CURR_PATH = path.dirname(path.abspath(__file__))
TEMPLATE_FILENAME = path.join(CURR_PATH, "gpu_matrixmul.cu")


# ------------------------------------------------------------------------------
def Matrixmul_opt(matrix_a, matrix_b,
                  block_size = DEFAULT_BLOCK_SIZE,
                  work_size = DEFAULT_WORK_SIZE,
                  unroll = DEFAULT_UNROLL,
                  spill = DEFAULT_SPILL,
                  prefetch = DEFAULT_PREFETCH):
    #matrix_a is the feature matrix, matrix_b is the weight matrix
    a_height, a_width = matrix_a.shape
    b_height, b_width = matrix_b.shape
    
    assert a_width == b_height #if not equal, raise an error

    # -- pad input matrices appropriately
    a_height_padded = int(np.ceil(a_height/block_size)) * block_size
    a_width_padded = int(np.ceil(a_width/block_size)) * (block_size*work_size)
    matrix_a_padded = np.zeros((a_height_padded, a_width_padded), np.float32)
    matrix_a_padded[:a_height,:a_width] = matrix_a

    b_height_padded = a_width_padded
    b_width_padded = int(np.ceil(b_width/(block_size*work_size))) * (block_size*work_size)
    matrix_b_padded = np.zeros((b_height_padded, b_width_padded), np.float32)
    matrix_b_padded[:b_height, :b_width] = matrix_b

    c_height_padded = a_height_padded
    c_width_padded = b_width_padded

    # -- upload padded input matrices to the GPU
    matrix_a_gpu = gpuarray.to_gpu(matrix_a_padded)
    matrix_b_gpu = gpuarray.to_gpu(matrix_b_padded)

    # -- create empty container matrix for the result (C = A * B)
    matrix_c_gpu = gpuarray.zeros((c_height_padded, c_width_padded), np.float32)

    # -- generate and compile the code
    # prepare the template parameters
    template_params = { 
        'BLOCK_SIZE': block_size, 
        'WORK_SIZE': work_size, 
        'UNROLL': unroll, 
        'SPILL': spill, 
        'PREFETCH': prefetch, 
        'A_WIDTH': a_width_padded,
        'A_HEIGHT': a_height_padded,
        'B_WIDTH': b_width_padded,
        }
    
    # run the template engine to get the code
    kernel_code = Template(
        file = TEMPLATE_FILENAME,
        searchList = [template_params],
        )
    
    # compile the code
    module = compiler.SourceModule(kernel_code)
    
    # get the kernel from the module
    matrixmul_func = module.get_function("matrixMul")

    # some info about the module
    print "number of registers used:", matrixmul_func.num_regs

    # block of threads
    # ATTENTION: block is (threadDim.x, threadDim.y, threadDim.z) 
    #            and not (threadDim.z, threadDim.y, threadDim.x)
    block =  block_size, block_size, 1
    
    # grid of blocks 
    # ATTENTION: it's (blockDim.x, blockDim.y) 
    #            and not (blockDim.y, blockDim.x)
    grid = int(c_width_padded / block_size /work_size), int(c_height_padded / block_size)

    # -- call the kernel on the GPU
    # Note that when we use time_kernel=True pycuda will automatically synchronize the kernel 
    # to make sure that the timing is correct. If you time the code yourself, you'll have to
    # synchronize the current Context.
    gpu_time = matrixmul_func(
        # -- output
        matrix_c_gpu,
        # -- inputs
        matrix_a_gpu, matrix_b_gpu,
        # -- grid of blocks
        grid = grid, 
        # -- block of threads
        block = block, 
        # -- time the kernel (approx.)
        time_kernel = True,
        )

    # get the GPU matrix back to CPU memory
    matrix_c_padded = matrix_c_gpu.get()
    matrix_c = matrix_c_padded[:a_height, :b_width]

    return matrix_c, gpu_time
#end Matrixmul_opt()



if __name__ == "__main__": 
    kaldi_trunk_path = '/usr/local/kaldi-trunk/'
    pl_feats_path = '/share/homes/yang/tupntnu_DS/sR/exp_rnn_lstm/delta2_and_smbr/PLFeats_DS_dev/phone_level_corpus.txt'
    pl_feats_matrix_with_label = pr.GetPLFeats(kaldi_trunk_path, pl_feats_path)
    #print pl_feats_matrix_with_label[50]

    # transform the feature matrix into numpy array
    pl_feats_matrix = []
    pl_feats_label = []

    for slice in pl_feats_matrix_with_label:
        pl_feats_matrix.append(slice[2:])  # every column has 434 dim, every row shows different syllables
        pl_feats_label.append(slice[0:1])
    #end for

    # matrix sizes
    feature_height = 5407
    feature_width = 434
    weight_height = feature_width
    weight_width = 10

    # create random weight matrices
    np.random.seed(30)
    #mat_a = np.random.randn(a_height, a_width).astype(np.float32)
    matrix_feature = np.asarray(pl_feats_matrix, dtype=np.float32) #OK
    matrix_weight = np.random.randn(weight_height, weight_width).astype(np.float32) #ok

    # compute reference on the cpu to verify GPU computation
    matrix_reference = np.dot(matrix_feature, matrix_weight)

    # -- this is a good place to auto-tune the code (using the optimization kwargs)
    # (note that you may need more that one iteration to get accurate timing estimates)
    matrix_result, gpu_time = Matrixmul_opt(matrix_feature, matrix_weight)

    # check for correctness
    diff = matrix_result - matrix_reference
    error = np.absolute(diff).max()
    assert error <= 1e-2
    l2norm = np.linalg.norm(diff)
    #print "l2norm: ", l2norm

    # print some stats
    print "gpu time:" + gpu_time
    gflop = matrix_result.size * (feature_width * 2.) / (1000**3.)
    gflops = gflop / gpu_time
    print "gflops:" + gflops

    # print multiplication result
    print '-'*80
    print matrix_result.shape
    print matrix_result
    print '-'*80
#end if __name__ == "__main__"