# -*- coding: utf-8 -*-

import prep_read_LLPLLR_features as pr
import numpy as np
from pycuda import driver, compiler, gpuarray, tools
#initialize the device
import pycuda.autoinit

def ShowGPUInfo():
    (free,total) = driver.mem_get_info()
    print('Global memory occupancy:%f%% free' % (free*100 / total))
    for devicenum in range(driver.Device.count()):
        device = driver.Device(devicenum)
        attrs = device.get_attributes()
        #Beyond this point is just pretty printing
        print('\n===Attributes for device %d' % devicenum)
        for (key,value) in attrs.iteritems():
            print('    %s:%s' % (str(key), str(value)))
        #end for print attrs
    #end for every device elements
#end ShowGPUInfo()


if __name__ == '__main__':
    kaldi_trunk_path = '/usr/local/kaldi-trunk/'
    pl_feats_path = '/share/homes/yang/tupntnu_DS/sR/exp_rnn_lstm/delta2_and_smbr/PLFeats_DS_dev/phone_level_corpus.txt'
    pl_feats_matrix_with_label = pr.GetPLFeats(kaldi_trunk_path, pl_feats_path)
    #print pl_feats_matrix_with_label[50]

    #把feature變成numpy array
    pl_feats_matrix = []
    pl_feats_label = []

    for slice in pl_feats_matrix_with_label:
        pl_feats_matrix.append(slice[2:])  #每一colom是434維, 每一個row表示不同的syllable
        pl_feats_label.append(slice[0:1])
    #end for

    kernel_code_template = '''
    __global__ void MatrixMulKernel(float *feature, float *weight, float *result) {
      // 2D Thread ID (assuming that only *one* block will be executed)
      int tx = threadIdx.x;
      int ty = threadIdx.y;

      // pValue is used to store the element of the matrix
      // that is computed by the thread
      float pValue = 0;

      // Each thread loads one row of M and one column of N,
      //   to produce one element of P.
      for (int k = 0; k < %(MATRIX_ROW_SIZE)s; ++k) {
        float elementA = feature[ty * %(MATRIX_ROW_SIZE)s + k];
        float elementB = weight[k * %(MATRIX_COL_SIZE)s + tx];
        pValue += elementA * elementB;
      } // for()

      // Write the matrix to device memory;
      // each thread writes one element
      result[ty * %(MATRIX_ROW_SIZE)s + tx] = pValue;
    } //MatrixMulKernel()
    '''
    #define the (square) matrix size, note that we'll only use *one* block of threads here
    #as a consequence this number (squared) can't exceed max_threads,
    MATRIX_SIZE = 4
    MATRIX_ROW_SIZE = 5407
    MATRIX_COL_SIZE = 434

    #prepare input features and weight matrix
    #feature_cpu = np.random.randn(5407, 434).astype(np.float32)
    feature_cpu = np.asarray(pl_feats_matrix, dtype=np.float32) #(5407, 434)
    #weight_cpu = np.random.randn(5407, 10).astype(np.float32)
    weight_cpu = np.identity(434)
    #feature_cpu = np.array([[1,2,3,4],
    #                        [5,6,7,8]]).astype(np.int16)
    #weight_cpu = 2*np.identity(4).astype(np.int16)

    print feature_cpu.shape

    #compute reference on the CPU to verify GPU computation
    result_cpu = np.dot(feature_cpu, weight_cpu)

    # transfer host (CPU) memory to device (GPU) memory
    print '# transfer host (CPU) memory to device (GPU) memory '
    feaeture_gpu = gpuarray.to_gpu(feature_cpu.astype(np.float32))
    weight_gpu = gpuarray.to_gpu(weight_cpu.astype(np.float32))

    # create empty gpu array for the result (result = feature * weight)
    print '# create empty gpu array for the result (result = feature * weight)'
    #result_gpu = gpuarray.empty((434,10), np.float32)
    result_gpu = gpuarray.empty((5407,434), np.float32)
    #result_gpu = gpuarray.empty((2,4), np.int16)

    # get the kernel code from the template
    # by specifying the constant MATRIX_SIZE
    print '# get the kernel code from the template '
    kernel_code = kernel_code_template % { 'MATRIX_SIZE': MATRIX_SIZE,
                                           'MATRIX_ROW_SIZE': MATRIX_ROW_SIZE,
                                           'MATRIX_COL_SIZE': MATRIX_COL_SIZE, }

    # compile the kernel code
    print '# compile the kernel code '
    mod = compiler.SourceModule(kernel_code)

    print '===='
    ShowGPUInfo()
    print '===='

    print '# get the kernel function from the compiled module'
    # get the kernel function from the compiled module
    matrixmul = mod.get_function('MatrixMulKernel')

    #call the kernel on the card
    #input= feaeture_gpu, weight_gpu, output=result_gpu
    matrixmul(feaeture_gpu, weight_gpu, result_gpu,
              # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
              block = (MATRIX_ROW_SIZE, MATRIX_COL_SIZE, 3),)

    # print the results
    print '-' * 80
    print 'Matrix Feature (GPU):'
    print feaeture_gpu.get().shape
    print feaeture_gpu.get()

    print '-' * 80
    print 'Matrix Weight (GPU):'
    print weight_gpu.get().shape
    print weight_gpu.get()

    print '-' * 80
    print 'Matrix Result (GPU):'
    print result_gpu.get().shape
    print result_gpu.get()

    print '-' * 80
    print 'Matrix Result (CPU):'
    print result_cpu.shape
    print result_cpu

    print "-" * 80
    print "CPU-GPU difference:"
    print result_cpu - result_gpu.get()

    np.allclose(result_cpu, result_gpu.get())
#end if __name__ == '__main__':