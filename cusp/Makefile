#compilers
CC=nvcc

#includes
CUDA_INSTALL_PATH = /usr/local/cuda
INCLUDES = -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_INSTALL_PATH)/samples/common/inc

#libs
CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib64 -lcudart -lcusparse
LIBS = $(CUDA_LIBS)

#MKL
MKL_INSTALL_DIR = /opt/intel/mkl
MKL_INCLUDE = -I$(MKL_INSTALL_DIR)/include -I$(MKL_INSTALL_DIR)
MKL_LIBS = -L$(MKL_INSTALL_DIR)/lib/intel64 -lmkl_rt -ldl
#-gencode=arch=compute_35,code=sm_35
#-gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_52,code=compute_52

all:
	nvcc -O3 -m64 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_61,code=compute_61 cusp.cu -o cusp $(INCLUDES) $(LIBS)
#	nvcc -O3 -m64 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_52,code=compute_52 spmm_hybrid.cu -o mkl $(INCLUDES) $(LIBS)


