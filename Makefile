#######################################################################################
.PHONY: help
help:
	@echo "Makefile Usage:"
	@echo ""
	@echo "Run 'bear -- make all' to generate compile_commands.json, for clang-based tools."
#######################################################################################

# compiler settings
NVCC      	= nvcc
CXXFLAGS	= -O2 -std=c++17

DEBUGFLAGS = -g -O0 -src-in-ptx -lineinfo -std=c++17
PROFFLAGS  = -O2 -g --generate-line-info -src-in-ptx -std=c++17

ROOT_DIR := $(abspath $(CURDIR))
SRCS     := $(abspath src/csr_spmm.cu \
					  src/csr_utils.cu \
					  src/dense_gemm.cu \
					  src/dense_utils.cu)
INCLUDE  := $(abspath include)

all: test_csr_utils test_spmm_csr


test_csr_utils: $(SRCS) tests/test_csr_utils.cu
	$(NVCC) $(DEBUGFLAGS) -I$(INCLUDE) $^ -o $@

test_spmm_csr: $(SRCS) tests/test_spmm_csr.cu
	$(NVCC) $(DEBUGFLAGS) -I$(INCLUDE) $^ -o $@


.PHONY: all clean



clean:
	rm -f test_csr_utils \
		  test_spmm_csr
