#######################################################################################
.PHONY: help
help:
	@echo ""
	@echo "Usage:"
	@echo "  make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  all               Build all tests and benchmarks"
	@echo "  test_csr_utils    Build CSR utils test"
	@echo "  test_csr_spmm     Build CSR SpMM test"
	@echo "  bench_csr_spmm    Benchmark custom CSR SpMM"
	@echo "  bench_cusparse    Benchmark cuSPARSE"
	@echo "  clean             Remove build outputs"
	@echo ""
	@echo "Tip: Run 'bear -- make all' to generate compile_commands.json"
#######################################################################################

# compiler settings
NVCC      	= nvcc
CXXFLAGS	= -O3 -std=c++17 --use_fast_math -Xptxas=-v -arch=sm_89 # NOTE: change -arch to fit your NVIDIA model

DEBUGFLAGS = -g -O0 -src-in-ptx -lineinfo -std=c++17
PROFFLAGS  = -O2 -g --generate-line-info -src-in-ptx -std=c++17

ROOT_DIR := $(abspath $(CURDIR))
SRCS     := $(abspath src/csr_spmm.cu \
					  src/csr_utils.cu \
					  src/dense_gemm.cu \
					  src/dense_utils.cu)
INCLUDE  := $(abspath include)

all: test_csr_utils test_csr_spmm bench_csr_spmm bench_cusparse


test_csr_utils: $(SRCS) tests/test_csr_utils.cu
	$(NVCC) $(DEBUGFLAGS) -I$(INCLUDE) $^ -o $@

test_csr_spmm: $(SRCS) tests/test_csr_spmm.cu
	$(NVCC) $(DEBUGFLAGS) -I$(INCLUDE) $^ -o $@

bench_csr_spmm: $(SRCS) benchmarks/bench_csr_spmm.cu
	$(NVCC) $(CXXFLAGS) -I$(INCLUDE) $^ -o $@

bench_cusparse: $(SRCS) benchmarks/bench_cusparse.cu
	$(NVCC) $(CXXFLAGS) -lcusparse -I$(INCLUDE) $^ -o $@

.PHONY: all clean


clean:
	rm -f test_csr_utils \
		  test_csr_spmm	\
		  bench_csr_spmm
