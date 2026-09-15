NVCC := nvcc
NVCC_FLAGS := -O3 -arch=native
BIN_DIR := bin

.PHONY: all clean vector_add reduction

all: vector_add reduction

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

vector_add: kernels/01_vector_add.cu | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $(BIN_DIR)/vector_add kernels/01_vector_add.cu

reduction: kernels/02_reduction.cu | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $(BIN_DIR)/reduction kernels/02_reduction.cu

clean:
	rm -rf $(BIN_DIR)
