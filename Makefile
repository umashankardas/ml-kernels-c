CC = gcc
CFLAGS = -O3 -march=native -Wall -Wextra -g -I.
LDFLAGS = -lm

OBJ_DIR = obj
BIN_DIR = bin
SRC_DIR = src
BENCH_SRC_DIR = bench/src

# Source files
UTILS_SRC = $(SRC_DIR)/utils/timer.c
MATMUL_SRC = $(SRC_DIR)/matmul/matmul_naive.c
MATMUL_BLOCK_SRC = $(SRC_DIR)/matmul/matmul_block.c
BENCH_SRC = $(BENCH_SRC_DIR)/bench-matmul.c

# Object files
OBJS = $(OBJ_DIR)/timer.o $(OBJ_DIR)/matmul_naive.o $(OBJ_DIR)/matmul_block.o $(OBJ_DIR)/matmul_simd.o
TARGET = $(BIN_DIR)/bench-matmul.exe

all: $(TARGET)

$(TARGET): $(BENCH_SRC) $(OBJS)
	if not exist $(BIN_DIR) mkdir $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJ_DIR)/timer.o: $(SRC_DIR)/utils/timer.c
	if not exist $(OBJ_DIR) mkdir $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/matmul_naive.o: $(SRC_DIR)/matmul/matmul_naive.c
	if not exist $(OBJ_DIR) mkdir $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# --- ADDED THIS RULE ---
$(OBJ_DIR)/matmul_block.o: $(SRC_DIR)/matmul/matmul_block.c
	if not exist $(OBJ_DIR) mkdir $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@


$(OBJ_DIR)/matmul_simd.o: $(SRC_DIR)/matmul/matmul_simd.c
	if not exist $(OBJ_DIR) mkdir $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	if exist $(OBJ_DIR) rmdir /s /q $(OBJ_DIR)
	if exist $(BIN_DIR) rmdir /s /q $(BIN_DIR)

.PHONY: all clean