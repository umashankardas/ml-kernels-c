CC      := gcc
CFLAGS  := -O3 -march=native -Wall -Wextra -g -I. -fopenmp
LDFLAGS := -lm -fopenmp

OBJ_DIR := obj
BIN_DIR := bin
SRC_DIR := src
BENCH_DIR := bench/src

# --- OBJECT GROUPS ---
UTILS_OBJS  := $(OBJ_DIR)/timer.o
MATMUL_OBJS := $(OBJ_DIR)/matmul_naive.o $(OBJ_DIR)/matmul_block.o \
               $(OBJ_DIR)/matmul_simd.o $(OBJ_DIR)/matmul_openmp.o \
               $(OBJ_DIR)/linear_simd.o $(OBJ_DIR)/linear_openmp.o \
               $(OBJ_DIR)/linear_relu_fused.o
ACT_OBJS    := $(OBJ_DIR)/relu.o $(OBJ_DIR)/softmax.o $(OBJ_DIR)/pooling.o
CONV_OBJS   := $(OBJ_DIR)/conv2d_naive.o $(OBJ_DIR)/conv2d_fast.o \
               $(OBJ_DIR)/im2col.o $(OBJ_DIR)/conv_transposed_naive.o \
               $(OBJ_DIR)/conv_transposed_fast.o

# All objects together
KERNEL_OBJS := $(UTILS_OBJS) $(MATMUL_OBJS) $(ACT_OBJS) $(CONV_OBJS)

# --- EXECUTABLE TARGETS ---
all: $(BIN_DIR)/bench-matmul.exe \
     $(BIN_DIR)/test-softmax.exe \
     $(BIN_DIR)/sim-model.exe \
     $(BIN_DIR)/sim-sd.exe \
     $(BIN_DIR)/sim-sd-full.exe \
     $(BIN_DIR)/test-conv.exe \
     $(BIN_DIR)/bench-conv.exe

# Manual Linking Rules (To avoid wildcard mismatch)
$(BIN_DIR)/bench-matmul.exe: $(BENCH_DIR)/bench-matmul.c $(KERNEL_OBJS)
	@if not exist $(BIN_DIR) mkdir $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/test-softmax.exe: $(BENCH_DIR)/test-softmax.c $(KERNEL_OBJS)
	@if not exist $(BIN_DIR) mkdir $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/sim-model.exe: $(BENCH_DIR)/sim-model.c $(KERNEL_OBJS)
	@if not exist $(BIN_DIR) mkdir $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/sim-sd.exe: $(BENCH_DIR)/sim-sd.c $(KERNEL_OBJS)
	@if not exist $(BIN_DIR) mkdir $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/sim-sd-full.exe: $(BENCH_DIR)/sim-sd-full.c $(KERNEL_OBJS)
	@if not exist $(BIN_DIR) mkdir $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/test-conv.exe: $(BENCH_DIR)/test-conv.c $(KERNEL_OBJS)
	@if not exist $(BIN_DIR) mkdir $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/bench-conv.exe: $(BENCH_DIR)/bench-conv.c $(KERNEL_OBJS)
	@if not exist $(BIN_DIR) mkdir $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# --- OBJECT COMPILATION (Pattern Rules) ---
$(OBJ_DIR)/%.o: $(SRC_DIR)/utils/%.c
	@if not exist $(OBJ_DIR) mkdir $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/activations/%.c
	@if not exist $(OBJ_DIR) mkdir $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/matmul/%.c
	@if not exist $(OBJ_DIR) mkdir $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/conv/%.c
	@if not exist $(OBJ_DIR) mkdir $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	@if exist $(OBJ_DIR) rmdir /s /q $(OBJ_DIR)
	@if exist $(BIN_DIR) rmdir /s /q $(BIN_DIR)

.PHONY: all clean