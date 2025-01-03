# Project Settings
debug ?= 0
NAME := matmul
SRC_DIR := src
BUILD_DIR := build
INCLUDE_DIR := include
LIB_DIR := lib
TESTS_DIR := tests
BIN_DIR := bin

# Compiler and tools
CC := nvcc
LINTER := clang-tidy
FORMATTER := clang-format

# Generate paths for all object files
OBJS := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o, $(wildcard $(SRC_DIR)/*.cu))
LIB_OBJS := $(patsubst $(LIB_DIR)/%.cu,$(BUILD_DIR)/lib/%.o, $(wildcard $(LIB_DIR)/**/*.cu))
TEST_OBJS := $(patsubst $(TESTS_DIR)/%.cu,$(BUILD_DIR)/tests/%.o, $(wildcard $(TESTS_DIR)/*.cu))
OBJS_NO_MAIN := $(filter-out $(BUILD_DIR)/main.o, $(OBJS))

# Compiler and linker flags
CFLAGS := -Xcompiler -fopenmp
LDFLAGS := -lm

ifeq ($(debug), 1)
	CFLAGS += -G
else
	CFLAGS += -O3
endif

# Targets

# Default target: build the main executable
all: dir $(BIN_DIR)/$(NAME) $(BIN_DIR)/$(NAME)_test

# Build main executable from object files
$(BIN_DIR)/$(NAME): $(OBJS) $(LIB_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

# Build object files for source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | dir
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -o $@ -c $<

# Build object files for library files
$(BUILD_DIR)/lib/%.o: $(LIB_DIR)/%.cu | dir
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -o $@ -c $<

# Build test executable
$(BIN_DIR)/$(NAME)_test: $(TEST_OBJS) $(OBJS_NO_MAIN) $(LIB_OBJS) | dir
	$(CC) $(CFLAGS) $(LDFLAGS) -lcunit -o $@ $^

# Compile test objects
$(BUILD_DIR)/tests/%.o: $(TESTS_DIR)/%.cu | dir
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -o $@ -c $<

# Run CUnit tests
test: $(BIN_DIR)/$(NAME)_test
	@$(BIN_DIR)/$(NAME)_test

# Run linter on source, include, and test directories
lint:
	@$(LINTER) --config-file=.clang-tidy $(SRC_DIR)/* $(INCLUDE_DIR)/* $(TESTS_DIR)/* -- $(CFLAGS)

# Run formatter on source, include, and test directories
format:
	@$(FORMATTER) -style=file -i $(SRC_DIR)/* $(INCLUDE_DIR)/* $(TESTS_DIR)/*

# Directory setup
dir:
	@mkdir -p $(BUILD_DIR) $(BIN_DIR)

# Clean build and bin directories
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

.PHONY: all test lint format dir clean