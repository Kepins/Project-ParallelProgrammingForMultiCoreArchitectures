# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -fopenmp -Iinclude -O3

# Directories
SRCDIR = src
BUILDDIR = build
INCDIR = include

# Source and object files
SRCS = $(wildcard $(SRCDIR)/*.c)
OBJS = $(patsubst $(SRCDIR)/%.c, $(BUILDDIR)/%.o, $(SRCS))

# Target executable
TARGET = $(BUILDDIR)/main

# Default rule: compile and link the project
all: $(TARGET)

# Linking
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS)

# Compiling source files into object files
$(BUILDDIR)/%.o: $(SRCDIR)/%.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Create the build directory if it doesn't exist
$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# Clean the build directory
clean:
	rm -rf $(BUILDDIR)

# Run the compiled program
run: $(TARGET)
	./$(TARGET)

.PHONY: all run clean
