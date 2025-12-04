CC = nvcc
CFLAGS = -O3 -arch=sm_60 -Xptxas -O3,-v -use_fast_math --fmad=true -lineinfo
TARGET = nbody_optimized

all: $(TARGET)

$(TARGET): optimized_final.cu
	$(CC) $(CFLAGS) -o $(TARGET) optimized_final.cu

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) *.o

.PHONY: all run clean