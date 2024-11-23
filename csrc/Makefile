.PHONY: prepro test clean

SRC_FILES  := test.cu pso/velocity.cu pso/position.cu pso/evolve.cu utils/utils.cu funcs/funcs.cu
INC_FLAGS  := -Ipso/include -Iutils/include -Ifuncs/include
CUDA_FLAGS := -O3 --generate-code=arch=compute_61,code=[compute_61,sm_61] -rdc=true

all: ${SRC_FILES}
	nvcc ${SRC_FILES} ${INC_FLAGS} ${CUDA_FLAGS} -o test

prepro: test.cu
	nvcc test.cu -E -I./include > prepro.txt

test:
	./test > out.txt

clean:
	rm -rf test out.txt