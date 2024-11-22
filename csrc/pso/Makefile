.PHONY: prepro test clean

all: test.cu levy.cu velocity.cu position.cu evolve.cu utils.cu
	nvcc test.cu levy.cu velocity.cu position.cu evolve.cu utils.cu -I./include -rdc=true -O3 -arch=sm_60 -o test

prepro: test.cu
	nvcc test.cu -E -I./include > prepro.txt

test:
	./test > out.txt

clean:
	rm -rf test out.txt