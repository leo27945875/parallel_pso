all: test.cu levy.cu utils.cu
	nvcc test.cu levy.cu utils.cu -I./include -rdc=true -O3 -o test

prepro: test.cu
	nvcc test.cu -E -I./include > prepro.txt

clean:
	rm -rf test