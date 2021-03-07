test_convective: test_convective.cu
	nvcc test_convective.cu -o test_convective
test_cuda: test.cu
	nvcc test.cu -o test

