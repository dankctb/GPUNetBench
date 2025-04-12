all:
			nvcc --gpu-architecture=compute_70 --gpu-code=sm_70 -Xptxas -dlcm=cg -O3 -o BW main.cu