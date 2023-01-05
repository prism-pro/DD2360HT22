
#include <stdio.h>
#include <sys/time.h>

#define DataType double

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
    int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) 
    out[i] = in1[i] + in2[i];

}
//@@ Insert code to implement timer start
double Timer_start(){ 
    struct timeval tos;
    gettimeofday(&tos,NULL);
    return ((double)tos.tv_sec + (double)tos.tv_usec / 1.e6);
}

//@@ Insert code to implement timer stop
double Timer_Consumption(double time_of_start){ 
    struct timeval toe;
    gettimeofday(&toe,NULL);
    return (((double)toe.tv_sec + (double)toe.tv_usec / 1.e6)-time_of_start);
}

int main(int argc, char **argv) {
  
  int inputLength;
  int S_seg;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  double time;
  //@@ Insert code below to read in inputLength from args
  inputLength = atoi(argv[1]);
  S_seg = atoi(argv[2]);
  if (inputLength > 0 & S_seg > 0)
  {
    printf("The input length is %d, and the size of segments is %d\n", inputLength,S_seg);
  }
  else
  {
    printf("The input length or segment size is an invalid number, please choose a number greater than 0");
    return 255;
  }
  int N_seg= ceil(inputLength/S_seg);

  //@@ Insert code below to allocate Host memory for input and output
  int size_of_type = sizeof(DataType);
  int InputSize=inputLength*size_of_type;
  hostInput1 = (DataType*) malloc(InputSize);
  hostInput2 = (DataType*) malloc(InputSize);
  hostOutput = (DataType*) malloc(InputSize);
  resultRef = (DataType*) malloc(InputSize);
  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  srand(inputLength);
  int MAX=10;
  for (int i = 0; i < inputLength; i++) {
      DataType randomNumber1 = (DataType) (rand()%MAX) ;
      DataType randomNumber2 = (DataType) (rand()%MAX) ;
      hostInput1[i] = randomNumber1;
      hostInput2[i] = randomNumber2;
      resultRef[i] = randomNumber1 + randomNumber2;
  }
  // create streams here:
cudaStream_t streams[N_seg];
for (int i = 0; i < N_seg; i++) {
  cudaStreamCreate(&streams[i]);
}

  //@@ Insert code below to allocate GPU memory here

  cudaMalloc(&deviceInput1, InputSize);
  cudaMalloc(&deviceInput2, InputSize);
  cudaMalloc(&deviceOutput, InputSize);

  //@@ launch streams copy here
  dim3 dimGrid(ceil(S_seg / 256));
  dim3 dimBlock(256);
  for (int i = 0; i < N_seg; i++)
  {	
  	offset = i * S_seg;
      	cudaMemcpyAsync(deviceInput1 + offset, hostInput1 + offset, S_seg * sizeof(DataType), 
                        cudaMemcpyHostToDevice, streams[i]);
	cudaMemcpyAsync(deviceInput2 + offset, hostInput2 + offset, S_seg * sizeof(DataType), 
                        cudaMemcpyHostToDevice, streams[i]); 

  }
 
  //@@ Launch the stream Kernels here
  for (int i = 0; i < N_seg; i++)
  {
    int offset = i * S_seg;
    	vecAdd<<<dimGrid, dimBlock, 0, streams[i]>>>
            (deviceInput1 + offset, deviceInput2 + offset, deviceOutput + offset, S_seg);
  }
  
  cudaDeviceSynchronize();
  time = Timer_Consumption(time);

  //@@ Copy the stream memory back to the CPU here
  time = Timer_start();
  for (int i = 0; i < N_seg; i++)
  {
    int offset = i * S_seg;
    	cudaMemcpyAsync(hostOutput + offset, deviceOutput + offset, S_seg * sizeof(DataType), 
                        cudaMemcpyDeviceToHost, streams[i]);
  }
  cudaDeviceSynchronize();
  time = Timer_Consumption(time);
  printf("Time comsuption of copying memory of %d data to the host is %f s \n ", inputLength, time);
     for(int i = 0; i < N_seg; i++) {
        cudaStreamDestroy(streams[i]);
    }
  //@@ Insert code below to compare the output with the reference

  bool equal = true;
  for (int i = 0; i < inputLength; i++) {
    if (fabs(hostOutput[i] - resultRef[i]) > 1e-8) {
      equal = false;
      break;
    }
  }

  if (equal == true)
  {
    printf("the results are equal.");
  }
  else 
  {
    printf("some of the results are unequal.");
  }
  
  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);
  //@@ Free the CPU memory here
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  free(resultRef);
  return 0;
}
