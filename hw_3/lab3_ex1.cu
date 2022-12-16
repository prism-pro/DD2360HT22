
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
  if (inputLength > 0)
  {
    printf("The input length is %d\n", inputLength);
  }
  else
  {
    printf("The input length is an invalid number, please choose a number greater than 0");
    return 255;
  }
  
  
  //@@ Insert code below to allocate Host memory for input and output
  int size_of_type = sizeof(DataType);
  int InputSize=inputLength*size_of_type
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

  //@@ Insert code below to allocate GPU memory here

  cudaMalloc(&deviceInput1, InputSize);
  cudaMalloc(&deviceInput2, InputSize);
  cudaMalloc(&deviceOutput, InputSize);

  //@@ Insert code to below to Copy memory to the GPU here
  time = Timer_start();
  cudaMemcpy(deviceInput1, hostInput1, InputSize, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, InputSize, cudaMemcpyHostToDevice);
  time = Timer_Consumption(time);
  printf("Time comsuption of copying memory of %d data to the GPU is %f s \n ", inputLength, time);

  //@@ Initialize the 1D grid and block dimensions here
  
  int ThreadNumPerBlock = 256; 
  int blockNum = (inputLength + ThreadNumPerBlock - 1) / ThreadNumPerBlock;

  //@@ Launch the GPU Kernel here
  time = Timer_start();
  vecAdd <<<blockNum, ThreadNumPerBlock>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  cudaDeviceSynchronize();
  time = Timer_Consumption(time);
  printf("Time comsuption of computing  %d data in the GPU is %f s \n ", inputLength, time);

  //@@ Copy the GPU memory back to the CPU here
  time = Timer_start();
  cudaMemcpy(hostOutput, deviceOutput, InputSize, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  time = Timer_Consumption(time);
  printf("Time comsuption of copying memory of %d data to the host is %f s \n ", inputLength, time);
 
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

