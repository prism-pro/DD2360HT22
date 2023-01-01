
#include <stdio.h>
#include <sys/time.h>
#include <random>

#define NUM_BINS 4096

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {

//@@ Insert code below to compute histogram of input using shared memory and atomics
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= num_elements)
    return;
  atomicAdd(&bins[input[idx]], 1);

}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {

//@@ Insert code below to clean up bins that saturate at 127
  int bin = blockIdx.x * blockDim.x + threadIdx.x;
  if (bin >= num_bins)
    return;

  if (bins[bin] > 127)
  {
    bins[bin] = 127;
  }

}


int main(int argc, char **argv) {
  
  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *resultRef;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

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
  hostInput = (DataType *)malloc(inputLength * sizeof(DataType));
  hostBins = (DataType *)malloc(NUM_BINS * sizeof(DataType));
  resultRef = (DataType *)malloc(NUM_BINS * sizeof(DataType));
  memset(resultRef, 0, NUM_BINS * sizeof(*resultRef));
  
  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
  srand(inputLength);
  for (int i = 0; i < inputLength; i++)
  {
    hostInput[i] = rand() % NUM_BINS;
  }

  //@@ Insert code below to create reference result in CPU
  for (int i = 0; i < inputLength; i++)
  {
    int j = hostInput[i];
    if (resultRef[j] < 127)
    {
      resultRef[j] += 1;
    }
  }

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput, inputLength * sizeof(DataType));
  cudaMalloc(&deviceBins, NUM_BINS * sizeof(DataType));

  //@@ Insert code to Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);

  //@@ Insert code to initialize GPU results
  cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));

  //@@ Initialize the grid and block dimensions here
  int TPB = 32;
  int blockNum = (inputLength + TPB - 1) / TPB;

  //@@ Launch the GPU Kernel here
  histogram_kernel<<<blockNum, TPB>>>(deviceInput, deviceBins, inputLength, NUM_BINS);

  //@@ Initialize the second grid and block dimensions here
  //just keep it the same

  //@@ Launch the second GPU Kernel here
  convert_kernel<<<blockNum, TPB>>>(deviceBins, NUM_BINS);

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

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
  cudaFree(deviceInput);
  cudaFree(deviceBins);

  //@@ Free the CPU memory here
  free(hostInput);
  free(hostBins);
  free(resultRef);

  return 0;
}

