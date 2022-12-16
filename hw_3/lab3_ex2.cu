
#include <stdio.h>
#include <sys/time.h>

#define DataType double
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
// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){
  //@@ Insert code to implement matrix multiplication here
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if ((col >= numBColumns) || (row >= numARows))
    return;

  //DataType tmpSum = 0.0;
  for (int k = 0; k < numAColumns; k++)
  {
    C[row * numBColumns + col] += A[row * numAColumns + k] * B[k * numBColumns + col];
  }
  
}

int main(int argc, char **argv) {
  
  DataType *hostA; // The A matrix
  DataType *hostB; // The B matrix
  DataType *hostC; // The output C matrix
  DataType *resultRef; // The reference result
  DataType *deviceA;
  DataType *deviceB;
  DataType *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;
  int numCColumns;

  //@@ Insert code below to read in numARows, numAColumns, numBColumns from args
  numARows = atoi(argv[1]);
  numAColumns = atoi(argv[2]);
  numBRows = atoi(argv[3]);
  numBColumns = atoi(argv[4]);
  if (numAColumns == numBRows)
  {
    numCRows=numARows;
    numCColumns= numBColumns;
    printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  }
  else
  {
    printf("incorrect dimension\n");
    return 255;
  }
  
  //@@ Insert code below to allocate Host memory for input and output
  int SizeA = numARows * numAColumns * sizeof(DataType);
  int SizeB = numBRows * numBColumns * sizeof(DataType);
  int SizeC = numCRows * numCColumns * sizeof(DataType);
  hostA = (DataType *)malloc(SizeA);
  hostB = (DataType *)malloc(SizeB);
  hostC = (DataType *)malloc(SizeC);
  resultRef = (DataType *)malloc(SizeC); 
  //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
  srand(time(0));
  int MAX=10;
 for (int i = 0; i < numARows; i++)
  {
    for (int j = 0; j < numAColumns; j++)
    {
      DataType randomNumber = (DataType)(rand() %MAX);
      hostA[i * numAColumns + j] = randomNumber;
    }
  }

  for (int i = 0; i < numBRows; i++)
  {
    for (int j = 0; j < numBColumns; j++)
    {
      DataType randomNumber = (DataType)(rand() %MAX);
      hostB[i * numBColumns + j] = randomNumber;
    }
  }

  for (int i = 0; i < numARows; i++)
  {
    for (int j = 0; j < numBColumns; j++)
    {
      resultRef[i * numBColumns + j] = 0.0;
      for (int k = 0; k < numAColumns; k++)
      {
        resultRef[i * numBColumns + j] += hostA[i * numAColumns + k] * hostB[k * numBColumns + j];
      }
    }
  }

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceA, SizeA);
  cudaMalloc(&deviceB, SizeB);
  cudaMalloc(&deviceC, SizeC);

  //@@ Insert code to below to Copy memory to the GPU here
  double time = Timer_start();
  cudaMemcpy(deviceA, hostA, SizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, SizeB, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  double time_h2d = Timer_Consumption(time);
  printf("Time to copy memory from  Host to Device: %f\n", time_h2d);

  //@@ Initialize the grid and block dimensions here
  int threadPerBlockX = 32;
  int threadPerBlockY = 32;
  int blockNumX = (numCColumns + threadPerBlockX - 1) / threadPerBlockX;
  int blockNumY = (numCRows + threadPerBlockY - 1) / threadPerBlockY;
  //printf("threads per block x: %i y: %i\n", threadPerBlockX, threadPerBlockY);
  //printf("blocks num x: %i, y: %i \n", blockNumX, blockNumY);

  //@@ Launch the GPU Kernel here
  time = Timer_start();
  gemm<<<dim3(blockNumX, blockNumY, 1), dim3(threadPerBlockX, threadPerBlockY, 1)>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);
  cudaDeviceSynchronize();
  double time_k = Timer_Consumption(time);
  printf("Time of running kernel: %f\n", time_k); 

  //@@ Copy the GPU memory back to the CPU here
  time = Timer_start();
  cudaMemcpy(hostC, deviceC, SizeC, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  double time_d2h = Timer_Consumption(time);
  printf("Time to copy memory from  Device to Host: %f\n", time_d2h);
  //@@ Insert code below to compare the output with the reference
  bool equal = true;
  for (int i = 0; i < numCRows; ++i)
  {
    for (int j = 0; j < numCColumns; ++j)
    {
      if (fabs(hostC[i * numCColumns + j] - resultRef[i * numCColumns + j]) > 1e-7)
      {
        equal = false;
        printf("%d row %d col host is %f, ref is %f \n", i,j,hostC[i * numCColumns + j],
        resultRef[i * numCColumns + j]);
        //break;
      }
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
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  //@@ Free the CPU memory here
  free(hostA);
  free(hostB);
  free(hostC);
  free(resultRef);

  return 0;
}

