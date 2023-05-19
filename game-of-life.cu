#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <iostream>
#include <array>
#include <fstream>
#include <string>

using namespace std;

#define CUDA_FLOAT float
#define BLOCKX_SIZE 16
#define BLOCKY_SIZE 2
#define BLOCKZ_SIZE 2
#define GRIDX_SIZE 2
#define GRIDY_SIZE 2
#define GRIDZ_SIZE 2

__global__ void mat_mult_kern (int *field, int *tmp_field)
{
    int tx = threadIdx.x + blockIdx.x * BLOCKX_SIZE;
    int ty = threadIdx.y + blockIdx.y * BLOCKY_SIZE;    //indexing threads
    int tz = threadIdx.z + blockIdx.z * BLOCKZ_SIZE; 
    int S = 0;

    tmp_field = field;
    for (int i = -1; i < 2; i+=2)
    {
        for (int j = -1; j < 2; j+=2)
        {
            for (int k = -1; k < 2; k+=2)
            {
                S += field[tx+i + (ty+j)*GRIDX_SIZE*BLOCKX_SIZE + (tz+k)*GRIDX_SIZE*BLOCKX_SIZE*GRIDY_SIZE*BLOCKY_SIZE];        //number of neighbours
            }   
        }    
    }

    if (tmp_field[tx + (ty)*GRIDX_SIZE*BLOCKX_SIZE + (tz)*GRIDX_SIZE*BLOCKX_SIZE*GRIDY_SIZE*BLOCKY_SIZE]==1&&4<=S<=7
        ||tmp_field[tx+1 + (ty)*GRIDX_SIZE*BLOCKX_SIZE + (tz)*GRIDX_SIZE*BLOCKX_SIZE*GRIDY_SIZE*BLOCKY_SIZE]==0&&6<=S<=7)
    {                                                                                                                        //deciding life or death
        field[tx + (ty)*GRIDX_SIZE*BLOCKX_SIZE + (tz)*GRIDX_SIZE*BLOCKX_SIZE*GRIDY_SIZE*BLOCKY_SIZE] = 1;
    } 
    else
    {
        field[tx + (ty)*GRIDX_SIZE*BLOCKX_SIZE + (tz)*GRIDX_SIZE*BLOCKX_SIZE*GRIDY_SIZE*BLOCKY_SIZE] = 0;
    }
    if (tx==0||tx==GRIDX_SIZE*BLOCKX_SIZE-1||ty==0||ty==GRIDY_SIZE*BLOCKY_SIZE-1||tz==0||tz==GRIDZ_SIZE*BLOCKZ_SIZE-1)
    {
        field[tx + (ty)*GRIDX_SIZE*BLOCKX_SIZE + (tz)*GRIDX_SIZE*BLOCKX_SIZE*GRIDY_SIZE*BLOCKY_SIZE] = 0;                   //filling buffer
    }
    field[tx + (ty)*GRIDX_SIZE*BLOCKX_SIZE + (tz)*GRIDX_SIZE*BLOCKX_SIZE*GRIDY_SIZE*BLOCKY_SIZE] = 0;
    //printf("%i %i %i %i\n",field[tx + (ty)*GRIDX_SIZE*BLOCKX_SIZE + (tz)*GRIDX_SIZE*BLOCKX_SIZE*GRIDY_SIZE*BLOCKY_SIZE] = 1, tx, ty, tz);
}

void writen(int* field, int x, int y, int z)
{
    field[x + (y)*GRIDX_SIZE*BLOCKX_SIZE + (z)*GRIDX_SIZE*BLOCKX_SIZE*GRIDY_SIZE*BLOCKY_SIZE] = 1;
}

int main(int argc, char **argv)
{   
    printf("[game-of-life] - Starting\n");
    const int xs = GRIDX_SIZE*BLOCKX_SIZE;
    const int ys = GRIDY_SIZE*BLOCKY_SIZE;
    const int zs = GRIDZ_SIZE*BLOCKZ_SIZE; 
    int* field;
    int* tmp_field;                    //initiating
    int* d_field;
    int* d_tmp_field;
    int* test;
    dim3 field_size(xs, ys, zs);
    
    field = (int*)malloc ((GRIDX_SIZE*BLOCKX_SIZE)*(GRIDY_SIZE*BLOCKY_SIZE)*(GRIDZ_SIZE*BLOCKZ_SIZE)*sizeof(int));          //allocating
    tmp_field = (int*)malloc ((GRIDX_SIZE*BLOCKX_SIZE)*(GRIDY_SIZE*BLOCKY_SIZE)*(GRIDZ_SIZE*BLOCKZ_SIZE)*sizeof(int));
    for(int i = 0; i < 10; i++)
    {
        for(int j = 0; j < 10; j++)
        {
           for(int k = 0; k < 10; k++)
            {
                field[i + (j)*GRIDX_SIZE*BLOCKX_SIZE + (k)*GRIDX_SIZE*BLOCKX_SIZE*GRIDY_SIZE*BLOCKY_SIZE] = 0;
            }
        }       
    }
    for(int i = 0; i < xs; i++)
    {
        for(int j = 0; j < ys; j++)
        {
           for(int k = 0; k < zs; k++)
            {
                tmp_field[i + (j)*GRIDX_SIZE*BLOCKX_SIZE + (k)*GRIDX_SIZE*BLOCKX_SIZE*GRIDY_SIZE*BLOCKY_SIZE] = 0;
            }
        }       
    }
    
    //cudaMalloc ((void**) &test, sizeof(int));
    //field[5 + (7)*GRIDX_SIZE*BLOCKX_SIZE + (5)*GRIDX_SIZE*BLOCKX_SIZE*GRIDY_SIZE*BLOCKY_SIZE] = 1;
    //writen(field, 5, 7, 5);
    /*
    write(field, 5, 8, 5);
    write(field, 6, 8, 5);
    write(field, 7, 5, 5);
    write(field, 8, 5, 5);          //start field configuration
    write(field, 8, 6, 5);
    write(field, 5, 7, 6);
    write(field, 5, 8, 6);
    write(field, 6, 8, 6);
    write(field, 7, 5, 6);
    write(field, 8, 5, 6);
    write(field, 8, 6, 6);
    */
    //cudaMalloc ((void **) &d_tmp_field, (GRIDX_SIZE*BLOCKX_SIZE)*(GRIDY_SIZE*BLOCKY_SIZE)*(GRIDZ_SIZE*BLOCKZ_SIZE)*sizeof(int)); 
    //cudaMalloc ((void **) &d_field, (GRIDX_SIZE*BLOCKX_SIZE)*(GRIDY_SIZE*BLOCKY_SIZE)*(GRIDZ_SIZE*BLOCKZ_SIZE)*sizeof(int));                    //allocating device memory
    //cudaMemcpy(d_field, d_tmp_field, (GRIDX_SIZE*BLOCKX_SIZE)*(GRIDY_SIZE*BLOCKY_SIZE)*(GRIDZ_SIZE*BLOCKZ_SIZE)*sizeof(int) ,cudaMemcpyDeviceToDevice);

    dim3 block = dim3(BLOCKX_SIZE, BLOCKY_SIZE, BLOCKZ_SIZE);       //grid parameters
    dim3 grid = dim3(GRIDX_SIZE, GRIDY_SIZE, GRIDZ_SIZE);

    //mat_mult_kern<<<grid, block>>> (d_field, d_tmp_field);   //kernel
    /*
    for(int i = 0; i < (GRIDX_SIZE * BLOCKX_SIZE + 2); i++)
    {
        for(int j = 0; j < (GRIDY_SIZE * BLOCKY_SIZE + 2); j++)
        {
            field[i + j*xs] = 0;
            field[i + j*GRIDX_SIZE*BLOCKX_SIZE + (GRIDX_SIZE*BLOCKX_SIZE+1)*(GRIDY_SIZE*BLOCKY_SIZE+1)*GRIDZ_SIZE*BLOCKZ_SIZE] = 0;
        }       
    }
    for(int i = 0; i < (GRIDX_SIZE * BLOCKX_SIZE + 2); i++)
    {
        for(int j = 0; j < (GRIDZ_SIZE * BLOCKZ_SIZE); j++)
        {
           
            field[i + j*GRIDX_SIZE*BLOCKX_SIZE] = 0;
            field[i + j*GRIDX_SIZE*BLOCKX_SIZE + GRIDX_SIZE*BLOCKX_SIZE*GRIDY_SIZE*BLOCKY_SIZE*GRIDZ_SIZE*BLOCKZ_SIZE] = 0;
        }       
    }
    */
    //cudaMemcpy(d_field, field, (xs)*(ys)*(zs)*sizeof(int) ,cudaMemcpyDeviceToHost);
    for(int i = 0; i < 10; i++)
    {
        for(int j = 0; j < 10; j++)
        {
           for(int k = 0; k < 10; k++)
            {
                cout << field[i + (j)*GRIDX_SIZE*BLOCKX_SIZE + (k)*GRIDX_SIZE*BLOCKX_SIZE*GRIDY_SIZE*BLOCKY_SIZE] << ' ' << i << ' ' << j << ' ' << k << '\n';
            }
        }       
    }
}   