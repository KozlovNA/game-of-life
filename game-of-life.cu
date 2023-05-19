#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <iostream>
#include <array>
#include <fstream>
#include <string>

using namespace std;

#define CUDA_FLOAT float
#define BLOCKX_SIZE 4
#define BLOCKY_SIZE 4
#define BLOCKZ_SIZE 4
#define GRIDX_SIZE 8
#define GRIDY_SIZE 8
#define GRIDZ_SIZE 8

__global__ void life_kern(int *field, int *tmp_field)
{   
    int tx = threadIdx.x + blockIdx.x * BLOCKX_SIZE;
    int ty = threadIdx.y + blockIdx.y * BLOCKY_SIZE;    //indexing threads
    int tz = threadIdx.z + blockIdx.z * BLOCKZ_SIZE; 
    int S = 0;
    //printf("%d, %d, %d, %d\n", tmp_field[tx + (ty)*GRIDX_SIZE*BLOCKX_SIZE + (tz)*GRIDX_SIZE*BLOCKX_SIZE*GRIDY_SIZE*BLOCKY_SIZE], tx, ty, tz);

    for (int i = -1; i < 2; i+=1)
    {
        for (int j = -1; j < 2; j+=1)
        {
            for (int k = -1; k < 2; k+=1)
            {   
                S += tmp_field[tx+i + (ty+j)*GRIDX_SIZE*BLOCKX_SIZE + (tz+k)*GRIDX_SIZE*BLOCKX_SIZE*GRIDY_SIZE*BLOCKY_SIZE];        //number of neighbours
            }   
        }    
    }
    S-=tmp_field[tx + (ty)*GRIDX_SIZE*BLOCKX_SIZE + (tz)*GRIDX_SIZE*BLOCKX_SIZE*GRIDY_SIZE*BLOCKY_SIZE];
    //if (tx==1&&ty==0&&tz==2)
    //printf("%d, %d, %d, %d\n", S, tx, ty, tz);
    if (((tmp_field[tx + (ty)*GRIDX_SIZE*BLOCKX_SIZE + (tz)*GRIDX_SIZE*BLOCKX_SIZE*GRIDY_SIZE*BLOCKY_SIZE]==1)&&(4<=S)&&(S<=7))
        ||((tmp_field[tx + (ty)*GRIDX_SIZE*BLOCKX_SIZE + (tz)*GRIDX_SIZE*BLOCKX_SIZE*GRIDY_SIZE*BLOCKY_SIZE]==0)&&(6<=S)&&(S<=7)))
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
    
    field = (int*)calloc ((GRIDX_SIZE*BLOCKX_SIZE)*(GRIDY_SIZE*BLOCKY_SIZE)*(GRIDZ_SIZE*BLOCKZ_SIZE), sizeof(int));          //allocating
    tmp_field = (int*)calloc ((GRIDX_SIZE*BLOCKX_SIZE)*(GRIDY_SIZE*BLOCKY_SIZE)*(GRIDZ_SIZE*BLOCKZ_SIZE), sizeof(int));
    
    
    //cudaMalloc ((void**) &test, sizeof(int));

    writen(field, 5, 7, 5);
    writen(field, 5, 8, 5);
    writen(field, 6, 8, 5);
    writen(field, 7, 5, 5);
    writen(field, 8, 5, 5);          //start field configuration
    writen(field, 8, 6, 5);
    writen(field, 5, 7, 6);
    writen(field, 5, 8, 6);
    writen(field, 6, 8, 6);
    writen(field, 7, 5, 6);
    writen(field, 8, 5, 6);
    writen(field, 8, 6, 6);
    /*
    writen(field, 8, 6, 5);
    writen(field, 8, 6, 7);
    writen(field, 8, 4, 6);
    */
    ofstream out;
    out.open("/home/starman/CUDA/game-of-life/visualisation/lnx64-compiled/data.txt");
    if (out.is_open())
    {   
        for(int i = 0; i < xs; i++)
        {
            for(int j = 0; j < ys; j++)
            {
                for(int k = 0; k < zs; k++)
                {   
                    if (field[i + (j)*GRIDX_SIZE*BLOCKX_SIZE + (k)*GRIDX_SIZE*BLOCKX_SIZE*GRIDY_SIZE*BLOCKY_SIZE]!=0)
                        out << field[i + (j)*GRIDX_SIZE*BLOCKX_SIZE + (k)*GRIDX_SIZE*BLOCKX_SIZE*GRIDY_SIZE*BLOCKY_SIZE] << ' ' << i << ' ' << j << ' ' << k << '\n';
                }
            }       
        }
        out << 0 << ' ' << 0 << ' ' << 0 << ' ' << 0 << ' '<< '\n'; 
    }

    cudaMalloc ((void **) &d_tmp_field, (GRIDX_SIZE*BLOCKX_SIZE)*(GRIDY_SIZE*BLOCKY_SIZE)*(GRIDZ_SIZE*BLOCKZ_SIZE)*sizeof(int)); 
    cudaMalloc ((void **) &d_field, (GRIDX_SIZE*BLOCKX_SIZE)*(GRIDY_SIZE*BLOCKY_SIZE)*(GRIDZ_SIZE*BLOCKZ_SIZE)*sizeof(int));
    cudaMemcpy(d_field, field, (GRIDX_SIZE*BLOCKX_SIZE)*(GRIDY_SIZE*BLOCKY_SIZE)*(GRIDZ_SIZE*BLOCKZ_SIZE)*sizeof(int) ,cudaMemcpyHostToDevice);
    cudaMemcpy(d_tmp_field, field, (GRIDX_SIZE*BLOCKX_SIZE)*(GRIDY_SIZE*BLOCKY_SIZE)*(GRIDZ_SIZE*BLOCKZ_SIZE)*sizeof(int) ,cudaMemcpyHostToDevice);                    //allocating device memory
    //cudaMemcpy(d_field, d_tmp_field, (GRIDX_SIZE*BLOCKX_SIZE)*(GRIDY_SIZE*BLOCKY_SIZE)*(GRIDZ_SIZE*BLOCKZ_SIZE)*sizeof(int) ,cudaMemcpyDeviceToDevice);

    dim3 block = dim3(BLOCKX_SIZE, BLOCKY_SIZE, BLOCKZ_SIZE);       //grid parameters
    dim3 grid = dim3(GRIDX_SIZE, GRIDY_SIZE, GRIDZ_SIZE);

    //life_kern<<<grid, block>>> (d_field, d_tmp_field);   //kernel

    //cudaMemcpy(d_field, field, (xs)*(ys)*(zs)*sizeof(int) ,cudaMemcpyDeviceToHost);
    for (int step = 0; step < 30; step++)
    {   
        cudaMemcpy(d_tmp_field, d_field, (GRIDX_SIZE*BLOCKX_SIZE)*(GRIDY_SIZE*BLOCKY_SIZE)*(GRIDZ_SIZE*BLOCKZ_SIZE)*sizeof(int) ,cudaMemcpyDeviceToDevice);
        life_kern<<<grid, block>>> (d_field, d_tmp_field);
        cudaMemcpy(field, d_field, (xs)*(ys)*(zs)*sizeof(int) ,cudaMemcpyDeviceToHost);
        if (out.is_open())
        {   
            for(int i = 0; i < xs; i++)
            {
                for(int j = 0; j < ys; j++)
                {
                    for(int k = 0; k < zs; k++)
                    {   if (field[i + (j)*GRIDX_SIZE*BLOCKX_SIZE + (k)*GRIDX_SIZE*BLOCKX_SIZE*GRIDY_SIZE*BLOCKY_SIZE]!=0)
                            out << field[i + (j)*GRIDX_SIZE*BLOCKX_SIZE + (k)*GRIDX_SIZE*BLOCKX_SIZE*GRIDY_SIZE*BLOCKY_SIZE] << ' ' << i << ' ' << j << ' ' << k << '\n';
                    }
                }       
            }
            out << 0 << ' ' << 0 << ' ' << 0 << ' ' << 0 << ' '<< '\n'; 
        } 
    }
    out.close();

    /*
    if (out.is_open())
    {   
        for(int i = 0; i < xs; i++)
        {
            for(int j = 0; j < ys; j++)
            {
                for(int k = 0; k < zs; k++)
                {   if (field[i + (j)*GRIDX_SIZE*BLOCKX_SIZE + (k)*GRIDX_SIZE*BLOCKX_SIZE*GRIDY_SIZE*BLOCKY_SIZE]!=0)
                        out << field[i + (j)*GRIDX_SIZE*BLOCKX_SIZE + (k)*GRIDX_SIZE*BLOCKX_SIZE*GRIDY_SIZE*BLOCKY_SIZE] << ' ' << i << ' ' << j << ' ' << k << '\n';
                }
            }       
        }
        out << 0 << ' ' << 0 << ' ' << 0 << ' ' << 0 << ' '<< '\n'; 
    }
    out.close();
    */
    //for(int i = 11130; i < 12000; i++) cout << field[i] << ' ' << i << '\n'; 
    //cout << "\n"<< xs*ys*zs << " " << (xs-1) + (ys-1)*xs + (zs-1)*xs*ys;
}   