#include <stdio.h>
#include <cuda_runtime.h>


//树形归约
__global__ void reduce_base(float* input, float* output, int n){
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    smem[tid] = (gid<n)?input[gid]:0.0f;
    __syncthreads();

    for(int step = blockDim.x /2;step>0;step>>=1){
        if(tid<step){
            smem[tid]+=smem[tid+step];
        }
        __syncthreads();
    }

    if(tid == 0)output[blockIdx.x] = smem[0];
}

// V1 ：stride index方法 消除Warp Divergence
__global__ void reduce_v1(float* input,float* output,int n){
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x*blockDim.x + threadIdx.x;

    smem[tid] = (gid<n)?input[gid] : 0.0f;
    __syncthreads();

    // 
    for(unsigned int s = 1;s<blockDim.x;s*=2){
        int index = threadIdx.x *2*s;
        if(index < blockDim.x){
            smem[index] += smem[index+s];
        }
        __syncthreads();
    }
    if(tid == 0){
        output[blockIdx.x] = smem[0];
    }
}

 
//V2：步长从大到小，同时消除 Warp Divergence 与 Bank Conflict
__global__ void reduce_v2(float * input,float *output,int n){
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockDim.x*blockIdx.x +threadIdx.x;

    smem[tid] = (gid>n)? input[gid]: 0.0f;

    __syncthreads();

    for(unsigned int s = blockDim.x / 2; s>0;s>>=1){
        if (tid < s) {
            smem[tid] += smem[tid+s];
        }
        __syncthreads();
    }


    if(tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}

//V3:每线程处理2个元素，减少空闲线程
__global__ void reduce_v3(float * input,float * output,int n){
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockDim.x* blockIdx.x+ threadIdx.x;

    //每个线程加载2个相距blockDim.x 的元素并求和
    float val = 0.0f;
    if(gid < n) val +=input[gid];
    if(gid+blockDim.x < n ) val+=input[gid + blockDim.x];
    smem[tid] = val;
    __syncthreads();

    for(unsigned int step = blockDim.x /2 ; step > 0 ; step>>=1){
        if(tid < step) {
            smem[tid] +=smem[tid+step];
        }
        __syncthreads();
    }

    if(tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}

//V4:展开最后一个warp
__device__ void warpReduce(volatile float* smem,int tid){
    smem[tid]+=smem[tid+32];
    smem[tid]+=smem[tid+16];
    smem[tid]+=smem[tid+8];
    smem[tid]+=smem[tid+4];
    smem[tid]+=smem[tid+2];
    smem[tid]+=smem[tid+1];
}

//V4:每线程处理多个元素
__global__ void reduce_v4(float* input, float* output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // 每线程处理 2 个元素（继承自 V3）
    float val = 0.0f;
    if (gid < n)              val += input[gid];
    if (gid + blockDim.x < n) val += input[gid + blockDim.x];
    smem[tid] = val;
    __syncthreads();

    // 规约循环仅执行到 step > 32
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    // 最后一个 Warp 内的规约，无需 __syncthreads()
    if (tid < 32) {
        warpReduce(smem, tid);
    }

    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}

// V6: Warp Shuffle + 两级规约
// Warp 内规约辅助函数
__device__ float warpReduceSum(float val) {
    // 每次将右半边的值加到左半边
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;  // lane 0 持有最终结果
}

__global__ void reduce_v6(float* input, float* output, int n) {
    int tid  = threadIdx.x;
    int gid  = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    int lane = tid % 32;      // 线程在 Warp 内的编号（0~31）
    int wid  = tid / 32;      // 该线程属于哪个 Warp

    // 每线程处理 2 个元素
    float val = 0.0f;
    if (gid < n)              val += input[gid];
    if (gid + blockDim.x < n) val += input[gid + blockDim.x];

    // 第一级：Warp 内规约
    val = warpReduceSum(val);

    // 将每个 Warp 的结果（仅 lane 0 有效）存入 Shared Memory
    __shared__ float warp_results[32];  // 最多 32 个 Warp（1024/32）
    if (lane == 0) {
        warp_results[wid] = val;
    }
    __syncthreads();

    // 第二级：Warp 间规约（用 Warp 0 处理）
    int num_warps = blockDim.x / 32;
    if (wid == 0) {
        val = (lane < num_warps) ? warp_results[lane] : 0.0f;
        val = warpReduceSum(val);
    }

    if (tid == 0) output[blockIdx.x] = val;
}



int main(){
    return 0;
}