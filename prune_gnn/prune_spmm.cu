#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include <torch/extension.h>

using namespace std;

const int WARP_SIZE = 32;
const int WARPS_PER_BLOCK = 8;
const int BLOCK = WARPS_PER_BLOCK * WARP_SIZE;

#define WARP 8192

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char *const func, const char *const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    }
}

__device__ inline void atomicAdd_F(float *address, float value)
{
    float old = value;
    while ((old = atomicExch(address, atomicExch(address, 0.0f) + old)) != 0.0f)
        ;
}

struct CoordinateT
{
    int x;
    int y;
};
CoordinateT MergePathSearch(int diagonal, int *RP, int *NZ_INDICES, int num_rows, int nnz);
std::vector<int *> generate_mp_sched(int num_mp_threads, int num_slots, int using_slot);

__global__ void spmm_forward_cuda_kernel_mp(
    float *output,
    float *input,
    int *row_pointers,
    int *column_index,
    float *degrees,
    int *startPartial_nz_start,
    int *endPartial_nz_start,
    int *startPartial_nnz,
    int *endPartial_nnz,
    int *startPartial_rowId,
    int *endPartial_rowId,
    int *startComplete_rowId,
    int *endComplete_rowId,
    int num_nodes,
    int dim,
    int num_warps,
    int threads_per_warp, int slots_per_mp_thread,
    int lanes_per_mp_thread, int usable_lanes_per_warp, int factor)
{
    if (dim < WARP_SIZE) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x; // global thread-id
        int warpId = tid / WARP_SIZE;                    // global warp-id
        int laneid = threadIdx.x % WARP_SIZE;            // warp thread-id -- laneid

        int mp_thread_id, slot_id, global_slot_id;

        if (warpId < num_warps)
        {

            if (laneid >= usable_lanes_per_warp)
                return;
            mp_thread_id = laneid / lanes_per_mp_thread;
            laneid = laneid % lanes_per_mp_thread; // laneid within warp

            slot_id = laneid / dim; // slot id within warp
            laneid = laneid % dim;  // lane id within slot
            global_slot_id = (threads_per_warp * warpId + mp_thread_id) * slots_per_mp_thread + slot_id;

            int start = startPartial_rowId[global_slot_id];
            int end = endPartial_rowId[global_slot_id];
            int startPartial_nz_index = startPartial_nz_start[global_slot_id];
            int startPartial_nnzs = startPartial_nnz[global_slot_id];
            int endPartial_nz_index = endPartial_nz_start[global_slot_id];
            int endPartial_nnzs = endPartial_nnz[global_slot_id];
            int comp_start = startComplete_rowId[global_slot_id];
            int comp_end = endComplete_rowId[global_slot_id];

            float partial_results_start = 0;
            float partial_results_end = 0;
            float output_temp = 0;
            float degree_norm_inv = 0;
            float src_norm = 0;
            int index = 0;
            int nnz = 0;
            int nz_start = 0;

            if (startPartial_nz_index != 0)
            {
                src_norm = degrees[start];
                for (int j = 0; j < startPartial_nnzs; j++)
                {
                    index = column_index[startPartial_nz_index++];
                    degree_norm_inv = __fmaf_rn(src_norm, degrees[index], 0);
                    partial_results_start += __fmaf_rn(degree_norm_inv, input[index * dim + laneid], 0);
                }
                atomicAdd_F((float *)&output[start * dim + laneid], partial_results_start);
            }

            for (int i = comp_start; i < comp_end; i += slots_per_mp_thread)
            {
                src_norm = degrees[i];
                output_temp = 0.0f;

                nnz = row_pointers[i + 1] - row_pointers[i];
                nz_start = row_pointers[i];

                #pragma unroll
                for (int j = 0; j < nnz; j++)
                {
                    index = column_index[nz_start];
                    degree_norm_inv = __fmaf_rn(src_norm, degrees[index], 0);
                    output_temp += __fmaf_rn(degree_norm_inv, input[index * dim + laneid], 0);
                    nz_start++;
                }

                output[i * dim + laneid] = output_temp;
            }

            if (endPartial_nz_index != 0)
            {
                src_norm = degrees[end];

                #pragma unroll
                for (int j = 0; j < endPartial_nnzs; j++)
                {
                    index = column_index[endPartial_nz_index++];
                    degree_norm_inv = __fmaf_rn(src_norm, degrees[index], 0);
                    partial_results_end += __fmaf_rn(degree_norm_inv, input[index * dim + laneid], 0);
                }

                atomicAdd_F((float *)&output[end * dim + laneid], partial_results_end);
            }
            return;
        }
    }
    else{ //if dim >= WARPSIZE

        int tid = blockIdx.x * blockDim.x + threadIdx.x; // global thread-id
        int warpId = tid / WARP_SIZE;                    // global warp-id
        int laneid = threadIdx.x % WARP_SIZE;            // warp thread-id -- laneid

        int global_slot_id;

        if (warpId < num_warps)
        {

            laneid += (warpId % factor) * WARP_SIZE;
            if (laneid >= dim)
                return;
            warpId = warpId / factor;
            global_slot_id = warpId;

            int start = startPartial_rowId[global_slot_id];
            int end = endPartial_rowId[global_slot_id];
            int startPartial_nz_index = startPartial_nz_start[global_slot_id];
            int startPartial_nnzs = startPartial_nnz[global_slot_id];
            int endPartial_nz_index = endPartial_nz_start[global_slot_id];
            int endPartial_nnzs = endPartial_nnz[global_slot_id];

            float partial_results_start = 0;
            float partial_results_end = 0;
            float output_temp = 0;
            float degree_norm_inv = 0;
            float src_norm = 0;
            int index = 0;
            int nnz = 0;
            int nz_start = 0;

            if (startPartial_nz_index != 0)
            {
                src_norm = degrees[start];

                for (int j = 0; j < startPartial_nnzs; j++)
                {
                    index = column_index[startPartial_nz_index++];
                    degree_norm_inv = __fmaf_rn(src_norm, degrees[index], 0);
                    partial_results_start += __fmaf_rn(degree_norm_inv, input[index * dim + laneid], 0);
                }
                atomicAdd_F(&output[start * dim + laneid], partial_results_start);
                start = start + 1;
            }

            for (int i = start; i < end; i++)
            {
                src_norm = degrees[i];
                output_temp = 0.0f;

                nnz = row_pointers[i + 1] - row_pointers[i];
                nz_start = row_pointers[i];

                #pragma unroll
                for (int j = 0; j < nnz; j++)
                {
                    index = column_index[nz_start];
                    degree_norm_inv = __fmaf_rn(src_norm, degrees[index], 0);
                    output_temp += __fmaf_rn(degree_norm_inv, input[index * dim + laneid], 0);
                    nz_start++;
                }

                output[i * dim + laneid] = output_temp;
            }

            if (endPartial_nz_index != 0)
            {
                src_norm = degrees[end];  

                #pragma unroll
                for (int j = 0; j < endPartial_nnzs; j++)
                {
                    index = column_index[endPartial_nz_index++];
                    degree_norm_inv = __fmaf_rn(src_norm, degrees[index], 0);
                    partial_results_end += __fmaf_rn(degree_norm_inv, input[index * dim + laneid], 0);
                }
                atomicAdd_F(&output[end * dim + laneid], partial_results_end);
            }

            return;
        }
    }
}


__global__ void spmm_forward_cuda_kernel_mp_gin(
    float *output,
    float *input,
    int *row_pointers,
    int *column_index,
    int *startPartial_nz_start,
    int *endPartial_nz_start,
    int *startPartial_nnz,
    int *endPartial_nnz,
    int *startPartial_rowId,
    int *endPartial_rowId,
    int *startComplete_rowId,
    int *endComplete_rowId,
    int num_nodes,
    int dim,
    int num_warps,
    int threads_per_warp, int slots_per_mp_thread,
    int lanes_per_mp_thread, int usable_lanes_per_warp, 
    int factor, float epsilon)
{

    if (dim < WARP_SIZE) {
    
        int tid = blockIdx.x * blockDim.x + threadIdx.x; // global thread-id
        int warpId = tid / WARP_SIZE;                    // global warp-id
        int laneid = threadIdx.x % WARP_SIZE;            // warp thread-id -- laneid

        int mp_thread_id, slot_id, global_slot_id;

        if (warpId < num_warps)
        {

            if (laneid >= usable_lanes_per_warp)
                return;
            mp_thread_id = laneid / lanes_per_mp_thread;
            laneid = laneid % lanes_per_mp_thread; // laneid within warp

            slot_id = laneid / dim; // slot id within warp
            laneid = laneid % dim;  // lane id within slot
            global_slot_id = (threads_per_warp * warpId + mp_thread_id) * slots_per_mp_thread + slot_id;

            int start = startPartial_rowId[global_slot_id];
            int end = endPartial_rowId[global_slot_id];
            int startPartial_nz_index = startPartial_nz_start[global_slot_id];
            int startPartial_nnzs = startPartial_nnz[global_slot_id];
            int endPartial_nz_index = endPartial_nz_start[global_slot_id];
            int endPartial_nnzs = endPartial_nnz[global_slot_id];
            int comp_start = startComplete_rowId[global_slot_id];
            int comp_end = endComplete_rowId[global_slot_id];

            float partial_results_start = 0;
            float partial_results_end = 0;
            float output_temp = 0;

            int index = 0;
            int nnz = 0;
            int nz_start = 0;

            if (startPartial_nz_index != 0)
            {
                for (int j = 0; j < startPartial_nnzs; j++)
                {
                    index = column_index[startPartial_nz_index++];
                    partial_results_start += input[index * dim + laneid];
                }
                atomicAdd_F((float *)&output[start * dim + laneid], epsilon * partial_results_start);
            }

            for (int i = comp_start; i < comp_end; i += slots_per_mp_thread)
            {
                output_temp = 0.0f;

                nnz = row_pointers[i + 1] - row_pointers[i];
                nz_start = row_pointers[i];

    #pragma unroll
                for (int j = 0; j < nnz; j++)
                {
                    index = column_index[nz_start];
                    output_temp += input[index * dim + laneid];
                    nz_start++;
                }

                output[i * dim + laneid] = epsilon * output_temp;
            }

            if (endPartial_nz_index != 0)
            {

    #pragma unroll
                for (int j = 0; j < endPartial_nnzs; j++)
                {
                    index = column_index[endPartial_nz_index++];
                    partial_results_end += input[index * dim + laneid];
                }

                atomicAdd_F((float *)&output[end * dim + laneid], epsilon * partial_results_end);
            }
            return;
        }

    }
    else {

        int tid = blockIdx.x * blockDim.x + threadIdx.x; // global thread-id
        int warpId = tid / WARP_SIZE;                    // global warp-id
        int laneid = threadIdx.x % WARP_SIZE;            // warp thread-id -- laneid

        int global_slot_id;

        if (warpId < num_warps)
        {

            laneid += (warpId % factor) * WARP_SIZE;
            if (laneid >= dim)
                return;
            warpId = warpId / factor;
            global_slot_id = warpId;

            int start = startPartial_rowId[global_slot_id];
            int end = endPartial_rowId[global_slot_id];
            int startPartial_nz_index = startPartial_nz_start[global_slot_id];
            int startPartial_nnzs = startPartial_nnz[global_slot_id];
            int endPartial_nz_index = endPartial_nz_start[global_slot_id];
            int endPartial_nnzs = endPartial_nnz[global_slot_id];

            float partial_results_start = 0;
            float partial_results_end = 0;
            float output_temp = 0;
        
            int index = 0;
            int nnz = 0;
            int nz_start = 0;

            if (startPartial_nz_index != 0)
            {
        
                for (int j = 0; j < startPartial_nnzs; j++)
                {
                    index = column_index[startPartial_nz_index++];
                
                    partial_results_start += input[index * dim + laneid];
                }
                atomicAdd_F(&output[start * dim + laneid], epsilon * partial_results_start);
                start = start + 1;
            }

            for (int i = start; i < end; i++)
            {
                
                output_temp = 0.0f;

                nnz = row_pointers[i + 1] - row_pointers[i];
                nz_start = row_pointers[i];

    #pragma unroll
                for (int j = 0; j < nnz; j++)
                {
                    index = column_index[nz_start];
                    output_temp += input[index * dim + laneid];
                    nz_start++;
                }

                output[i * dim + laneid] = epsilon * output_temp;
            }

            if (endPartial_nz_index != 0)
            {
            
    #pragma unroll
                for (int j = 0; j < endPartial_nnzs; j++)
                {
                    index = column_index[endPartial_nz_index++];
                    partial_results_end += input[index * dim + laneid];
                }
                atomicAdd_F(&output[end * dim + laneid], epsilon * partial_results_end);
            }

            return;
        }

    }

}

CoordinateT MergePathSearch(int diagonal, int *RP, int *NZ_INDICES, int num_rows, int nnz)
{

    int x_min = max(diagonal - nnz, 0);
    int x_max = min(diagonal, num_rows);

    while (x_min < x_max)
    {
        // so this is div by 2
        int pivot = (x_min + x_max) >> 1;
        if (RP[pivot] <= NZ_INDICES[diagonal - pivot - 1])
        {
            x_min = pivot + 1;
        }
        else
        {
            x_max = pivot;
        }
    }
    return CoordinateT{min(x_min, num_rows), diagonal - x_min};
}

std::vector<int *> generate_mp_sched(int num_mp_threads, int num_slots, int using_slot, int NNZ, int R_PTR_NUM, int *row_ptr)
{

    int total_slots = num_mp_threads * num_slots;
    int *startPartial_nz_start = new int[total_slots];
    int *endPartial_nz_start = new int[total_slots];
    int *startPartial_nnz = new int[total_slots];
    int *endPartial_nnz = new int[total_slots];
    int *startPartial_rowId = new int[total_slots];
    int *endPartial_rowId = new int[total_slots];
    int *startComplete_rowId = new int[total_slots];
    int *endComplete_rowId = new int[total_slots];

    int *NZ_INDICES = new int[NNZ];

    for (int i = 0; i < NNZ; i++)
    {
        NZ_INDICES[i] = i;
    }

    for (int i = 0; i < total_slots; i++)
    {
        startPartial_nz_start[i] = 0;
        endPartial_nz_start[i] = 0;
        startPartial_nnz[i] = 0;
        endPartial_nnz[i] = 0;
        startPartial_rowId[i] = 0;
        endPartial_rowId[i] = 0;
        startComplete_rowId[i] = 0;
        endComplete_rowId[i] = 0;
    }

    for (int mp_thread_id = 0; mp_thread_id < num_mp_threads; mp_thread_id++)
    {

        int num_merge_items = R_PTR_NUM + NNZ;
        int items_per_thread = (num_merge_items + num_mp_threads - 1) / num_mp_threads;

        int diagonal = min(items_per_thread * mp_thread_id, num_merge_items);
        int diagonal_end = min(diagonal + items_per_thread, num_merge_items);

        CoordinateT thread_coord = MergePathSearch(diagonal, row_ptr, NZ_INDICES, R_PTR_NUM, NNZ);
        CoordinateT thread_coord_end = MergePathSearch(diagonal_end, row_ptr, NZ_INDICES, R_PTR_NUM, NNZ);

        int start_row = thread_coord.x - 1;
        int end_row = thread_coord_end.x - 1;
        if (start_row < 0)
            start_row = 0;

        int start_partial_nnz = 0;

        int nz_start_index = thread_coord.y;
        if (row_ptr[start_row] == nz_start_index)
        { // start row is a complete row
            nz_start_index = 0;
        }
        if (mp_thread_id == 0)
        {
            nz_start_index = 0;
        }

        int nz_end_index = thread_coord_end.y;
        if (row_ptr[end_row] == nz_end_index)
        { // end row is a complete row
            nz_end_index = 0;
        }
        if (nz_start_index != 0)
        {
            if (start_row == end_row && nz_end_index != 0)
            { // there is a single partial row
                start_partial_nnz = nz_end_index - nz_start_index;
                nz_end_index = 0;
            }
            else
            {
                start_partial_nnz = row_ptr[start_row + 1] - nz_start_index;
            }
        }

        int end_row_nnz = 0;
        if (nz_end_index != 0)
        {
            end_row_nnz = nz_end_index - row_ptr[end_row];
            nz_end_index = row_ptr[end_row];
        }

        int slot_counter = 0;
        int max_nnz_per_slot, est_num_slots;

        int max_nnz_threshold = 100000000; // set to high number to never break partial rows

        if (using_slot == 0)
        {

            startPartial_nz_start[mp_thread_id * num_slots] = nz_start_index;
            startPartial_nnz[mp_thread_id * num_slots] = start_partial_nnz;

            endPartial_nz_start[mp_thread_id * num_slots] = nz_end_index;
            endPartial_nnz[mp_thread_id * num_slots] = end_row_nnz;

            startPartial_rowId[mp_thread_id * num_slots] = start_row;
            endPartial_rowId[mp_thread_id * num_slots] = end_row;

            startComplete_rowId[mp_thread_id * num_slots] = start_row;
            endComplete_rowId[mp_thread_id * num_slots] = end_row;
        }

        else
        {

            /* Start Partial Row */
            if (nz_start_index != 0)
            { // if we have a start partial row, we divide it among est_num_slots slots

                int startPartial_last_nz_index = nz_start_index + start_partial_nnz; // last nz of the start partial row

                est_num_slots = ceil((double)start_partial_nnz / max_nnz_threshold); // estimated number of slots for partial row

                if (est_num_slots > num_slots)
                {
                    est_num_slots = num_slots;
                    max_nnz_per_slot = ceil((double)start_partial_nnz / num_slots);
                }
                else
                {
                    max_nnz_per_slot = max_nnz_threshold;
                }

                for (int i = 0; i < est_num_slots; i++)
                { //  loop to distribute the non-zeros among slots

                    startPartial_nz_start[mp_thread_id * num_slots + slot_counter] = nz_start_index;
                    startPartial_rowId[mp_thread_id * num_slots + slot_counter] = start_row;

                    if (nz_start_index + max_nnz_per_slot > startPartial_last_nz_index)
                    { // if we exceed the nnz boundary, then the last slot gets remaining nzs
                        startPartial_nnz[mp_thread_id * num_slots + slot_counter] = startPartial_last_nz_index - nz_start_index;
                    }
                    else
                    {
                        startPartial_nnz[mp_thread_id * num_slots + slot_counter] = max_nnz_per_slot; // else, each slot gets the max_nnz_per_slot
                    }

                    slot_counter++;
                    nz_start_index += max_nnz_per_slot;

                    if (slot_counter >= num_slots)
                        slot_counter = 0;
                }

                start_row += 1;
            }

            if (slot_counter >= num_slots)
                slot_counter = 0;

            /* Complete Rows */
            for (int i = 0; i < num_slots; i++)
            {

                if (start_row + i == end_row)
                    break;

                startComplete_rowId[mp_thread_id * num_slots + slot_counter] = start_row + i;
                endComplete_rowId[mp_thread_id * num_slots + slot_counter] = end_row;

                slot_counter++;

                if (slot_counter >= num_slots)
                    slot_counter = 0;
            }

            if (slot_counter >= num_slots)
                slot_counter = 0;

            /* End Partial Row */
            if (nz_end_index != 0)
            {

                int endPartial_last_nz_index = nz_end_index + end_row_nnz;

                est_num_slots = ceil((double)end_row_nnz / max_nnz_threshold); // estimated number of slots for partial row

                if (est_num_slots > num_slots)
                {
                    est_num_slots = num_slots;
                    max_nnz_per_slot = ceil((double)end_row_nnz / num_slots);
                }
                else
                {
                    max_nnz_per_slot = max_nnz_threshold;
                }

                for (int i = 0; i < est_num_slots; i++)
                { // loop to distribute the non-zeros among slots
                    endPartial_nz_start[mp_thread_id * num_slots + slot_counter] = nz_end_index;
                    endPartial_rowId[mp_thread_id * num_slots + slot_counter] = end_row;

                    if (nz_end_index + max_nnz_per_slot > endPartial_last_nz_index)
                    { // if we exceed the nnz boundary, then the last slot gets remaining nzs
                        endPartial_nnz[mp_thread_id * num_slots + slot_counter] = endPartial_last_nz_index - nz_end_index;
                    }
                    else
                    {
                        endPartial_nnz[mp_thread_id * num_slots + slot_counter] = max_nnz_per_slot; // else, each slot gets the max_nnz_per_slot
                    }

                    nz_end_index += max_nnz_per_slot;

                    slot_counter++;
                    if (slot_counter >= num_slots)
                    {
                        slot_counter = 0;
                    }
                }
            }
        }
    }

    return {startPartial_nz_start, startPartial_nnz,
            endPartial_nz_start, endPartial_nnz,
            startPartial_rowId, endPartial_rowId,
            startComplete_rowId, endComplete_rowId};
}

torch::Tensor prune_spmm(torch::Tensor input,
                          torch::Tensor row_pointer,
                          torch::Tensor column_index,
                          torch::Tensor degrees,
                          int threads_per_warp, int gin, float epsilon)
{
    const int NODE_NUM = input.size(0);
    const int dim = input.size(1);
    const int nnz = column_index.size(0);
    const int R_PTR_NUM = NODE_NUM + 1;

    input = input.reshape({NODE_NUM * dim, 1});

    // Create output tensor
    torch::Tensor output = torch::zeros(NODE_NUM * dim, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    int cost = ceil(nnz / (WARP * threads_per_warp));

    if (cost < 10)
        cost = 10;
    if (cost > 500)
        cost = 500;

    // cout << "TpW: " << threads_per_warp << endl;
    // cout << "Cost: " << cost << endl;

    int num_mp_threads = (R_PTR_NUM + nnz - 1) / cost;

    // if num_mp_threads < 1024 then set it to 1024
    if (num_mp_threads < 1024)
        num_mp_threads = 1024;

    // cout << "Total number of rows: " << NODE_NUM << " and non-zeros: " << nnz << endl;

    int *d_startPartial_nz_start, *d_startPartial_nnz, *d_endPartial_nz_start, *d_endPartial_nnz;
    int *d_startPartial_rowId, *d_endPartial_rowId;
    int *d_startComplete_rowId, *d_endComplete_rowId;

    int slots_per_mp_thread, using_slot;

    if (dim < WARP_SIZE)
    {
        slots_per_mp_thread = floor((double)32 / (threads_per_warp * dim)); // user input
        using_slot = 1;
    }
    else
    {
        slots_per_mp_thread = 1;
        using_slot = 0;
    }

    int total_slots = num_mp_threads * slots_per_mp_thread;
    
    // Row pointer must be on CPU before generating MP schedules
    auto row_pointer_cpu = row_pointer.to(torch::kCPU);

    auto mp_sched = generate_mp_sched(num_mp_threads, slots_per_mp_thread, using_slot, nnz, R_PTR_NUM, (int *) row_pointer_cpu.data_ptr());

    // Transfer schedule to GPU
    cudaMalloc((void **)&d_startPartial_nz_start, total_slots * sizeof(int));
    cudaMemcpy(d_startPartial_nz_start, mp_sched[0], total_slots * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_startPartial_nnz, total_slots * sizeof(int));
    cudaMemcpy(d_startPartial_nnz, mp_sched[1], total_slots * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_endPartial_nz_start, total_slots * sizeof(int));
    cudaMemcpy(d_endPartial_nz_start, mp_sched[2], total_slots * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_endPartial_nnz, total_slots * sizeof(int));
    cudaMemcpy(d_endPartial_nnz, mp_sched[3], total_slots * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_startPartial_rowId, total_slots * sizeof(int));
    cudaMemcpy(d_startPartial_rowId, mp_sched[4], total_slots * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_endPartial_rowId, total_slots * sizeof(int));
    cudaMemcpy(d_endPartial_rowId, mp_sched[5], total_slots * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_startComplete_rowId, total_slots * sizeof(int));
    cudaMemcpy(d_startComplete_rowId, mp_sched[6], total_slots * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_endComplete_rowId, total_slots * sizeof(int));
    cudaMemcpy(d_endComplete_rowId, mp_sched[7], total_slots * sizeof(int), cudaMemcpyHostToDevice);

    // Get GPU data pointers
    auto d_row_pointer = row_pointer.data_ptr();
    auto d_column_index = column_index.data_ptr();
    auto d_degrees = degrees.data_ptr();

    auto d_output = output.data_ptr();
    auto d_input = input.data_ptr();

    int factor = 0;
    int grid = 0;
    int total_warps = 0;

    if (gin == 1){      
        
        // out of 32 how many threads we use
        int lanes_per_mp_thread = slots_per_mp_thread * dim;                   // always remain 32 with tpw=1
        int usable_lanes_per_warp = lanes_per_mp_thread * threads_per_warp; // 32 with tpw=1

        if (dim < WARP_SIZE) {

            grid = (num_mp_threads * WARP_SIZE + BLOCK - 1) / (BLOCK * threads_per_warp);
            total_warps = num_mp_threads / threads_per_warp; // total_warps needs to be multiplied with dim, tpw =1
            factor = 1;
        }
        else
        {
            factor = ceil((double)dim / WARP_SIZE);
            total_warps = num_mp_threads * factor;
            grid = (total_warps * WARP_SIZE + BLOCK - 1) / (BLOCK);
        }      
        
        spmm_forward_cuda_kernel_mp_gin<<<grid, BLOCK>>>(
            (float *)d_output, (float *)d_input,
            (int *)d_row_pointer, (int *)d_column_index,
            (int *)d_startPartial_nz_start,
            (int *)d_endPartial_nz_start,
            (int *)d_startPartial_nnz,
            (int *)d_endPartial_nnz,
            (int *)d_startPartial_rowId,
            (int *)d_endPartial_rowId,
            (int *)d_startComplete_rowId,
            (int *)d_endComplete_rowId,
            NODE_NUM, dim, total_warps,
            threads_per_warp, slots_per_mp_thread,
            lanes_per_mp_thread, usable_lanes_per_warp, factor, epsilon);
            
            cudaDeviceSynchronize();

    }
    
    else{

        // out of 32 how many threads we use
        int lanes_per_mp_thread = slots_per_mp_thread * dim;                   // always remain 32 with tpw=1
        int usable_lanes_per_warp = lanes_per_mp_thread * threads_per_warp; // 32 with tpw=1

        if (dim < WARP_SIZE) {

            grid = (num_mp_threads * WARP_SIZE + BLOCK - 1) / (BLOCK * threads_per_warp);
            total_warps = num_mp_threads / threads_per_warp; // total_warps needs to be multiplied with dim, tpw =1
            factor = 1;

        }
        else {
            factor = ceil((double)dim / WARP_SIZE);
            total_warps = num_mp_threads * factor;
            grid = (total_warps * WARP_SIZE + BLOCK - 1) / (BLOCK);
        }

        spmm_forward_cuda_kernel_mp<<<grid, BLOCK>>>(
                (float *)d_output, (float *)d_input,
                (int *)d_row_pointer, (int *)d_column_index, (float *) d_degrees, 
                (int *)d_startPartial_nz_start,
                (int *)d_endPartial_nz_start,
                (int *)d_startPartial_nnz,
                (int *)d_endPartial_nnz,
                (int *)d_startPartial_rowId,
                (int *)d_endPartial_rowId,
                (int *)d_startComplete_rowId,
                (int *)d_endComplete_rowId,
                NODE_NUM, dim, total_warps,
                threads_per_warp, slots_per_mp_thread,
                lanes_per_mp_thread, usable_lanes_per_warp, factor);
            cudaDeviceSynchronize();
          
    }
    return output.reshape({NODE_NUM, dim});
}