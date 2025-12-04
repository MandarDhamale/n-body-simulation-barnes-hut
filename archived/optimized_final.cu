#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>

// CUDA includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>

using namespace cooperative_groups;

// Constants in constant memory (fastest access)
__constant__ float const_G = 1.0f;
__constant__ float const_softening = 0.5f;
__constant__ float const_theta = 0.7f;
__constant__ float const_dt = 0.01f;
__constant__ float const_max_velocity = 20.0f;

// Optimized body structure (packed for memory efficiency)
struct __align__(16) Body {
    float4 position;    // x, y, z, mass
    float4 velocity;    // vx, vy, vz, unused
    float4 acceleration; // ax, ay, az, unused
};

// Optimized octree node (packed)
struct __align__(16) OctreeNode {
    float4 center_of_mass; // x, y, z, total_mass
    float4 bounds_min;     // min_x, min_y, min_z, size
    float4 bounds_max;     // max_x, max_y, max_z, unused
    int body_index;
    int children[8];
    bool is_leaf;
};

// Fast reciprocal square root (CUDA intrinsic)
__device__ __forceinline__ float fast_inv_sqrt(float x) {
    return rsqrtf(x);
}

// Fast inverse cube using intrinsics
__device__ __forceinline__ float fast_inv_cube(float dist_sq) {
    float inv_dist = rsqrtf(dist_sq);
    return inv_dist * inv_dist * inv_dist;
}

// Optimized direct N-body with shared memory tiling and divergence elimination
__global__ void optimizedDirectForces(Body* bodies, int n_bodies) {
    extern __shared__ float4 shared_data[]; // Position + mass
    float4* shared_pos_mass = shared_data;
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float3 accel = make_float3(0.0f, 0.0f, 0.0f);
    
    if (i < n_bodies) {
        float4 body_i = bodies[i].position;
        float3 pos_i = make_float3(body_i.x, body_i.y, body_i.z);
        
        // Process in tiles for memory coalescing
        for (int tile = 0; tile < gridDim.x; tile++) {
            int j = tile * blockDim.x + threadIdx.x;
            
            // Load tile into shared memory
            if (j < n_bodies) {
                shared_pos_mass[threadIdx.x] = bodies[j].position;
            }
            __syncthreads();
            
            // Process tile with minimal divergence
            #pragma unroll 8
            for (int k = 0; k < min(blockDim.x, n_bodies - tile * blockDim.x); k++) {
                float4 body_j = shared_pos_mass[k];
                int global_j = tile * blockDim.x + k;
                
                // No conditional branch - use mask instead
                float mask = (i != global_j) ? 1.0f : 0.0f;
                
                float3 r = make_float3(
                    body_j.x - pos_i.x,
                    body_j.y - pos_i.y,
                    body_j.z - pos_i.z
                );
                
                float dist_sq = r.x * r.x + r.y * r.y + r.z * r.z + const_softening;
                float inv_dist_cube = fast_inv_cube(dist_sq);
                float force = const_G * body_j.w * inv_dist_cube * mask;
                
                // FMA (Fused Multiply-Add) for better performance
                accel.x = __fmaf_rn(force, r.x, accel.x);
                accel.y = __fmaf_rn(force, r.y, accel.y);
                accel.z = __fmaf_rn(force, r.z, accel.z);
            }
            __syncthreads();
        }
        
        // Store acceleration
        bodies[i].acceleration.x = accel.x;
        bodies[i].acceleration.y = accel.y;
        bodies[i].acceleration.z = accel.z;
    }
}

// Warp-optimized direct method (even faster for some GPUs)
__global__ void warpOptimizedForces(Body* bodies, int n_bodies) {
    // Use warp-level programming
    thread_block_tile<32> tile = tiled_partition<32>(this_thread_block());
    int lane_id = threadIdx.x % 32;  // REMOVED: int warp_id = threadIdx.x / 32;
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= n_bodies) return;
    
    float3 accel = make_float3(0.0f, 0.0f, 0.0f);
    float4 body_i = bodies[i].position;
    float3 pos_i = make_float3(body_i.x, body_i.y, body_i.z);
    
    // Warp processes 32 bodies at a time
    for (int base_j = 0; base_j < n_bodies; base_j += 32) {
        int j = base_j + lane_id;
        float4 body_j;
        
        if (j < n_bodies) {
            body_j = bodies[j].position;
        }
        
        // Broadcast within warp (simulates shared memory but warp-wide)
        body_j = tile.shfl(body_j, lane_id);
        
        // All threads in warp process the same body_j
        float mask = (i != (base_j + lane_id)) ? 1.0f : 0.0f;
        
        float3 r = make_float3(
            body_j.x - pos_i.x,
            body_j.y - pos_i.y,
            body_j.z - pos_i.z
        );
        
        float dist_sq = r.x * r.x + r.y * r.y + r.z * r.z + const_softening;
        float inv_dist_cube = fast_inv_cube(dist_sq);
        float force = const_G * body_j.w * inv_dist_cube * mask;
        
        accel.x = __fmaf_rn(force, r.x, accel.x);
        accel.y = __fmaf_rn(force, r.y, accel.y);
        accel.z = __fmaf_rn(force, r.z, accel.z);
        
        tile.sync();
    }
    
    // Reduce within warp for better memory access pattern
    accel.x = tile.shfl_down(accel.x, 16);
    accel.x = tile.shfl_down(accel.x, 8);
    accel.x = tile.shfl_down(accel.x, 4);
    accel.x = tile.shfl_down(accel.x, 2);
    accel.x = tile.shfl_down(accel.x, 1);
    
    accel.y = tile.shfl_down(accel.y, 16);
    accel.y = tile.shfl_down(accel.y, 8);
    accel.y = tile.shfl_down(accel.y, 4);
    accel.y = tile.shfl_down(accel.y, 2);
    accel.y = tile.shfl_down(accel.y, 1);
    
    accel.z = tile.shfl_down(accel.z, 16);
    accel.z = tile.shfl_down(accel.z, 8);
    accel.z = tile.shfl_down(accel.z, 4);
    accel.z = tile.shfl_down(accel.z, 2);
    accel.z = tile.shfl_down(accel.z, 1);
    
    if (lane_id == 0) {
        bodies[i].acceleration.x = accel.x;
        bodies[i].acceleration.y = accel.y;
        bodies[i].acceleration.z = accel.z;
    }
}

// Optimized position update with velocity clamping
__global__ void updateBodiesOptimized(Body* bodies, int n_bodies) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bodies) return;
    
    // Velocity Verlet integration - part 1
    bodies[i].velocity.x = __fmaf_rn(bodies[i].acceleration.x, const_dt * 0.5f, bodies[i].velocity.x);
    bodies[i].velocity.y = __fmaf_rn(bodies[i].acceleration.y, const_dt * 0.5f, bodies[i].velocity.y);
    bodies[i].velocity.z = __fmaf_rn(bodies[i].acceleration.z, const_dt * 0.5f, bodies[i].velocity.z);
    
    // Velocity clamping with fast approximate length
    float speed_sq = bodies[i].velocity.x * bodies[i].velocity.x +
                    bodies[i].velocity.y * bodies[i].velocity.y +
                    bodies[i].velocity.z * bodies[i].velocity.z;
    
    if (speed_sq > const_max_velocity * const_max_velocity) {
        float scale = const_max_velocity * fast_inv_sqrt(speed_sq);
        bodies[i].velocity.x *= scale;
        bodies[i].velocity.y *= scale;
        bodies[i].velocity.z *= scale;
    }
    
    // Update position
    bodies[i].position.x = __fmaf_rn(bodies[i].velocity.x, const_dt, bodies[i].position.x);
    bodies[i].position.y = __fmaf_rn(bodies[i].velocity.y, const_dt, bodies[i].position.y);
    bodies[i].position.z = __fmaf_rn(bodies[i].velocity.z, const_dt, bodies[i].position.z);
}

// Complete Verlet update (part 2)
__global__ void completeVerletUpdateOptimized(Body* bodies, int n_bodies) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bodies) return;
    
    bodies[i].velocity.x = __fmaf_rn(bodies[i].acceleration.x, const_dt * 0.5f, bodies[i].velocity.x);
    bodies[i].velocity.y = __fmaf_rn(bodies[i].acceleration.y, const_dt * 0.5f, bodies[i].velocity.y);
    bodies[i].velocity.z = __fmaf_rn(bodies[i].acceleration.z, const_dt * 0.5f, bodies[i].velocity.z);
}

// Reset accelerations (memset alternative)
__global__ void resetAccelerationsOptimized(Body* bodies, int n_bodies) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bodies) return;
    
    bodies[i].acceleration.x = 0.0f;
    bodies[i].acceleration.y = 0.0f;
    bodies[i].acceleration.z = 0.0f;
}

// Optimized Barnes-Hut with stack in shared memory
__global__ void optimizedBarnesHutForces(Body* bodies, OctreeNode* tree, int n_bodies, int max_nodes) {
    extern __shared__ int shared_stack[];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bodies) return;
    
    float3 accel = make_float3(0.0f, 0.0f, 0.0f);
    float4 body_i = bodies[i].position;
    float3 pos_i = make_float3(body_i.x, body_i.y, body_i.z);
    
    // Private stack in registers for speed
    int local_stack[32];
    int local_stack_ptr = 0;
    local_stack[local_stack_ptr++] = 0; // Root
    
    while (local_stack_ptr > 0) {
        int node_idx = local_stack[--local_stack_ptr];
        OctreeNode node = tree[node_idx];
        
        if (node.center_of_mass.w == 0.0f) continue; // Empty node
        
        // Pre-calculate values before branch
        float3 r = make_float3(
            node.center_of_mass.x - pos_i.x,
            node.center_of_mass.y - pos_i.y,
            node.center_of_mass.z - pos_i.z
        );
        
        float dist_sq = r.x * r.x + r.y * r.y + r.z * r.z + const_softening;
        float dist = sqrtf(dist_sq);
        float ratio = node.bounds_min.w / dist; // size stored in bounds_min.w
        
        // Branch prediction friendly
        bool use_node = node.is_leaf || (ratio < const_theta);
        
        if (use_node) {
            float mask = (node.body_index != i) ? 1.0f : 0.0f;
            float inv_dist_cube = fast_inv_cube(dist_sq);
            float force = const_G * node.center_of_mass.w * inv_dist_cube * mask;
            
            accel.x = __fmaf_rn(force, r.x, accel.x);
            accel.y = __fmaf_rn(force, r.y, accel.y);
            accel.z = __fmaf_rn(force, r.z, accel.z);
        } else {
            // Push children - unrolled for better performance
            #pragma unroll
            for (int child = 0; child < 8; child++) {
                if (node.children[child] != -1) {
                    local_stack[local_stack_ptr++] = node.children[child];
                }
            }
        }
    }
    
    // Store result
    bodies[i].acceleration.x = accel.x;
    bodies[i].acceleration.y = accel.y;
    bodies[i].acceleration.z = accel.z;
}

// Performance monitoring structure
struct PerformanceMetrics {
    float initialization_time;
    float average_step_time;
    float min_step_time;
    float max_step_time;
    float total_time;
    int steps_completed;
};

// Initialize bodies with optimized random number generation
void initializeBodiesOptimized(Body* bodies, int n_bodies) {
    // Use memory-aligned initialization
    #pragma omp parallel for
    for (int i = 0; i < n_bodies; i++) {
        // Fast pseudo-random using thread-local state
        unsigned int seed = i * 123456789;
        
        // Central massive body
        if (i == 0) {
            bodies[i].position = make_float4(0.0f, 0.0f, 0.0f, 5000.0f);
            bodies[i].velocity = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            bodies[i].acceleration = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        } else {
            // Fast random numbers
            seed = (seed * 1103515245 + 12345) & 0x7fffffff;
            float r1 = (float)seed / 2147483647.0f;
            seed = (seed * 1103515245 + 12345) & 0x7fffffff;
            float r2 = (float)seed / 2147483647.0f;
            seed = (seed * 1103515245 + 12345) & 0x7fffffff;
            float r3 = (float)seed / 2147483647.0f;
            
            float angle = 2.0f * M_PI * r1;
            float radius = 10.0f + 35.0f * powf(r2, 0.7f);
            
            // Spiral structure
            angle += 0.3f * logf(radius / 10.0f);
            
            bodies[i].position.x = radius * cosf(angle);
            bodies[i].position.y = radius * sinf(angle);
            bodies[i].position.z = 0.2f * (r3 - 0.5f);
            bodies[i].position.w = 0.5f + 4.5f * r2 * r2; // Mass
            
            // Orbital velocity
            float distance = sqrtf(bodies[i].position.x * bodies[i].position.x +
                                  bodies[i].position.y * bodies[i].position.y);
            float speed = sqrtf(1.0f * 5000.0f / distance) * (1.0f + 0.2f * (r3 - 0.5f));
            
            bodies[i].velocity.x = -speed * sinf(angle) + 0.05f * (r1 - 0.5f) * cosf(angle);
            bodies[i].velocity.y = speed * cosf(angle) + 0.05f * (r1 - 0.5f) * sinf(angle);
            bodies[i].velocity.z = 0.01f * (r2 - 0.5f);
            bodies[i].acceleration = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
}

// Calculate optimal block size dynamically
int calculateOptimalBlockSize(int n_bodies, void* kernel) {
    int minGridSize, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, n_bodies);
    return blockSize;
}

// Main simulation with performance optimization
int main() {
    const int N_BODIES = 2048;          // Larger simulation
    const int STEPS = 500;
    const bool USE_BARNES_HUT = true;  // Set to true for Barnes-Hut
    const bool USE_WARP_OPTIMIZED = true; // Even faster for compatible GPUs
    const bool SAVE_DATA = true;
    
    std::cout << "FULLY OPTIMIZED N-Body Simulation\n";
    std::cout << "===================================\n";
    std::cout << "Bodies: " << N_BODIES << "\n";
    std::cout << "Steps: " << STEPS << "\n";
    std::cout << "Method: " << (USE_BARNES_HUT ? "Barnes-Hut O(N log N)" : 
                               (USE_WARP_OPTIMIZED ? "Warp-Optimized Direct" : "Shared Memory Direct")) << "\n";
    
    // Host memory with alignment
    Body* h_bodies;
    cudaMallocHost(&h_bodies, N_BODIES * sizeof(Body)); // Pinned memory for faster transfers
    
    // Initialize on host
    auto start_init = std::chrono::high_resolution_clock::now();
    initializeBodiesOptimized(h_bodies, N_BODIES);
    auto end_init = std::chrono::high_resolution_clock::now();
    float init_time = std::chrono::duration<float, std::milli>(end_init - start_init).count();
    
    // Device memory
    Body* d_bodies;
    cudaMalloc(&d_bodies, N_BODIES * sizeof(Body));
    
    // Async memory copy with stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(d_bodies, h_bodies, N_BODIES * sizeof(Body), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    
    // Performance metrics
    PerformanceMetrics metrics = {0};
    metrics.initialization_time = init_time;
    metrics.min_step_time = 1000000.0f;
    
    // Calculate optimal configuration
    void* selected_kernel = USE_WARP_OPTIMIZED ? 
        (void*)warpOptimizedForces : (void*)optimizedDirectForces;
    
    int block_size = calculateOptimalBlockSize(N_BODIES, selected_kernel);
    int grid_size = (N_BODIES + block_size - 1) / block_size;
    
    std::cout << "\nOptimization Parameters:\n";
    std::cout << "  Block size: " << block_size << "\n";
    std::cout << "  Grid size: " << grid_size << "\n";
    std::cout << "  Shared memory per block: " << block_size * sizeof(float4) / 1024 << " KB\n";
    
    // CUDA events for precise timing
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    
    std::cout << "\nStarting optimized simulation...\n";
    
    // Warm-up run
    cudaEventRecord(start_event, stream);
    resetAccelerationsOptimized<<<grid_size, block_size, 0, stream>>>(d_bodies, N_BODIES);
    
    if (USE_WARP_OPTIMIZED) {
        warpOptimizedForces<<<grid_size, block_size, 0, stream>>>(d_bodies, N_BODIES);
    } else {
        size_t shared_mem = block_size * sizeof(float4);
        optimizedDirectForces<<<grid_size, block_size, shared_mem, stream>>>(d_bodies, N_BODIES);
    }
    
    updateBodiesOptimized<<<grid_size, block_size, 0, stream>>>(d_bodies, N_BODIES);
    cudaEventRecord(stop_event, stream);
    cudaEventSynchronize(stop_event);
    
    // Main simulation loop
    for (int step = 0; step < STEPS; step++) {
        cudaEventRecord(start_event, stream);
        
        // Reset accelerations
        resetAccelerationsOptimized<<<grid_size, block_size, 0, stream>>>(d_bodies, N_BODIES);
        
        // Compute forces
        if (USE_WARP_OPTIMIZED) {
            warpOptimizedForces<<<grid_size, block_size, 0, stream>>>(d_bodies, N_BODIES);
        } else {
            size_t shared_mem = block_size * sizeof(float4);
            optimizedDirectForces<<<grid_size, block_size, shared_mem, stream>>>(d_bodies, N_BODIES);
        }
        
        // Update positions
        updateBodiesOptimized<<<grid_size, block_size, 0, stream>>>(d_bodies, N_BODIES);
        
        cudaEventRecord(stop_event, stream);
        cudaEventSynchronize(stop_event);
        
        float step_time = 0.0f;
        cudaEventElapsedTime(&step_time, start_event, stop_event);
        
        metrics.total_time += step_time;
        metrics.min_step_time = fminf(metrics.min_step_time, step_time);
        metrics.max_step_time = fmaxf(metrics.max_step_time, step_time);
        metrics.steps_completed++;
        
        // Progress reporting
        if (step % 100 == 0) {
            std::cout << "Step " << step << " completed in " << step_time << " ms\n";
            
            // Optional: Copy data for analysis (but slows down simulation)
            if (SAVE_DATA && step % 200 == 0) {
                cudaMemcpyAsync(h_bodies, d_bodies, N_BODIES * sizeof(Body), 
                              cudaMemcpyDeviceToHost, stream);
                // Could save to file here
            }
        }
    }
    
    // Complete velocity update
    completeVerletUpdateOptimized<<<grid_size, block_size, 0, stream>>>(d_bodies, N_BODIES);
    
    // Copy final results back
    cudaMemcpyAsync(h_bodies, d_bodies, N_BODIES * sizeof(Body), 
                   cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // Calculate final metrics
    metrics.average_step_time = metrics.total_time / metrics.steps_completed;
    
    // Performance report
    std::cout << "\n=== PERFORMANCE REPORT ===\n";
    std::cout << "Initialization time: " << metrics.initialization_time << " ms\n";
    std::cout << "Total simulation time: " << metrics.total_time << " ms\n";
    std::cout << "Average step time: " << metrics.average_step_time << " ms\n";
    std::cout << "Min step time: " << metrics.min_step_time << " ms\n";
    std::cout << "Max step time: " << metrics.max_step_time << " ms\n";
    std::cout << "Steps per second: " << 1000.0f / metrics.average_step_time << "\n";
    std::cout << "Body updates per second: " 
              << N_BODIES * (1000.0f / metrics.average_step_time) << "\n";
    
    // Cleanup
    cudaStreamDestroy(stream);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaFreeHost(h_bodies);
    cudaFree(d_bodies);
    
    return 0;
}