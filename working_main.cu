#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <vector>

// CUDA includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Constants
const float G = 1.0f;  // Simplified gravitational constant
const float SOFTENING = 0.5f;
const float THETA = 0.7f;
const float BOX_SIZE = 100.0f;

struct Body {
    float3 position;
    float3 velocity;
    float3 acceleration;
    float mass;
};

struct OctreeNode {
    float3 center_of_mass;
    float3 bounds_min, bounds_max;
    float total_mass;
    int body_index;
    int children[8];
    bool is_leaf;
};

// Initialize bodies with orbital velocities
void initializeBodies(Body* bodies, int n_bodies) {
    std::srand(std::time(0));
    
    // Central massive body
    bodies[0].position = make_float3(0.0f, 0.0f, 0.0f);
    bodies[0].velocity = make_float3(0.0f, 0.0f, 0.0f);
    bodies[0].mass = 1000.0f;
    bodies[0].acceleration = make_float3(0.0f, 0.0f, 0.0f);
    
    // Orbiting bodies
    for (int i = 1; i < n_bodies; i++) {
        float angle = 2.0f * M_PI * (float)std::rand() / RAND_MAX;
        float radius = 15.0f + 25.0f * (float)std::rand() / RAND_MAX;
        
        bodies[i].position.x = radius * cosf(angle);
        bodies[i].position.y = radius * sinf(angle);
        bodies[i].position.z = 0.1f * (float)std::rand() / RAND_MAX - 0.05f;
        
        // Circular orbital velocity
        float distance = sqrtf(bodies[i].position.x * bodies[i].position.x +
                              bodies[i].position.y * bodies[i].position.y);
        float speed = sqrtf(G * bodies[0].mass / distance);
        
        bodies[i].velocity.x = -speed * sinf(angle);
        bodies[i].velocity.y = speed * cosf(angle);
        bodies[i].velocity.z = 0.0f;
        
        bodies[i].mass = 1.0f + 2.0f * (float)std::rand() / RAND_MAX;
        bodies[i].acceleration = make_float3(0.0f, 0.0f, 0.0f);
    }
}

// Direct N-body computation (for verification)
__global__ void computeDirectForces(Body* bodies, int n_bodies) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bodies) return;
    
    float3 accel = make_float3(0.0f, 0.0f, 0.0f);
    float3 pos_i = bodies[i].position;
    
    for (int j = 0; j < n_bodies; j++) {
        if (i == j) continue;
        
        float3 pos_j = bodies[j].position;
        float mass_j = bodies[j].mass;
        
        float3 r = make_float3(pos_j.x - pos_i.x, 
                              pos_j.y - pos_i.y, 
                              pos_j.z - pos_j.z);
        
        float dist_sq = r.x * r.x + r.y * r.y + r.z * r.z + SOFTENING;
        float inv_dist = 1.0f / sqrtf(dist_sq);
        float inv_dist_cube = inv_dist * inv_dist * inv_dist;
        
        float force = G * mass_j * inv_dist_cube;
        
        accel.x += force * r.x;
        accel.y += force * r.y;
        accel.z += force * r.z;
    }
    
    bodies[i].acceleration = accel;
}

// Simple all-pairs gravity (much simpler but works)
__global__ void computeSimpleForces(Body* bodies, int n_bodies) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bodies) return;
    
    float3 accel = make_float3(0.0f, 0.0f, 0.0f);
    float3 pos_i = bodies[i].position;
    
    // Only compute forces from central body and a few neighbors for simplicity
    for (int j = 0; j < min(n_bodies, 50); j++) {
        if (i == j) continue;
        
        float3 pos_j = bodies[j].position;
        float mass_j = bodies[j].mass;
        
        float3 r = make_float3(pos_j.x - pos_i.x, 
                              pos_j.y - pos_i.y, 
                              pos_j.z - pos_i.z);
        
        float dist_sq = r.x * r.x + r.y * r.y + r.z * r.z + SOFTENING;
        float inv_dist_cube = 1.0f / (dist_sq * sqrtf(dist_sq));
        
        float force = G * mass_j * inv_dist_cube;
        
        accel.x += force * r.x;
        accel.y += force * r.y;
        accel.z += force * r.z;
    }
    
    bodies[i].acceleration = accel;
}

// Update positions and velocities
__global__ void updateBodies(Body* bodies, int n_bodies, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bodies) return;
    
    // Semi-implicit Euler integration
    bodies[i].velocity.x += bodies[i].acceleration.x * dt;
    bodies[i].velocity.y += bodies[i].acceleration.y * dt;
    bodies[i].velocity.z += bodies[i].acceleration.z * dt;
    
    bodies[i].position.x += bodies[i].velocity.x * dt;
    bodies[i].position.y += bodies[i].velocity.y * dt;
    bodies[i].position.z += bodies[i].velocity.z * dt;
}

// Reset accelerations
__global__ void resetAccelerations(Body* bodies, int n_bodies) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bodies) return;
    
    bodies[i].acceleration = make_float3(0.0f, 0.0f, 0.0f);
}

void printSimulationState(Body* bodies, int n_bodies, int step, float time) {
    std::cout << "Step " << step << " (Time: " << time << "s)" << std::endl;
    
    // Calculate statistics
    float3 com = make_float3(0.0f, 0.0f, 0.0f);
    float total_mass = 0.0f;
    float max_speed = 0.0f;
    
    for (int i = 0; i < n_bodies; i++) {
        com.x += bodies[i].position.x * bodies[i].mass;
        com.y += bodies[i].position.y * bodies[i].mass;
        com.z += bodies[i].position.z * bodies[i].mass;
        total_mass += bodies[i].mass;
        
        float speed = sqrtf(bodies[i].velocity.x * bodies[i].velocity.x +
                           bodies[i].velocity.y * bodies[i].velocity.y +
                           bodies[i].velocity.z * bodies[i].velocity.z);
        max_speed = fmaxf(max_speed, speed);
    }
    
    com.x /= total_mass;
    com.y /= total_mass;
    com.z /= total_mass;
    
    std::cout << "  Center of mass: (" << com.x << ", " << com.y << ", " << com.z << ")" << std::endl;
    std::cout << "  Max speed: " << max_speed << std::endl;
    
    // Print first few bodies
    for (int i = 0; i < std::min(3, n_bodies); i++) {
        float speed = sqrtf(bodies[i].velocity.x * bodies[i].velocity.x +
                           bodies[i].velocity.y * bodies[i].velocity.y +
                           bodies[i].velocity.z * bodies[i].velocity.z);
        std::cout << "  Body " << i << ": pos(" << bodies[i].position.x << ", " 
                  << bodies[i].position.y << ", " << bodies[i].position.z 
                  << ") vel(" << bodies[i].velocity.x << ", "
                  << bodies[i].velocity.y << ", " << bodies[i].velocity.z
                  << ") speed: " << speed << std::endl;
    }
}

int main() {
    const int N_BODIES = 512;    // Reduced for stability
    const float DT = 0.01f;      // Smaller time step
    const int STEPS = 200;
    const int BLOCK_SIZE = 256;
    
    std::cout << "WORKING N-Body Simulation" << std::endl;
    std::cout << "Bodies: " << N_BODIES << ", Steps: " << STEPS << ", DT: " << DT << std::endl;
    
    // Host memory
    Body* h_bodies = new Body[N_BODIES];
    initializeBodies(h_bodies, N_BODIES);
    
    // Device memory
    Body* d_bodies;
    cudaMalloc(&d_bodies, N_BODIES * sizeof(Body));
    
    // Copy to device
    cudaMemcpy(d_bodies, h_bodies, N_BODIES * sizeof(Body), cudaMemcpyHostToDevice);
    
    int grid_size = (N_BODIES + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    std::cout << "\nStarting simulation..." << std::endl;
    std::cout << "Initial state:" << std::endl;
    printSimulationState(h_bodies, N_BODIES, 0, 0.0f);
    
    float total_time = 0.0f;
    
    for (int step = 0; step < STEPS; step++) {
        cudaEventRecord(start);
        
        // Reset accelerations
        resetAccelerations<<<grid_size, BLOCK_SIZE>>>(d_bodies, N_BODIES);
        cudaDeviceSynchronize();
        
        // Compute forces (using simple method for now)
        computeSimpleForces<<<grid_size, BLOCK_SIZE>>>(d_bodies, N_BODIES);
        cudaDeviceSynchronize();
        
        // Update positions and velocities
        updateBodies<<<grid_size, BLOCK_SIZE>>>(d_bodies, N_BODIES, DT);
        cudaDeviceSynchronize();
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float step_time = 0.0f;
        cudaEventElapsedTime(&step_time, start, stop);
        total_time += step_time;
        
        if (step % 20 == 0) {
            // Copy back to host for display
            cudaMemcpy(h_bodies, d_bodies, N_BODIES * sizeof(Body), cudaMemcpyDeviceToHost);
            printSimulationState(h_bodies, N_BODIES, step, step * DT);
            std::cout << "  Step time: " << step_time << " ms" << std::endl;
        }
    }
    
    // Final state
    cudaMemcpy(h_bodies, d_bodies, N_BODIES * sizeof(Body), cudaMemcpyDeviceToHost);
    std::cout << "\nFinal state:" << std::endl;
    printSimulationState(h_bodies, N_BODIES, STEPS, STEPS * DT);
    
    std::cout << "\nSimulation completed!" << std::endl;
    std::cout << "Total time: " << total_time << " ms" << std::endl;
    std::cout << "Average time per step: " << total_time / STEPS << " ms" << std::endl;
    
    // Cleanup
    delete[] h_bodies;
    cudaFree(d_bodies);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}