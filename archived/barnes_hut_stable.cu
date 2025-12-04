#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <vector>

// CUDA includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Constants
const float G = 1.0f;
const float SOFTENING = 0.5f;
const float THETA = 0.7f;
const float DT = 0.01f;
const float BOX_SIZE = 100.0f;

// Body structure
struct Body {
    float3 position;
    float3 velocity;
    float3 acceleration;
    float mass;
};

// OctreeNode structure
struct OctreeNode {
    float3 center_of_mass;
    float3 bounds_min;
    float3 bounds_max;
    float total_mass;
    int body_index;
    int children[8];
    bool is_leaf;
    float size;
};

// Kernel: Reset accelerations
__global__ void resetAccelerations(Body* bodies, int n_bodies) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_bodies) {
        bodies[idx].acceleration.x = 0.0f;
        bodies[idx].acceleration.y = 0.0f;
        bodies[idx].acceleration.z = 0.0f;
    }
}

// Kernel: Barnes-Hut force calculation
__global__ void computeBarnesHutForces(Body* bodies, OctreeNode* tree, int n_bodies, int tree_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bodies || i < 0) return;
    
    float3 accel = make_float3(0.0f, 0.0f, 0.0f);
    float3 pos_i = bodies[i].position;
    
    // Stack for iterative tree traversal
    int stack[128];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0; // Root node
    
    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        
        // Safety check
        if (node_idx < 0 || node_idx >= tree_size) continue;
        
        OctreeNode node = tree[node_idx];
        
        if (node.total_mass <= 0.0f) continue;
        
        float3 r = make_float3(
            node.center_of_mass.x - pos_i.x,
            node.center_of_mass.y - pos_i.y,
            node.center_of_mass.z - pos_i.z
        );
        
        float dist_sq = r.x*r.x + r.y*r.y + r.z*r.z + SOFTENING;
        float dist = sqrtf(dist_sq);
        
        // Barnes-Hut criterion
        if (node.is_leaf || (node.size / dist < THETA)) {
            if (node.body_index != i) {
                float inv_dist_cube = 1.0f / (dist_sq * dist);
                float force = G * node.total_mass * inv_dist_cube;
                
                accel.x += force * r.x;
                accel.y += force * r.y;
                accel.z += force * r.z;
            }
        } else {
            for (int child = 0; child < 8; child++) {
                int child_idx = node.children[child];
                if (child_idx != -1 && child_idx < tree_size && stack_ptr < 128) {
                    stack[stack_ptr++] = child_idx;
                }
            }
        }
    }
    
    bodies[i].acceleration = accel;
}

// Kernel: Update positions
__global__ void updateBodies(Body* bodies, int n_bodies, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bodies || i < 0) return;
    
    bodies[i].velocity.x += bodies[i].acceleration.x * dt;
    bodies[i].velocity.y += bodies[i].acceleration.y * dt;
    bodies[i].velocity.z += bodies[i].acceleration.z * dt;
    
    bodies[i].position.x += bodies[i].velocity.x * dt;
    bodies[i].position.y += bodies[i].velocity.y * dt;
    bodies[i].position.z += bodies[i].velocity.z * dt;
}

// Helper: Get octant
int getOctant(const float3& point, const float3& center) {
    int octant = 0;
    if (point.x >= center.x) octant |= 1;
    if (point.y >= center.y) octant |= 2;
    if (point.z >= center.z) octant |= 4;
    return octant;
}

// Build octree safely
int buildOctree(Body* bodies, OctreeNode* tree, int n_bodies, int max_nodes) {
    // Initialize root
    tree[0].bounds_min = make_float3(-BOX_SIZE, -BOX_SIZE, -BOX_SIZE);
    tree[0].bounds_max = make_float3(BOX_SIZE, BOX_SIZE, BOX_SIZE);
    tree[0].is_leaf = true;
    tree[0].body_index = -1;
    tree[0].total_mass = 0.0f;
    tree[0].size = 2.0f * BOX_SIZE;
    for (int i = 0; i < 8; i++) tree[0].children[i] = -1;
    
    int node_count = 1;
    
    for (int body_idx = 0; body_idx < n_bodies; body_idx++) {
        Body body = bodies[body_idx];
        int current_node = 0;
        
        while (true) {
            if (current_node >= node_count || current_node >= max_nodes) break;
            
            OctreeNode* node = &tree[current_node];
            
            if (node->is_leaf) {
                if (node->body_index == -1) {
                    // Empty leaf
                    node->body_index = body_idx;
                    node->center_of_mass = body.position;
                    node->total_mass = body.mass;
                    break;
                } else {
                    // Split node
                    int old_body_idx = node->body_index;
                    Body old_body = bodies[old_body_idx];
                    
                    node->is_leaf = false;
                    
                    float3 center = make_float3(
                        (node->bounds_min.x + node->bounds_max.x) * 0.5f,
                        (node->bounds_min.y + node->bounds_max.y) * 0.5f,
                        (node->bounds_min.z + node->bounds_max.z) * 0.5f
                    );
                    
                    for (int octant = 0; octant < 8; octant++) {
                        if (node_count >= max_nodes) break;
                        
                        int child_idx = node_count;
                        node->children[octant] = child_idx;
                        
                        OctreeNode* child = &tree[child_idx];
                        child->is_leaf = true;
                        child->body_index = -1;
                        child->total_mass = 0.0f;
                        
                        // Set bounds
                        child->bounds_min.x = (octant & 1) ? center.x : node->bounds_min.x;
                        child->bounds_max.x = (octant & 1) ? node->bounds_max.x : center.x;
                        child->bounds_min.y = (octant & 2) ? center.y : node->bounds_min.y;
                        child->bounds_max.y = (octant & 2) ? node->bounds_max.y : center.y;
                        child->bounds_min.z = (octant & 4) ? center.z : node->bounds_min.z;
                        child->bounds_max.z = (octant & 4) ? node->bounds_max.z : center.z;
                        
                        child->size = (child->bounds_max.x - child->bounds_min.x) * 0.5f;
                        for (int i = 0; i < 8; i++) child->children[i] = -1;
                        
                        node_count++;
                    }
                    
                    // Re-insert old body
                    int old_octant = getOctant(old_body.position, center);
                    int old_child = node->children[old_octant];
                    if (old_child != -1 && old_child < max_nodes) {
                        tree[old_child].body_index = old_body_idx;
                        tree[old_child].center_of_mass = old_body.position;
                        tree[old_child].total_mass = old_body.mass;
                    }
                    
                    // Continue with current body
                    int new_octant = getOctant(body.position, center);
                    current_node = node->children[new_octant];
                    if (current_node == -1) break;
                }
            } else {
                float3 center = make_float3(
                    (node->bounds_min.x + node->bounds_max.x) * 0.5f,
                    (node->bounds_min.y + node->bounds_max.y) * 0.5f,
                    (node->bounds_min.z + node->bounds_max.z) * 0.5f
                );
                
                int octant = getOctant(body.position, center);
                int child_idx = node->children[octant];
                
                if (child_idx == -1) {
                    // Create new child
                    if (node_count >= max_nodes) break;
                    
                    int new_child = node_count++;
                    node->children[octant] = new_child;
                    
                    OctreeNode* child = &tree[new_child];
                    child->is_leaf = true;
                    child->body_index = body_idx;
                    child->center_of_mass = body.position;
                    child->total_mass = body.mass;
                    
                    child->bounds_min.x = (octant & 1) ? center.x : node->bounds_min.x;
                    child->bounds_max.x = (octant & 1) ? node->bounds_max.x : center.x;
                    child->bounds_min.y = (octant & 2) ? center.y : node->bounds_min.y;
                    child->bounds_max.y = (octant & 2) ? node->bounds_max.y : center.y;
                    child->bounds_min.z = (octant & 4) ? center.z : node->bounds_min.z;
                    child->bounds_max.z = (octant & 4) ? node->bounds_max.z : center.z;
                    
                    child->size = (child->bounds_max.x - child->bounds_min.x) * 0.5f;
                    for (int i = 0; i < 8; i++) child->children[i] = -1;
                    break;
                } else {
                    current_node = child_idx;
                }
            }
        }
    }
    
    return node_count;
}

// Compute mass distribution
void computeMassDistribution(OctreeNode* tree, int node_idx, int max_nodes) {
    if (node_idx >= max_nodes) return;
    
    OctreeNode* node = &tree[node_idx];
    
    if (!node->is_leaf) {
        float3 com = make_float3(0.0f, 0.0f, 0.0f);
        float total_mass = 0.0f;
        int child_count = 0;
        
        for (int i = 0; i < 8; i++) {
            int child_idx = node->children[i];
            if (child_idx != -1 && child_idx < max_nodes) {
                computeMassDistribution(tree, child_idx, max_nodes);
                OctreeNode* child = &tree[child_idx];
                if (child->total_mass > 0) {
                    com.x += child->center_of_mass.x * child->total_mass;
                    com.y += child->center_of_mass.y * child->total_mass;
                    com.z += child->center_of_mass.z * child->total_mass;
                    total_mass += child->total_mass;
                    child_count++;
                }
            }
        }
        
        if (total_mass > 0) {
            node->center_of_mass.x = com.x / total_mass;
            node->center_of_mass.y = com.y / total_mass;
            node->center_of_mass.z = com.z / total_mass;
            node->total_mass = total_mass;
        }
    }
}

// Initialize bodies safely
void initializeBodies(Body* bodies, int n_bodies) {
    srand(time(NULL));
    
    bodies[0].position = make_float3(0.0f, 0.0f, 0.0f);
    bodies[0].velocity = make_float3(0.0f, 0.0f, 0.0f);
    bodies[0].mass = 10000.0f;
    bodies[0].acceleration = make_float3(0.0f, 0.0f, 0.0f);
    
    for (int i = 1; i < n_bodies; i++) {
        float angle = 2.0f * M_PI * (rand() / (float)RAND_MAX);
        float radius = 10.0f + 35.0f * (rand() / (float)RAND_MAX);
        
        bodies[i].position.x = radius * cosf(angle);
        bodies[i].position.y = radius * sinf(angle);
        bodies[i].position.z = 0.1f * (rand() / (float)RAND_MAX - 0.5f);
        
        float distance = sqrtf(bodies[i].position.x * bodies[i].position.x +
                              bodies[i].position.y * bodies[i].position.y);
        float speed = sqrtf(G * bodies[0].mass / distance);
        
        bodies[i].velocity.x = -speed * sinf(angle);
        bodies[i].velocity.y = speed * cosf(angle);
        bodies[i].velocity.z = 0.0f;
        
        bodies[i].mass = 1.0f + 4.0f * (rand() / (float)RAND_MAX);
        bodies[i].acceleration = make_float3(0.0f, 0.0f, 0.0f);
    }
}

// Save data to file
void saveData(Body* bodies, int n_bodies, int step, const char* filename) {
    std::ofstream file(filename, std::ios::app);
    if (!file) return;
    
    file << "Step " << step << "\n";
    for (int i = 0; i < n_bodies; i++) {
        file << bodies[i].position.x << " "
             << bodies[i].position.y << " "
             << bodies[i].position.z << " "
             << bodies[i].mass << "\n";
    }
    file << "\n";
    file.close();
}

int main() {
    const int N_BODIES = 2048;      // Reduced for stability
    const int STEPS = 100;
    const int BLOCK_SIZE = 256;
    const int MAX_TREE_NODES = N_BODIES * 4;  // Increased margin
    
    std::cout << "==================================================\n";
    std::cout << "       STABLE BARNES-HUT N-BODY SIMULATION       \n";
    std::cout << "==================================================\n\n";
    
    std::cout << "Configuration:\n";
    std::cout << "  Bodies: " << N_BODIES << "\n";
    std::cout << "  Steps: " << STEPS << "\n";
    std::cout << "  Algorithm: Barnes-Hut O(N log N)\n";
    
    // Clear output file
    std::ofstream clearfile("stable_output.txt");
    clearfile.close();
    
    // Host allocations with bounds checking
    Body* h_bodies = nullptr;
    OctreeNode* h_tree = nullptr;
    
    try {
        h_bodies = new Body[N_BODIES];
        h_tree = new OctreeNode[MAX_TREE_NODES];
    } catch (std::bad_alloc& e) {
        std::cerr << "Memory allocation failed: " << e.what() << "\n";
        return 1;
    }
    
    initializeBodies(h_bodies, N_BODIES);
    saveData(h_bodies, N_BODIES, 0, "stable_output.txt");
    
    // Device memory
    Body* d_bodies = nullptr;
    OctreeNode* d_tree = nullptr;
    
    cudaError_t cuda_status;
    
    cuda_status = cudaMalloc(&d_bodies, N_BODIES * sizeof(Body));
    if (cuda_status != cudaSuccess) {
        std::cerr << "cudaMalloc failed for bodies: " << cudaGetErrorString(cuda_status) << "\n";
        delete[] h_bodies;
        delete[] h_tree;
        return 1;
    }
    
    cuda_status = cudaMalloc(&d_tree, MAX_TREE_NODES * sizeof(OctreeNode));
    if (cuda_status != cudaSuccess) {
        std::cerr << "cudaMalloc failed for tree: " << cudaGetErrorString(cuda_status) << "\n";
        cudaFree(d_bodies);
        delete[] h_bodies;
        delete[] h_tree;
        return 1;
    }
    
    cudaMemcpy(d_bodies, h_bodies, N_BODIES * sizeof(Body), cudaMemcpyHostToDevice);
    
    int grid_size = (N_BODIES + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    std::cout << "\nRunning simulation...\n";
    
    float total_time = 0.0f;
    
    for (int step = 0; step < STEPS; step++) {
        cudaEventRecord(start);
        
        resetAccelerations<<<grid_size, BLOCK_SIZE>>>(d_bodies, N_BODIES);
        cudaDeviceSynchronize();
        
        // Build tree on host
        int tree_size = buildOctree(h_bodies, h_tree, N_BODIES, MAX_TREE_NODES);
        computeMassDistribution(h_tree, 0, tree_size);
        
        cudaMemcpy(d_tree, h_tree, tree_size * sizeof(OctreeNode), cudaMemcpyHostToDevice);
        
        computeBarnesHutForces<<<grid_size, BLOCK_SIZE>>>(d_bodies, d_tree, N_BODIES, tree_size);
        cudaDeviceSynchronize();
        
        updateBodies<<<grid_size, BLOCK_SIZE>>>(d_bodies, N_BODIES, DT);
        cudaDeviceSynchronize();
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float step_time = 0.0f;
        cudaEventElapsedTime(&step_time, start, stop);
        total_time += step_time;
        
        // Copy back for saving
        cudaMemcpy(h_bodies, d_bodies, N_BODIES * sizeof(Body), cudaMemcpyDeviceToHost);
        
        if (step % 10 == 0) {
            saveData(h_bodies, N_BODIES, step, "stable_output.txt");
            std::cout << "  Step " << step << ": " << step_time << " ms, Tree nodes: " << tree_size << "\n";
        }
    }
    
    saveData(h_bodies, N_BODIES, STEPS, "stable_output.txt");
    
    std::cout << "\n==================================================\n";
    std::cout << "PERFORMANCE METRICS:\n";
    std::cout << "  Total time: " << total_time << " ms\n";
    std::cout << "  Average time per step: " << total_time / STEPS << " ms\n";
    std::cout << "  Steps per second: " << 1000.0f / (total_time / STEPS) << "\n";
    std::cout << "  Body updates per second: " 
              << N_BODIES * (1000.0f / (total_time / STEPS)) << "\n";
    std::cout << "\nOutput saved to: stable_output.txt\n";
    std::cout << "==================================================\n";
    
    // Cleanup in correct order
    cudaFree(d_bodies);
    cudaFree(d_tree);
    delete[] h_bodies;
    delete[] h_tree;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}