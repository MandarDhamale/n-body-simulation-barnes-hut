#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <fstream>
#include <string>
#include <stack>

// CUDA includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Constants
const float G = 1.0f;
const float SOFTENING = 0.5f;
const float THETA = 0.7f;
const float BOX_SIZE = 100.0f;
const float DT = 0.01f;

struct Body {
    float3 position;
    float3 velocity;
    float3 acceleration;
    float mass;
    int id;
};

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

// Initialize bodies
void initializeBodies(Body* bodies, int n_bodies) {
    std::srand(std::time(0));
    
    // Central massive body
    bodies[0].position = make_float3(0.0f, 0.0f, 0.0f);
    bodies[0].velocity = make_float3(0.0f, 0.0f, 0.0f);
    bodies[0].mass = 10000.0f;
    bodies[0].acceleration = make_float3(0.0f, 0.0f, 0.0f);
    bodies[0].id = 0;
    
    // Orbiting bodies in a disk
    for (int i = 1; i < n_bodies; i++) {
        float angle = 2.0f * M_PI * (float)std::rand() / RAND_MAX;
        float radius = 10.0f + 40.0f * (float)std::rand() / RAND_MAX;
        
        bodies[i].position.x = radius * cosf(angle);
        bodies[i].position.y = radius * sinf(angle);
        bodies[i].position.z = 0.1f * ((float)std::rand() / RAND_MAX - 0.5f);
        
        // Circular orbital velocity
        float distance = sqrtf(bodies[i].position.x * bodies[i].position.x +
                              bodies[i].position.y * bodies[i].position.y);
        float speed = sqrtf(G * bodies[0].mass / distance);
        
        bodies[i].velocity.x = -speed * sinf(angle);
        bodies[i].velocity.y = speed * cosf(angle);
        bodies[i].velocity.z = 0.0f;
        
        bodies[i].mass = 1.0f + 4.0f * (float)std::rand() / RAND_MAX;
        bodies[i].acceleration = make_float3(0.0f, 0.0f, 0.0f);
        bodies[i].id = i;
    }
}

// Build octree on CPU (simpler and more reliable)
void buildOctreeCPU(Body* bodies, OctreeNode* tree, int& node_count, int n_bodies, int max_nodes) {
    // Initialize root
    tree[0].bounds_min = make_float3(-BOX_SIZE, -BOX_SIZE, -BOX_SIZE);
    tree[0].bounds_max = make_float3(BOX_SIZE, BOX_SIZE, BOX_SIZE);
    tree[0].is_leaf = true;
    tree[0].body_index = -1;
    tree[0].total_mass = 0.0f;
    tree[0].size = 2.0f * BOX_SIZE;
    for (int i = 0; i < 8; i++) tree[0].children[i] = -1;
    
    node_count = 1;
    
    // Insert each body
    for (int body_idx = 0; body_idx < n_bodies; body_idx++) {
        Body body = bodies[body_idx];
        int current_node = 0;
        
        while (true) {
            OctreeNode* node = &tree[current_node];
            
            if (node->is_leaf) {
                if (node->body_index == -1) {
                    // Empty leaf - insert here
                    node->body_index = body_idx;
                    node->center_of_mass = body.position;
                    node->total_mass = body.mass;
                    break;
                } else {
                    // Leaf has one body - need to split
                    int old_body_idx = node->body_index;
                    Body old_body = bodies[old_body_idx];
                    
                    // Convert to internal node
                    node->is_leaf = false;
                    
                    // Create children
                    float3 center = make_float3(
                        (node->bounds_min.x + node->bounds_max.x) * 0.5f,
                        (node->bounds_min.y + node->bounds_max.y) * 0.5f,
                        (node->bounds_min.z + node->bounds_max.z) * 0.5f
                    );
                    
                    for (int octant = 0; octant < 8; octant++) {
                        int child_idx = node_count++;
                        if (child_idx >= max_nodes) continue;
                        
                        node->children[octant] = child_idx;
                        OctreeNode* child = &tree[child_idx];
                        
                        child->is_leaf = true;
                        child->body_index = -1;
                        child->total_mass = 0.0f;
                        
                        // Set child bounds
                        child->bounds_min.x = (octant & 1) ? center.x : node->bounds_min.x;
                        child->bounds_max.x = (octant & 1) ? node->bounds_max.x : center.x;
                        child->bounds_min.y = (octant & 2) ? center.y : node->bounds_min.y;
                        child->bounds_max.y = (octant & 2) ? node->bounds_max.y : center.y;
                        child->bounds_min.z = (octant & 4) ? center.z : node->bounds_min.z;
                        child->bounds_max.z = (octant & 4) ? node->bounds_max.z : center.z;
                        
                        child->size = (child->bounds_max.x - child->bounds_min.x) * 0.5f;
                        for (int i = 0; i < 8; i++) child->children[i] = -1;
                    }
                    
                    // Re-insert old body
                    float3 old_center = make_float3(
                        (node->bounds_min.x + node->bounds_max.x) * 0.5f,
                        (node->bounds_min.y + node->bounds_max.y) * 0.5f,
                        (node->bounds_min.z + node->bounds_max.z) * 0.5f
                    );
                    
                    int old_octant = 0;
                    if (old_body.position.x >= old_center.x) old_octant |= 1;
                    if (old_body.position.y >= old_center.y) old_octant |= 2;
                    if (old_body.position.z >= old_center.z) old_octant |= 4;
                    
                    int old_child = node->children[old_octant];
                    if (old_child != -1) {
                        tree[old_child].body_index = old_body_idx;
                        tree[old_child].center_of_mass = old_body.position;
                        tree[old_child].total_mass = old_body.mass;
                    }
                    
                    // Continue with current body
                    int new_octant = 0;
                    if (body.position.x >= center.x) new_octant |= 1;
                    if (body.position.y >= center.y) new_octant |= 2;
                    if (body.position.z >= center.z) new_octant |= 4;
                    
                    current_node = node->children[new_octant];
                }
            } else {
                // Internal node - go to appropriate child
                float3 center = make_float3(
                    (node->bounds_min.x + node->bounds_max.x) * 0.5f,
                    (node->bounds_min.y + node->bounds_max.y) * 0.5f,
                    (node->bounds_min.z + node->bounds_max.z) * 0.5f
                );
                
                int octant = 0;
                if (body.position.x >= center.x) octant |= 1;
                if (body.position.y >= center.y) octant |= 2;
                if (body.position.z >= center.z) octant |= 4;
                
                int child_idx = node->children[octant];
                if (child_idx == -1) {
                    // Create new child
                    child_idx = node_count++;
                    if (child_idx >= max_nodes) break;
                    
                    node->children[octant] = child_idx;
                    OctreeNode* child = &tree[child_idx];
                    
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
}

// Compute mass distribution recursively
void computeMassDistribution(OctreeNode* tree, int node_idx) {
    OctreeNode* node = &tree[node_idx];
    
    if (!node->is_leaf) {
        float3 com = make_float3(0.0f, 0.0f, 0.0f);
        float total_mass = 0.0f;
        
        for (int i = 0; i < 8; i++) {
            if (node->children[i] != -1) {
                computeMassDistribution(tree, node->children[i]);
                OctreeNode* child = &tree[node->children[i]];
                if (child->total_mass > 0) {
                    com.x += child->center_of_mass.x * child->total_mass;
                    com.y += child->center_of_mass.y * child->total_mass;
                    com.z += child->center_of_mass.z * child->total_mass;
                    total_mass += child->total_mass;
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

// Barnes-Hut force calculation kernel
__global__ void barnesHutKernel(Body* bodies, OctreeNode* tree, int n_bodies, int node_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bodies) return;
    
    float3 acceleration = make_float3(0.0f, 0.0f, 0.0f);
    float3 pos_i = bodies[i].position;
    
    // Use stack for iterative tree traversal
    int stack[64];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0; // Start from root
    
    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        OctreeNode node = tree[node_idx];
        
        if (node.total_mass == 0) continue;
        
        // Vector from body to node
        float3 r = make_float3(
            node.center_of_mass.x - pos_i.x,
            node.center_of_mass.y - pos_i.y,
            node.center_of_mass.z - pos_i.z
        );
        
        float dist_sq = r.x * r.x + r.y * r.y + r.z * r.z + SOFTENING;
        float dist = sqrtf(dist_sq);
        
        // Barnes-Hut criterion
        if (node.is_leaf || (node.size / dist < THETA)) {
            // Use approximation
            if (node.body_index != i) {
                float inv_dist_cube = 1.0f / (dist_sq * dist);
                float force = G * node.total_mass * inv_dist_cube;
                
                acceleration.x += force * r.x;
                acceleration.y += force * r.y;
                acceleration.z += force * r.z;
            }
        } else {
            // Open node and process children
            for (int child = 0; child < 8; child++) {
                if (node.children[child] != -1) {
                    if (stack_ptr < 64) {
                        stack[stack_ptr++] = node.children[child];
                    }
                }
            }
        }
    }
    
    bodies[i].acceleration = acceleration;
}

// Update bodies
__global__ void updateBodies(Body* bodies, int n_bodies, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bodies) return;
    
    // Update velocity
    bodies[i].velocity.x += bodies[i].acceleration.x * dt;
    bodies[i].velocity.y += bodies[i].acceleration.y * dt;
    bodies[i].velocity.z += bodies[i].acceleration.z * dt;
    
    // Update position
    bodies[i].position.x += bodies[i].velocity.x * dt;
    bodies[i].position.y += bodies[i].velocity.y * dt;
    bodies[i].position.z += bodies[i].velocity.z * dt;
}

// Save data to file
void saveData(const Body* bodies, int n_bodies, int step, const char* filename) {
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
    const int N_BODIES = 2048;      // Barnes-Hut excels with many bodies
    const int STEPS = 100;          // Fewer steps for demo
    const int BLOCK_SIZE = 256;
    const int MAX_TREE_NODES = N_BODIES * 3;
    
    std::cout << "REAL Barnes-Hut N-Body Simulation\n";
    std::cout << "==================================\n";
    std::cout << "Bodies: " << N_BODIES << "\n";
    std::cout << "Steps: " << STEPS << "\n";
    std::cout << "Algorithm: Barnes-Hut O(N log N)\n";
    std::cout << "Theta: " << THETA << "\n";
    
    // Clear output file
    std::ofstream clearfile("barnes_hut_data.txt");
    clearfile.close();
    
    // Host memory
    Body* h_bodies = new Body[N_BODIES];
    OctreeNode* h_tree = new OctreeNode[MAX_TREE_NODES];
    
    initializeBodies(h_bodies, N_BODIES);
    
    // Device memory
    Body* d_bodies;
    OctreeNode* d_tree;
    
    cudaMalloc(&d_bodies, N_BODIES * sizeof(Body));
    cudaMalloc(&d_tree, MAX_TREE_NODES * sizeof(OctreeNode));
    
    // Copy initial bodies to device
    cudaMemcpy(d_bodies, h_bodies, N_BODIES * sizeof(Body), cudaMemcpyHostToDevice);
    
    int grid_size = (N_BODIES + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    std::cout << "\nStarting Barnes-Hut simulation...\n";
    
    float total_time = 0.0f;
    
    // Save initial state
    saveData(h_bodies, N_BODIES, 0, "barnes_hut_data.txt");
    
    for (int step = 0; step < STEPS; step++) {
        cudaEventRecord(start);
        
        // Reset accelerations on device
        cudaMemset(d_bodies, 0, N_BODIES * sizeof(Body)); // Simple reset
        
        // Build tree on CPU (more reliable for demo)
        int node_count = 0;
        buildOctreeCPU(h_bodies, h_tree, node_count, N_BODIES, MAX_TREE_NODES);
        
        // Compute mass distribution
        computeMassDistribution(h_tree, 0);
        
        // Copy tree to device
        cudaMemcpy(d_tree, h_tree, MAX_TREE_NODES * sizeof(OctreeNode), cudaMemcpyHostToDevice);
        
        // Compute forces using Barnes-Hut
        barnesHutKernel<<<grid_size, BLOCK_SIZE>>>(d_bodies, d_tree, N_BODIES, node_count);
        cudaDeviceSynchronize();
        
        // Update positions
        updateBodies<<<grid_size, BLOCK_SIZE>>>(d_bodies, N_BODIES, DT);
        cudaDeviceSynchronize();
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float step_time = 0.0f;
        cudaEventElapsedTime(&step_time, start, stop);
        total_time += step_time;
        
        // Copy back to host for saving and next tree build
        cudaMemcpy(h_bodies, d_bodies, N_BODIES * sizeof(Body), cudaMemcpyDeviceToHost);
        
        // Save data every 10 steps
        if (step % 10 == 0) {
            saveData(h_bodies, N_BODIES, step, "barnes_hut_data.txt");
            std::cout << "Step " << step << " - Time: " << step_time << " ms, Tree nodes: " << node_count << "\n";
        }
    }
    
    // Save final state
    saveData(h_bodies, N_BODIES, STEPS, "barnes_hut_data.txt");
    
    std::cout << "\n=== PERFORMANCE REPORT ===\n";
    std::cout << "Total time: " << total_time << " ms\n";
    std::cout << "Average time per step: " << total_time / STEPS << " ms\n";
    std::cout << "Steps per second: " << 1000.0f / (total_time / STEPS) << "\n";
    std::cout << "Body updates per second: " 
              << N_BODIES * (1000.0f / (total_time / STEPS)) << "\n";
    std::cout << "\nData saved to 'barnes_hut_data.txt'\n";
    
    // Cleanup
    delete[] h_bodies;
    delete[] h_tree;
    cudaFree(d_bodies);
    cudaFree(d_tree);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}