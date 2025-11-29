#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <fstream>
#include <string>

// CUDA includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Constants
const float G = 1.0f;
const float SOFTENING = 0.5f;
const float THETA = 0.7f;
const float BOX_SIZE = 100.0f;
const float MAX_VELOCITY = 20.0f;  // Velocity clamping for stability

struct Body {
    float3 position;
    float3 velocity;
    float3 acceleration;
    float mass;
    int id;
};

struct OctreeNode {
    float3 center_of_mass;
    float3 bounds_min, bounds_max;
    float total_mass;
    int body_index;
    int children[8];
    bool is_leaf;
    float size;  // Node size for Barnes-Hut criterion
};

// Initialize bodies with more realistic distribution
void initializeBodies(Body* bodies, int n_bodies) {
    std::srand(std::time(0));
    
    // Central massive black hole
    bodies[0].position = make_float3(0.0f, 0.0f, 0.0f);
    bodies[0].velocity = make_float3(0.0f, 0.0f, 0.0f);
    bodies[0].mass = 5000.0f;
    bodies[0].acceleration = make_float3(0.0f, 0.0f, 0.0f);
    bodies[0].id = 0;
    
    // Create spiral galaxy-like distribution
    int arms = 3;
    for (int i = 1; i < n_bodies; i++) {
        float arm = (i % arms) * (2.0f * M_PI / arms);
        float angle = arm + 0.5f * (float)std::rand() / RAND_MAX;
        float radius = 10.0f + 35.0f * powf((float)std::rand() / RAND_MAX, 0.7f);
        
        // Spiral density wave
        angle += 0.3f * logf(radius / 10.0f);
        
        bodies[i].position.x = radius * cosf(angle);
        bodies[i].position.y = radius * sinf(angle);
        bodies[i].position.z = 0.2f * (float)std::rand() / RAND_MAX - 0.1f;
        
        // Keplerian orbital velocity with some random component
        float distance = sqrtf(bodies[i].position.x * bodies[i].position.x +
                              bodies[i].position.y * bodies[i].position.y);
        float kepler_speed = sqrtf(G * bodies[0].mass / distance);
        
        // Add some random velocity dispersion
        float speed_variation = 0.2f * (float)std::rand() / RAND_MAX - 0.1f;
        float actual_speed = kepler_speed * (1.0f + speed_variation);
        
        // Tangential velocity with small radial component
        float tangential_angle = angle + M_PI / 2.0f;
        float radial_component = 0.05f * (float)std::rand() / RAND_MAX - 0.025f;
        
        bodies[i].velocity.x = actual_speed * cosf(tangential_angle) + 
                              radial_component * cosf(angle);
        bodies[i].velocity.y = actual_speed * sinf(tangential_angle) + 
                              radial_component * sinf(angle);
        bodies[i].velocity.z = 0.01f * (float)std::rand() / RAND_MAX - 0.005f;
        
        // Mass following Salpeter-like distribution
        bodies[i].mass = 0.5f + 4.5f * powf((float)std::rand() / RAND_MAX, 2.0f);
        bodies[i].acceleration = make_float3(0.0f, 0.0f, 0.0f);
        bodies[i].id = i;
    }
}

// Reset accelerations
__global__ void resetAccelerations(Body* bodies, int n_bodies) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bodies) return;
    
    bodies[i].acceleration = make_float3(0.0f, 0.0f, 0.0f);
}

// Simple all-pairs gravity for comparison
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
                              pos_j.z - pos_i.z);
        
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

// Initialize octree structure
__global__ void initializeTree(OctreeNode* tree, int tree_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tree_size) return;
    
    tree[idx].center_of_mass = make_float3(0.0f, 0.0f, 0.0f);
    tree[idx].bounds_min = make_float3(-BOX_SIZE, -BOX_SIZE, -BOX_SIZE);
    tree[idx].bounds_max = make_float3(BOX_SIZE, BOX_SIZE, BOX_SIZE);
    tree[idx].total_mass = 0.0f;
    tree[idx].body_index = -1;
    tree[idx].is_leaf = true;
    tree[idx].size = 2.0f * BOX_SIZE;
    
    for (int i = 0; i < 8; i++) {
        tree[idx].children[i] = -1;
    }
}

// Get octant for a point relative to center
__device__ int getOctant(const float3& point, const float3& center) {
    int octant = 0;
    if (point.x >= center.x) octant |= 1;
    if (point.y >= center.y) octant |= 2;
    if (point.z >= center.z) octant |= 4;
    return octant;
}

// Compute child bounds
__device__ void computeChildBounds(int octant, const float3& parent_min, 
                                 const float3& parent_max, float3& child_min, 
                                 float3& child_max) {
    float3 center = make_float3(
        (parent_min.x + parent_max.x) * 0.5f,
        (parent_min.y + parent_max.y) * 0.5f,
        (parent_min.z + parent_max.z) * 0.5f
    );
    
    child_min.x = (octant & 1) ? center.x : parent_min.x;
    child_max.x = (octant & 1) ? parent_max.x : center.x;
    
    child_min.y = (octant & 2) ? center.y : parent_min.y;
    child_max.y = (octant & 2) ? parent_max.y : center.y;
    
    child_min.z = (octant & 4) ? center.z : parent_min.z;
    child_max.z = (octant & 4) ? parent_max.z : center.z;
}

// Build octree - single threaded for correctness
__global__ void buildOctree(Body* bodies, OctreeNode* tree, int n_bodies, int max_nodes) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    int node_count = 1; // Root node at index 0
    
    // Initialize root
    tree[0].bounds_min = make_float3(-BOX_SIZE, -BOX_SIZE, -BOX_SIZE);
    tree[0].bounds_max = make_float3(BOX_SIZE, BOX_SIZE, BOX_SIZE);
    tree[0].is_leaf = true;
    tree[0].body_index = -1;
    tree[0].total_mass = 0.0f;
    tree[0].size = 2.0f * BOX_SIZE;
    
    // Insert each body into the tree
    for (int body_idx = 0; body_idx < n_bodies; body_idx++) {
        int current_node = 0;
        Body body = bodies[body_idx];
        
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
                    // Leaf with one body - need to split
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
                        if (node_count >= max_nodes) continue;
                        
                        int child_idx = node_count++;
                        node->children[octant] = child_idx;
                        
                        OctreeNode* child = &tree[child_idx];
                        child->is_leaf = true;
                        child->body_index = -1;
                        child->total_mass = 0.0f;
                        computeChildBounds(octant, node->bounds_min, node->bounds_max,
                                         child->bounds_min, child->bounds_max);
                        child->size = (child->bounds_max.x - child->bounds_min.x) * 0.5f;
                    }
                    
                    // Re-insert old body
                    int old_octant = getOctant(old_body.position, center);
                    int old_child = node->children[old_octant];
                    if (old_child != -1) {
                        tree[old_child].body_index = old_body_idx;
                        tree[old_child].center_of_mass = old_body.position;
                        tree[old_child].total_mass = old_body.mass;
                    }
                    
                    // Continue with current body
                    int new_octant = getOctant(body.position, center);
                    current_node = node->children[new_octant];
                }
            } else {
                // Internal node - traverse to appropriate child
                float3 center = make_float3(
                    (node->bounds_min.x + node->bounds_max.x) * 0.5f,
                    (node->bounds_min.y + node->bounds_max.y) * 0.5f,
                    (node->bounds_min.z + node->bounds_max.z) * 0.5f
                );
                
                int octant = getOctant(body.position, center);
                int child_idx = node->children[octant];
                
                if (child_idx == -1) {
                    // Should not happen in proper implementation
                    break;
                } else {
                    current_node = child_idx;
                }
            }
        }
    }
}

// Compute center of mass for all nodes (bottom-up)
__global__ void computeMassDistribution(OctreeNode* tree, Body* bodies, int max_nodes) {
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_idx >= max_nodes) return;
    
    OctreeNode* node = &tree[node_idx];
    
    if (!node->is_leaf) {
        float3 com = make_float3(0.0f, 0.0f, 0.0f);
        float total_mass = 0.0f;
        int valid_children = 0;
        
        for (int i = 0; i < 8; i++) {
            int child_idx = node->children[i];
            if (child_idx != -1) {
                OctreeNode* child = &tree[child_idx];
                if (child->total_mass > 0) {
                    com.x += child->center_of_mass.x * child->total_mass;
                    com.y += child->center_of_mass.y * child->total_mass;
                    com.z += child->center_of_mass.z * child->total_mass;
                    total_mass += child->total_mass;
                    valid_children++;
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

// Barnes-Hut force calculation with iterative tree traversal
__global__ void computeBarnesHutForces(Body* bodies, OctreeNode* tree, int n_bodies, int max_nodes) {
    int body_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (body_idx >= n_bodies) return;
    
    Body* body = &bodies[body_idx];
    float3 acceleration = make_float3(0.0f, 0.0f, 0.0f);
    
    // Use stack for iterative tree traversal
    int stack[64];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0; // Start from root
    
    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        OctreeNode* node = &tree[node_idx];
        
        if (node->total_mass == 0) continue;
        
        // Vector from body to node
        float3 r = make_float3(
            node->center_of_mass.x - body->position.x,
            node->center_of_mass.y - body->position.y,
            node->center_of_mass.z - body->position.z
        );
        
        float dist_sq = r.x * r.x + r.y * r.y + r.z * r.z + SOFTENING;
        float dist = sqrtf(dist_sq);
        
        // Barnes-Hut criterion: s/d < θ
        if (node->is_leaf || (node->size / dist < THETA)) {
            // Use this node as approximation (or direct if leaf)
            if (node->body_index != body_idx) { // Avoid self-force
                float inv_dist_cube = 1.0f / (dist_sq * dist);
                float force = G * node->total_mass * inv_dist_cube;
                
                acceleration.x += force * r.x;
                acceleration.y += force * r.y;
                acceleration.z += force * r.z;
            }
        } else {
            // Need to open node - push children onto stack
            for (int i = 0; i < 8; i++) {
                if (node->children[i] != -1) {
                    if (stack_ptr < 64) {
                        stack[stack_ptr++] = node->children[i];
                    }
                }
            }
        }
    }
    
    body->acceleration = acceleration;
}

// Update positions with velocity Verlet integration (more stable)
__global__ void updateBodiesVerlet(Body* bodies, int n_bodies, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bodies) return;
    
    // Velocity Verlet integration
    bodies[i].velocity.x += bodies[i].acceleration.x * dt * 0.5f;
    bodies[i].velocity.y += bodies[i].acceleration.y * dt * 0.5f;
    bodies[i].velocity.z += bodies[i].acceleration.z * dt * 0.5f;
    
    // Clamp velocity for stability
    float speed_sq = bodies[i].velocity.x * bodies[i].velocity.x +
                    bodies[i].velocity.y * bodies[i].velocity.y +
                    bodies[i].velocity.z * bodies[i].velocity.z;
    
    if (speed_sq > MAX_VELOCITY * MAX_VELOCITY) {
        float scale = MAX_VELOCITY / sqrtf(speed_sq);
        bodies[i].velocity.x *= scale;
        bodies[i].velocity.y *= scale;
        bodies[i].velocity.z *= scale;
    }
    
    // Update position
    bodies[i].position.x += bodies[i].velocity.x * dt;
    bodies[i].position.y += bodies[i].velocity.y * dt;
    bodies[i].position.z += bodies[i].velocity.z * dt;
    
    // Second half of velocity update will be done in next step
}

// Complete velocity update for Verlet
__global__ void completeVerletUpdate(Body* bodies, int n_bodies, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bodies) return;
    
    bodies[i].velocity.x += bodies[i].acceleration.x * dt * 0.5f;
    bodies[i].velocity.y += bodies[i].acceleration.y * dt * 0.5f;
    bodies[i].velocity.z += bodies[i].acceleration.z * dt * 0.5f;
}

// Save positions to file for visualization
void savePositionsToFile(const Body* bodies, int n_bodies, int step, const std::string& filename) {
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) return;
    
    file << "Step " << step << "\n";
    for (int i = 0; i < n_bodies; i++) {
        file << bodies[i].position.x << " " << bodies[i].position.y << " " 
             << bodies[i].position.z << " " << bodies[i].mass << "\n";
    }
    file << "\n";
    file.close();
}

// Calculate and print simulation statistics
void printStatistics(const Body* bodies, int n_bodies, int step, float time, 
                    float step_time, bool use_barnes_hut) {
    float3 total_momentum = make_float3(0.0f, 0.0f, 0.0f);
    float total_energy = 0.0f;
    float max_speed = 0.0f;
    
    for (int i = 0; i < n_bodies; i++) {
        total_momentum.x += bodies[i].velocity.x * bodies[i].mass;
        total_momentum.y += bodies[i].velocity.y * bodies[i].mass;
        total_momentum.z += bodies[i].velocity.z * bodies[i].mass;
        
        float speed_sq = bodies[i].velocity.x * bodies[i].velocity.x +
                        bodies[i].velocity.y * bodies[i].velocity.y +
                        bodies[i].velocity.z * bodies[i].velocity.z;
        max_speed = fmaxf(max_speed, sqrtf(speed_sq));
        
        // Kinetic energy
        total_energy += 0.5f * bodies[i].mass * speed_sq;
        
        // Potential energy (simplified)
        for (int j = i + 1; j < n_bodies; j++) {
            float3 r = make_float3(
                bodies[j].position.x - bodies[i].position.x,
                bodies[j].position.y - bodies[i].position.y,
                bodies[j].position.z - bodies[i].position.z
            );
            float dist = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z + SOFTENING);
            total_energy -= G * bodies[i].mass * bodies[j].mass / dist;
        }
    }
    
    std::cout << "Step " << step << " (Time: " << time << "s)" << std::endl;
    std::cout << "  Method: " << (use_barnes_hut ? "Barnes-Hut" : "Direct") << std::endl;
    std::cout << "  Step time: " << step_time << " ms" << std::endl;
    std::cout << "  Total energy: " << total_energy << std::endl;
    std::cout << "  Max speed: " << max_speed << std::endl;
    std::cout << "  Total momentum: (" << total_momentum.x << ", " 
              << total_momentum.y << ", " << total_momentum.z << ")" << std::endl;
}

int main() {
    const int N_BODIES = 2048;        // More bodies to show Barnes-Hut advantage
    const float DT = 0.01f;
    const int STEPS = 500;            // More steps for better visualization
    const int BLOCK_SIZE = 256;
    const int MAX_TREE_NODES = N_BODIES * 3;
    
    const bool USE_BARNES_HUT = true; // Toggle between methods
    const bool SAVE_DATA = true;      // Save data for visualization
    
    std::cout << "ENHANCED N-Body Simulation" << std::endl;
    std::cout << "Bodies: " << N_BODIES << ", Steps: " << STEPS << ", DT: " << DT << std::endl;
    std::cout << "Method: " << (USE_BARNES_HUT ? "Barnes-Hut O(N log N)" : "Direct O(N²)") << std::endl;
    
    // Host memory
    Body* h_bodies = new Body[N_BODIES];
    initializeBodies(h_bodies, N_BODIES);
    
    // Device memory
    Body* d_bodies;
    OctreeNode* d_tree;
    
    cudaMalloc(&d_bodies, N_BODIES * sizeof(Body));
    cudaMalloc(&d_tree, MAX_TREE_NODES * sizeof(OctreeNode));
    
    // Copy to device
    cudaMemcpy(d_bodies, h_bodies, N_BODIES * sizeof(Body), cudaMemcpyHostToDevice);
    
    int grid_size = (N_BODIES + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int tree_grid_size = (MAX_TREE_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Clear output file
    if (SAVE_DATA) {
        std::ofstream clearfile("nbody_data.txt");
        clearfile.close();
    }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    std::cout << "\nStarting simulation..." << std::endl;
    
    float total_time = 0.0f;
    float barnes_hut_time = 0.0f;
    float direct_time = 0.0f;
    
    for (int step = 0; step < STEPS; step++) {
        cudaEventRecord(start);
        
        // Reset accelerations
        resetAccelerations<<<grid_size, BLOCK_SIZE>>>(d_bodies, N_BODIES);
        cudaDeviceSynchronize();
        
        if (USE_BARNES_HUT && step % 5 == 0) {
            // Rebuild tree periodically (every 5 steps for efficiency)
            initializeTree<<<tree_grid_size, BLOCK_SIZE>>>(d_tree, MAX_TREE_NODES);
            cudaDeviceSynchronize();
            
            buildOctree<<<1, 1>>>(d_bodies, d_tree, N_BODIES, MAX_TREE_NODES);
            cudaDeviceSynchronize();
            
            computeMassDistribution<<<tree_grid_size, BLOCK_SIZE>>>(d_tree, d_bodies, MAX_TREE_NODES);
            cudaDeviceSynchronize();
        }
        
        // Compute forces
        if (USE_BARNES_HUT) {
            computeBarnesHutForces<<<grid_size, BLOCK_SIZE>>>(d_bodies, d_tree, N_BODIES, MAX_TREE_NODES);
        } else {
            computeDirectForces<<<grid_size, BLOCK_SIZE>>>(d_bodies, N_BODIES);
        }
        cudaDeviceSynchronize();
        
        // Update positions
        updateBodiesVerlet<<<grid_size, BLOCK_SIZE>>>(d_bodies, N_BODIES, DT);
        cudaDeviceSynchronize();
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float step_time = 0.0f;
        cudaEventElapsedTime(&step_time, start, stop);
        total_time += step_time;
        
        if (USE_BARNES_HUT) {
            barnes_hut_time += step_time;
        } else {
            direct_time += step_time;
        }
        
        if (step % 50 == 0) {
            // Copy back for statistics and saving
            cudaMemcpy(h_bodies, d_bodies, N_BODIES * sizeof(Body), cudaMemcpyDeviceToHost);
            printStatistics(h_bodies, N_BODIES, step, step * DT, step_time, USE_BARNES_HUT);
            
            if (SAVE_DATA) {
                savePositionsToFile(h_bodies, N_BODIES, step, "nbody_data.txt");
            }
        }
    }
    
    // Final state
    cudaMemcpy(h_bodies, d_bodies, N_BODIES * sizeof(Body), cudaMemcpyDeviceToHost);
    std::cout << "\nFinal state:" << std::endl;
    printStatistics(h_bodies, N_BODIES, STEPS, STEPS * DT, 0.0f, USE_BARNES_HUT);
    
    if (SAVE_DATA) {
        savePositionsToFile(h_bodies, N_BODIES, STEPS, "nbody_data.txt");
        std::cout << "Data saved to 'nbody_data.txt'" << std::endl;
    }
    
    std::cout << "\nSimulation completed!" << std::endl;
    std::cout << "Total time: " << total_time << " ms" << std::endl;
    std::cout << "Average time per step: " << total_time / STEPS << " ms" << std::endl;
    
    if (USE_BARNES_HUT) {
        std::cout << "Barnes-Hut performance: " << barnes_hut_time << " ms total" << std::endl;
    } else {
        std::cout << "Direct method performance: " << direct_time << " ms total" << std::endl;
    }
    
    // Cleanup
    delete[] h_bodies;
    cudaFree(d_bodies);
    cudaFree(d_tree);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}