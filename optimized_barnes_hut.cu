#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <vector>

// CUDA includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ==========================================
// CONSTANTS & CONFIGURATION
// ==========================================
const float G = 1.0f;           // Gravitational constant (normalized)
const float SOFTENING = 0.5f;   // Prevents singularity (division by zero) at r=0
const float THETA = 0.7f;       // Accuracy knob: 0.0=Exact(Slow), 1.0=Approx(Fast)
const float DT = 0.01f;         // Time step per simulation frame
const float BOX_SIZE = 100.0f;  // Simulation boundary (half-width)

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// ==========================================
// DATA STRUCTURES
// ==========================================

// Body structure 
// ALIGNMENT NOTE: alignas(16) ensures 128-bit memory alignment.
// This allows the GPU to read 'float4' sized chunks in a single transaction,
// optimizing memory bandwidth (Coalesced Access).
struct alignas(16) Body {
    float3 position;
    float3 velocity;
    float3 acceleration;
    float mass;
};

// OctreeNode structure
// Represents a cube in space. 
// - If is_leaf=true: It contains 0 or 1 body.
// - If is_leaf=false: It contains up to 8 children (sub-cubes).
struct alignas(16) OctreeNode {
    float3 center_of_mass;  // Weighted position of all bodies inside
    float total_mass;       // Sum of all masses inside
    float3 bounds_min;      // Top-left-front corner coordinate
    int body_index;         // Index of the particle (if leaf), otherwise -1
    int children[8];        // Indices of child nodes in the tree array
    bool is_leaf;           // Flag: Is this a terminal node?
    float size;             // Width of the cube
};

// ==========================================
// CUDA KERNELS (GPU CODE)
// ==========================================

// Kernel 1: Reset Accelerations
// Simple utility to clear the acceleration vector before each step.
__global__ void resetAccelerations(Body* bodies, int n_bodies) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_bodies) {
        bodies[idx].acceleration = make_float3(0.0f, 0.0f, 0.0f);
    }
}

// Kernel 2: Compute Forces using Barnes-Hut
__global__ void computeBarnesHutForces(Body* bodies, OctreeNode* tree, 
                                       int n_bodies, int tree_size, float G, float THETA, float SOFTENING) {
    // Identify which body this specific thread handles
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bodies) return;
    
    float3 accel = make_float3(0.0f, 0.0f, 0.0f);
    float3 pos_i = bodies[i].position;
    
    // Iterative Tree Traversal using a Stack
    // GPUs have small stack memory, so we manage our own stack array instead of using recursion.
    // A depth of 64 is sufficient for a tree covering a massive spatial volume.
    int stack[64];
    int stack_ptr = 0;
    
    // Push the Root Node (index 0) onto the stack to start
    stack[stack_ptr++] = 0; 
    
    while (stack_ptr > 0) {
        // Pop a node from the stack
        int node_idx = stack[--stack_ptr];
        
        // Safety check for bounds
        if (node_idx < 0 || node_idx >= tree_size) continue;
        
        OctreeNode node = tree[node_idx];
        
        // Skip empty nodes 
        if (node.total_mass <= 0.0f) continue;
        
        // Calculate vector r from Body[i] to the Node's Center of Mass
        float3 r = make_float3(
            node.center_of_mass.x - pos_i.x,
            node.center_of_mass.y - pos_i.y,
            node.center_of_mass.z - pos_i.z
        );
        
        // Distance squared + Softening (to avoid infinity at dist=0)
        float dist_sq = r.x*r.x + r.y*r.y + r.z*r.z + SOFTENING;
        float dist = sqrtf(dist_sq);
        
        // --- MULTIPOLE ACCEPTANCE CRITERION (MAC) ---
        // We calculate force if:
        // 1. The node is a LEAF (contains an actual body)
        // 2. The node is FAR ENOUGH away to treat as a single mass (size/dist < THETA)
        
        bool is_close = (node.size / dist) >= THETA;
        
        if (node.is_leaf || !is_close) {
            // EXCLUSION CHECK: Don't calculate force from yourself
            if (node.body_index != i) {
                // Newton's Gravity Formula: F = G * m1 * m2 / r^2
                // We calculate Acceleration: a = F / m1 = G * m2 / r^2
                // Vector form: a = (G * m2 / r^3) * vec_r
                
                float inv_dist_cube = 1.0f / (dist_sq * dist);
                float s = G * node.total_mass * inv_dist_cube;
                
                accel.x += s * r.x;
                accel.y += s * r.y;
                accel.z += s * r.z;
            }
        } else {
            // If node is Internal AND too close, we cannot approximate.
            // Push all 8 children to the stack to check them individually.
            for (int child = 0; child < 8; child++) {
                int child_idx = node.children[child];
                if (child_idx != -1) {
                    stack[stack_ptr++] = child_idx;
                }
            }
        }
    }
    
    // Save the computed acceleration back to global memory
    bodies[i].acceleration = accel;
}

// Kernel 3: Velocity Verlet Integration
// Updates position and velocity based on the computed acceleration.
__global__ void updateBodies(Body* bodies, int n_bodies, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bodies) return;
    
    // 1. Update Velocity (Half-step)
    bodies[i].velocity.x += 0.5f * bodies[i].acceleration.x * dt;
    bodies[i].velocity.y += 0.5f * bodies[i].acceleration.y * dt;
    bodies[i].velocity.z += 0.5f * bodies[i].acceleration.z * dt;
    
    // 2. Update Position
    bodies[i].position.x += bodies[i].velocity.x * dt;
    bodies[i].position.y += bodies[i].velocity.y * dt;
    bodies[i].position.z += bodies[i].velocity.z * dt;
    
}

// ==========================================
// HOST FUNCTIONS (CPU)
// ==========================================

// Build Octree on CPU
// This function takes the array of bodies and inserts them one by one into the tree.
// Returns the total number of nodes used.
int buildOctreeOptimized(Body* bodies, OctreeNode* tree, int n_bodies, int max_nodes) {
    // 1. Initialize Root Node
    tree[0].bounds_min = make_float3(-BOX_SIZE, -BOX_SIZE, -BOX_SIZE);
    tree[0].center_of_mass = make_float3(0.0f, 0.0f, 0.0f);
    tree[0].is_leaf = true;
    tree[0].body_index = -1;
    tree[0].total_mass = 0.0f;
    tree[0].size = 2.0f * BOX_SIZE;
    for (int i = 0; i < 8; i++) tree[0].children[i] = -1;
    
    int node_count = 1;
    
    // 2. Insert bodies
    for (int body_idx = 0; body_idx < n_bodies; body_idx++) {
        Body body = bodies[body_idx];
        int current_node = 0;
        
        while (true) {
            if (current_node >= max_nodes) break; // Safety break
            
            OctreeNode* node = &tree[current_node];
            
            if (node->is_leaf) {
                if (node->body_index == -1) {
                    // Case A: Empty Leaf -> Place body here
                    node->body_index = body_idx;
                    node->center_of_mass = body.position;
                    node->total_mass = body.mass;
                    break;
                } else {
                    // Case B: Occupied Leaf -> Split needed
                    // 1. Save the old body currently in this leaf
                    int old_body_idx = node->body_index;
                    Body old_body = bodies[old_body_idx];
                    
                    // 2. Mark current node as Internal (not leaf)
                    node->is_leaf = false;
                    node->body_index = -1; // Internal nodes don't hold bodies directly
                    
                    float3 center = make_float3(
                        node->bounds_min.x + node->size * 0.5f,
                        node->bounds_min.y + node->size * 0.5f,
                        node->bounds_min.z + node->size * 0.5f
                    );
                    
                    // 3. Create 8 children for this node
                    for (int octant = 0; octant < 8; octant++) {
                        if (node_count >= max_nodes) break;
                        int child_idx = node_count++;
                        node->children[octant] = child_idx;
                        
                        OctreeNode* child = &tree[child_idx];
                        child->is_leaf = true;
                        child->body_index = -1;
                        child->total_mass = 0.0f;
                        child->size = node->size * 0.5f;
                        
                        // Determine child bounds using bitwise logic
                        // (If bit 0 is set, x is in the right half, etc.)
                        child->bounds_min.x = (octant & 1) ? center.x : node->bounds_min.x;
                        child->bounds_min.y = (octant & 2) ? center.y : node->bounds_min.y;
                        child->bounds_min.z = (octant & 4) ? center.z : node->bounds_min.z;
                        
                        for (int i = 0; i < 8; i++) child->children[i] = -1;
                    }
                    
                    // 4. Re-insert the OLD body into the appropriate child
                    // (We don't break here, we just push it down)
                    float3 delta = make_float3(
                        old_body.position.x - center.x,
                        old_body.position.y - center.y,
                        old_body.position.z - center.z
                    );
                    int octant = 0;
                    if (delta.x >= 0) octant |= 1;
                    if (delta.y >= 0) octant |= 2;
                    if (delta.z >= 0) octant |= 4;
                    
                    int child_idx = node->children[octant];
                    tree[child_idx].body_index = old_body_idx;
                    tree[child_idx].center_of_mass = old_body.position;
                    tree[child_idx].total_mass = old_body.mass;
                    
                    // 5. Continue the loop to insert the NEW body (body_idx)
                    // The while loop will now see this node is not a leaf and descend.
                }
            } else {
                // Case C: Internal Node -> Traverse down
                float3 center = make_float3(
                    node->bounds_min.x + node->size * 0.5f,
                    node->bounds_min.y + node->size * 0.5f,
                    node->bounds_min.z + node->size * 0.5f
                );
                
                float3 delta = make_float3(
                    body.position.x - center.x,
                    body.position.y - center.y,
                    body.position.z - center.z
                );
                
                int octant = 0;
                if (delta.x >= 0) octant |= 1;
                if (delta.y >= 0) octant |= 2;
                if (delta.z >= 0) octant |= 4;
                
                int child_idx = node->children[octant];
                
                // If child doesn't exist, create it
                if (child_idx == -1) {
                     // Error handling or dynamic creation
                     break; 
                }
                
                // Move focus to the child node
                current_node = child_idx;
            }
        }
    }
    return node_count;
}

// Compute Mass Distribution (Post-Order Traversal)
// Propagates mass and center-of-mass from leaves up to the root.
void computeMassDistributionIterative(OctreeNode* tree, int tree_size) {
    // Iterate backwards from the last node to 0. 
    // This ensures children are processed before their parents.
    for (int i = tree_size - 1; i >= 0; i--) {
        OctreeNode* node = &tree[i];
        
        if (!node->is_leaf) {
            float3 com = make_float3(0.0f, 0.0f, 0.0f);
            float total_mass = 0.0f;
            
            for (int child = 0; child < 8; child++) {
                int child_idx = node->children[child];
                if (child_idx != -1 && child_idx < tree_size) {
                    OctreeNode* child_node = &tree[child_idx];
                    if (child_node->total_mass > 0) {
                        com.x += child_node->center_of_mass.x * child_node->total_mass;
                        com.y += child_node->center_of_mass.y * child_node->total_mass;
                        com.z += child_node->center_of_mass.z * child_node->total_mass;
                        total_mass += child_node->total_mass;
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
}

// Helper: Random Disk Initialization
void initializeBodies(Body* bodies, int n_bodies) {
    srand(time(NULL));
    
    // Supermassive Black Hole at Center
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
        
        // Orbital Velocity calculation: v = sqrt(GM/r)
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

// Helper: Save to file
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

// ==========================================
// MAIN FUNCTION
// ==========================================
int main() {
    const int N_BODIES = 2000; // Number of bodies in the simulation, using 2000 for project submission to reduce project size, change to 7000 for full simulation, or any other value
    const int STEPS = 150;       // Number of simulation steps, using 150 for project submission to reduce project size, change to 1000 for full simulation
    const int BLOCK_SIZE = 256; 
    const int MAX_TREE_NODES = N_BODIES * 4;
    
    std::cout << "==================================================\n";
    std::cout << "   OPTIMIZED BARNES-HUT \n";
    std::cout << "==================================================\n\n";
    
    std::cout << "Configuration:\n";
    std::cout << "  Bodies: " << N_BODIES << "\n";
    std::cout << "  Steps: " << STEPS << "\n";
    std::cout << "  Block Size: " << BLOCK_SIZE << "\n";
    
    // Clean output file
    std::ofstream clearfile("optimized_output.txt");
    clearfile.close();
    
    // --- 1. HOST MEMORY ALLOCATION ---
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
    saveData(h_bodies, N_BODIES, 0, "optimized_output.txt");
    
    // --- 2. DEVICE MEMORY ALLOCATION ---
    Body* d_bodies = nullptr;
    OctreeNode* d_tree = nullptr;
    
    cudaMalloc(&d_bodies, N_BODIES * sizeof(Body));
    cudaMalloc(&d_tree, MAX_TREE_NODES * sizeof(OctreeNode));
    
    // Copy Initial State to GPU
    cudaMemcpy(d_bodies, h_bodies, N_BODIES * sizeof(Body), cudaMemcpyHostToDevice);
    
    int grid_size = (N_BODIES + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    std::cout << "Running simulation...\n";
    float total_time = 0.0f;
    
    // --- 3. SIMULATION LOOP ---
    for (int step = 0; step < STEPS; step++) {
        cudaEventRecord(start);
        
        // A. Reset Accelerations
        resetAccelerations<<<grid_size, BLOCK_SIZE>>>(d_bodies, N_BODIES);
        cudaDeviceSynchronize();
        
        // B. Build Tree (Host Side)
        // Copy bodies back to host to build the tree
        cudaMemcpy(h_bodies, d_bodies, N_BODIES * sizeof(Body), cudaMemcpyDeviceToHost);
        int tree_size = buildOctreeOptimized(h_bodies, h_tree, N_BODIES, MAX_TREE_NODES);
        computeMassDistributionIterative(h_tree, tree_size);
        
        // Copy Tree to Device
        cudaMemcpy(d_tree, h_tree, tree_size * sizeof(OctreeNode), cudaMemcpyHostToDevice);
        
        // C. Compute Forces (The Optimized Barnes-Hut Kernel)
        computeBarnesHutForces<<<grid_size, BLOCK_SIZE>>>(
            d_bodies, d_tree, N_BODIES, tree_size, G, THETA, SOFTENING);
        cudaDeviceSynchronize();
        
        // D. Update Positions (Integration)
        updateBodies<<<grid_size, BLOCK_SIZE>>>(d_bodies, N_BODIES, DT);
        cudaDeviceSynchronize();
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float step_time = 0.0f;
        cudaEventElapsedTime(&step_time, start, stop);
        total_time += step_time;
        
        // Output progress
        if (step % 1 == 0) {
            cudaMemcpy(h_bodies, d_bodies, N_BODIES * sizeof(Body), cudaMemcpyDeviceToHost); 
            saveData(h_bodies, N_BODIES, step, "optimized_output.txt");
            std::cout << "  Step " << step << ": " << step_time << " ms, Tree nodes: " << tree_size << "\n";
        }
    }
    
    // Final save
    cudaMemcpy(h_bodies, d_bodies, N_BODIES * sizeof(Body), cudaMemcpyDeviceToHost);
    saveData(h_bodies, N_BODIES, STEPS, "optimized_output.txt");
    
    // --- 4. RESULTS ---
    std::cout << "\n==================================================\n";
    std::cout << "PERFORMANCE RESULTS:\n";
    std::cout << "  Total time: " << total_time << " ms\n";
    std::cout << "  Average time per step: " << total_time / STEPS << " ms\n";
    std::cout << "\nOutput saved to: simulation_output.txt\n";
    std::cout << "==================================================\n";
    
    // Cleanup
    cudaFree(d_bodies);
    cudaFree(d_tree);
    delete[] h_bodies;
    delete[] h_tree;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}