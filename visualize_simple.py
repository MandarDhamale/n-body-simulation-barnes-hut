import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

print("BARNES-HUT SIMULATION VISUALIZATION")
print("=" * 50)

# Read data
data = []
with open('optimized_output.txt', 'r') as f:
    lines = f.readlines()

step_data = []
for line in lines:
    if line.startswith('Step'):
        if step_data:
            data.append(np.array(step_data))
            step_data = []
    elif line.strip():
        values = list(map(float, line.split()))
        if len(values) >= 4:
            step_data.append(values[:4])  # x, y, z, mass

if step_data:
    data.append(np.array(step_data))

print(f"Loaded {len(data)} time steps")

# Create simple visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Initial state
axes[0].scatter(data[0][:, 0], data[0][:, 1], s=1, alpha=0.5)
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].set_title('Initial State')
axes[0].grid(True, alpha=0.3)
axes[0].axis('equal')

# Middle state
mid_idx = len(data) // 2
axes[1].scatter(data[mid_idx][:, 0], data[mid_idx][:, 1], s=1, alpha=0.5)
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
axes[1].set_title(f'Middle State (Step {mid_idx})')
axes[1].grid(True, alpha=0.3)
axes[1].axis('equal')

# Final state
axes[2].scatter(data[-1][:, 0], data[-1][:, 1], s=1, alpha=0.5)
axes[2].set_xlabel('X')
axes[2].set_ylabel('Y')
axes[2].set_title(f'Final State (Step {len(data)-1})')
axes[2].grid(True, alpha=0.3)
axes[2].axis('equal')

plt.tight_layout()
plt.savefig('barnes_hut_results.png', dpi=150)
print("Saved: barnes_hut_results.png")
plt.show()

# Create animation
print("\nCreating animation...")
fig_anim, ax_anim = plt.subplots(figsize=(8, 8))
scat = ax_anim.scatter([], [], s=1, alpha=0.5)
ax_anim.set_xlim(-100, 100)
ax_anim.set_ylim(-100, 100)
ax_anim.set_xlabel('X')
ax_anim.set_ylabel('Y')
ax_anim.set_title('Barnes-Hut N-Body Simulation')

def update(frame):
    scat.set_offsets(data[frame][:, :2])
    ax_anim.set_title(f'Barnes-Hut Simulation - Step {frame}')
    return scat,

anim = animation.FuncAnimation(fig_anim, update, frames=len(data), 
                               interval=50, blit=False, repeat=True)

anim.save('barnes_hut_animation.gif', writer='pillow', fps=20)
print("Saved: barnes_hut_animation.gif")
print("\nVisualization complete!")