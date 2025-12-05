import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

print("BARNES-HUT SIMULATION VISUALIZATION (COMPLETE VERSION)")
print("=" * 50)

# ==========================================
# CONFIGURATION
# ==========================================
# Colors
BG_COLOR = 'black'
STAR_COLOR = '#FFD700' 
TEXT_COLOR = 'white'

# Speed Controls
ANIMATION_INTERVAL = 200 
SAVE_FPS = 60
# ==========================================

# 1. Read data
print("Reading data...")
data = []
try:
    with open('optimized_output.txt', 'r') as f:
        lines = f.readlines()

    step_data = []
    for line in lines:
        if line.startswith('Step'):
            if step_data:
                data.append(np.array(step_data))
                step_data = []
        elif line.strip():
            try:
                values = list(map(float, line.split()))
                if len(values) >= 4:
                    step_data.append(values[:4])  # x, y, z, mass
            except ValueError:
                continue

    if step_data:
        data.append(np.array(step_data))
    print(f"Loaded {len(data)} time steps")

except FileNotFoundError:
    print("Error: 'optimized_output.txt' not found. Run the simulation first.")
    exit()

if not data:
    print("Error: No data loaded.")
    exit()

# Helper function to style plots for dark mode
def style_dark_plot(ax, title_text):
    ax.set_facecolor(BG_COLOR)
    ax.set_title(title_text, color=TEXT_COLOR)
    ax.set_xlabel('X', color=TEXT_COLOR)
    ax.set_ylabel('Y', color=TEXT_COLOR)
    ax.tick_params(axis='x', colors=TEXT_COLOR)
    ax.tick_params(axis='y', colors=TEXT_COLOR)
    ax.grid(True, color='gray', alpha=0.2)
    for spine in ax.spines.values():
        spine.set_edgecolor(TEXT_COLOR)
    ax.axis('equal')

# ==========================================
# PART 1: STATIC SNAPSHOTS
# ==========================================
print("\nGenerating static snapshots...")
fig_static, axes = plt.subplots(1, 3, figsize=(15, 5))
fig_static.patch.set_facecolor(BG_COLOR)

# Initial state
axes[0].scatter(data[0][:, 0], data[0][:, 1], s=1, color=STAR_COLOR, alpha=0.6)
style_dark_plot(axes[0], 'Initial State')

# Middle state
mid_idx = len(data) // 2
axes[1].scatter(data[mid_idx][:, 0], data[mid_idx][:, 1], s=1, color=STAR_COLOR, alpha=0.6)
style_dark_plot(axes[1], f'Middle State (Step {mid_idx})')

# Final state
axes[2].scatter(data[-1][:, 0], data[-1][:, 1], s=1, color=STAR_COLOR, alpha=0.6)
style_dark_plot(axes[2], f'Final State (Step {len(data)-1})')

plt.tight_layout()
plt.savefig('barnes_hut_results_dark.png', dpi=150, facecolor=BG_COLOR)
print("Saved image: barnes_hut_results_dark.png")

# ==========================================
# PART 2: ANIMATION
# ==========================================
print("\nPreparing animation...")
fig_anim, ax_anim = plt.subplots(figsize=(8, 8))
fig_anim.patch.set_facecolor(BG_COLOR)

# Set bounds
all_pos = np.vstack(data)
max_val = np.max(np.abs(all_pos[:, :2])) * 1.2
ax_anim.set_xlim(-max_val, max_val)
ax_anim.set_ylim(-max_val, max_val)

# Style the animation plot
style_dark_plot(ax_anim, "N-Body Simulation")
scat = ax_anim.scatter([], [], s=2, color=STAR_COLOR, alpha=0.8)

# Update function
def update(frame):
    current_data = data[frame]
    scat.set_offsets(current_data[:, :2])
    ax_anim.set_title(f'Time Step: {frame}', color=TEXT_COLOR)
    return scat,

# Create Animation
anim = animation.FuncAnimation(
    fig_anim, 
    update, 
    frames=len(data), 
    interval=ANIMATION_INTERVAL, 
    blit=False
)

# Save Animation
print(f"Saving animation with {SAVE_FPS} FPS...")
try:
    anim.save('barnes_hut_galaxy.gif', writer='pillow', fps=SAVE_FPS)
    print("Success! Saved as 'barnes_hut_galaxy.gif'")
except Exception as e:
    print(f"Could not save GIF: {e}")
    print("Showing live animation instead...")

plt.show()