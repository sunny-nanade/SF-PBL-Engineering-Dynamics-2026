import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import matplotlib.patches as patches

"""
CONVEYOR BELT SIMULATION - MATHEMATICAL MODEL & HYPOTHESES
============================================================

HYPOTHESES:
1. The block is modeled as a point mass with friction contact
2. Conveyor belt moves at constant velocity V_belt
3. Coulomb friction model (friction independent of velocity magnitude)
4. No vertical motion or rotation (2D simulation)
5. Belt surface is perfectly rigid and flat
6. Air resistance and rolling resistance are neglected
7. Friction coefficient μ remains constant
8. No slip-stick oscillations (pure kinetic friction)

MATHEMATICAL MODEL:
------------------
Equation of motion: m * dv/dt = μ * m * g * sign(V_belt - v)

Simplified: dv/dt = μ * g * sign(V_belt - v)

State space:
dx/dt = v
dv/dt = μ * g * sign(V_belt - v)

Analytical solution: v(t) = V_belt - (V_belt - v₀) * exp(-μ * g * t / V_belt)

Characteristic time: τ = V_belt / (μ * g)
"""

class ConveyorBeltSimulation:
    def __init__(self, mu=0.3, V_belt=1.0, m=1.0, g=9.81):
        self.mu = mu          # Friction coefficient (hypothesis: constant)
        self.V_belt = V_belt  # Belt velocity (hypothesis: constant speed)
        self.m = m            # Mass (kg) - cancels out in acceleration
        self.g = g            # Gravity (m/s²)
        self.state = np.array([0.0, 0.0])  # [position, velocity]
        
    def equations_of_motion(self, t, state):
        """Differential equations based on Coulomb friction model"""
        x, v = state
        v_rel = self.V_belt - v
        
        # Coulomb friction model: force independent of velocity magnitude
        if abs(v_rel) < 1e-6:
            acceleration = 0.0  # Steady state reached
        else:
            acceleration = self.mu * self.g * np.sign(v_rel)
            
        return [v, acceleration]
    
    def simulate(self, t_span=(0, 10), t_eval=None):
        """Run numerical simulation using Runge-Kutta method"""
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 1000)
            
        solution = solve_ivp(
            self.equations_of_motion,
            t_span,
            self.state,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-6
        )
        
        return {'t': solution.t, 'x': solution.y[0], 'v': solution.y[1]}
    
    def analytical_velocity(self, t, v0=0):
        """Analytical solution for validation"""
        return self.V_belt - (self.V_belt - v0) * np.exp(-self.mu * self.g * t / max(abs(self.V_belt), 0.01))

class ConveyorBeltVisualization:
    def __init__(self, simulation, belt_length=10):
        self.sim = simulation
        self.belt_length = belt_length
        
        # Create figure with three subplots
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.suptitle('CONVEYOR BELT SIMULATION - Block Dynamics with Coulomb Friction', fontsize=14, fontweight='bold')
        
        # Subplot 1: Animation
        self.ax1 = self.fig.add_subplot(3, 1, 1)
        
        # Subplot 2: Velocity and Position over time
        self.ax2 = self.fig.add_subplot(3, 1, 2)
        
        # Subplot 3: Phase portrait and hypotheses
        self.ax3 = self.fig.add_subplot(3, 1, 3)
        
        self.setup_animation_plot()
        self.setup_time_plot()
        self.setup_phase_plot()
        
    def setup_animation_plot(self):
        """Setup the animation subplot"""
        self.ax1.set_xlim(-1, self.belt_length + 1)
        self.ax1.set_ylim(-1, 2.5)
        self.ax1.set_xlabel('Position (m)', fontsize=10)
        self.ax1.set_ylabel('Height (m)', fontsize=10)
        self.ax1.set_title('Block Motion Animation', fontsize=11, fontweight='bold')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_aspect('equal')
        
        # Draw conveyor belt structure
        belt_rect = patches.Rectangle((0, -0.15), self.belt_length, 0.3, 
                                       facecolor='#8B7355', edgecolor='black', alpha=0.7)
        self.ax1.add_patch(belt_rect)
        
        # Draw belt surface line
        self.ax1.axhline(y=0, color='#654321', linewidth=2)
        
        # Draw rollers
        for x in [0, self.belt_length]:
            roller = patches.Circle((x, 0), 0.2, facecolor='#555555', edgecolor='black', linewidth=2)
            self.ax1.add_patch(roller)
        
        # Draw block
        self.block = patches.Rectangle((-0.4, -0.1), 0.8, 0.4, 
                                        facecolor='#4169E1', edgecolor='black', linewidth=2, alpha=0.9)
        self.ax1.add_patch(self.block)
        
        # Add block label
        self.block_text = self.ax1.text(0, 0.3, 'Block', ha='center', fontsize=8, fontweight='bold')
        
        # Velocity arrow
        self.velocity_arrow = None
        
        # Belt velocity indicator
        self.ax1.arrow(self.belt_length/2, -0.4, self.sim.V_belt * 0.5, 0, 
                       head_width=0.15, head_length=0.3, fc='green', ec='green', alpha=0.8)
        self.ax1.text(self.belt_length/2, -0.6, f'Belt Velocity: {self.sim.V_belt:.1f} m/s →', 
                      ha='center', fontsize=10, color='green', fontweight='bold')
        
    def setup_time_plot(self):
        """Setup velocity vs time and position vs time plot"""
        self.ax2.set_xlabel('Time (s)', fontsize=10)
        self.ax2.set_ylabel('Velocity (m/s)', fontsize=10, color='blue')
        self.ax2.set_title('Velocity and Position vs Time', fontsize=11, fontweight='bold')
        self.ax2.grid(True, alpha=0.3)
        
        self.velocity_line, = self.ax2.plot([], [], 'b-', linewidth=2, label='Block Velocity')
        self.ax2.axhline(y=self.sim.V_belt, color='green', linestyle='--', 
                         linewidth=1.5, alpha=0.7, label=f'Belt Velocity ({self.sim.V_belt} m/s)')
        self.ax2.set_ylim(-0.5, max(self.sim.V_belt + 1, 2))
        self.ax2.legend(loc='upper left')
        
        # Position on secondary y-axis
        self.ax2_pos = self.ax2.twinx()
        self.ax2_pos.set_ylabel('Position (m)', fontsize=10, color='orange')
        self.position_line, = self.ax2_pos.plot([], [], 'orange', linewidth=2, label='Block Position')
        self.ax2_pos.legend(loc='upper right')
        
    def setup_phase_plot(self):
        """Setup phase portrait with hypotheses text"""
        self.ax3.set_xlabel('Velocity (m/s)', fontsize=10)
        self.ax3.set_ylabel('Acceleration (m/s²)', fontsize=10)
        self.ax3.set_title('Phase Portrait & Model Hypotheses', fontsize=11, fontweight='bold')
        self.ax3.grid(True, alpha=0.3)
        
        self.phase_line, = self.ax3.plot([], [], 'purple', linewidth=2, label='Trajectory')
        self.phase_point, = self.ax3.plot([], [], 'ro', markersize=8, label='Current State')
        
        # Add hypotheses text box
        hypotheses_text = """HYPOTHESES:
1. Point mass model (no rotation)
2. Constant belt velocity
3. Coulomb friction (μ constant)
4. Rigid surfaces
5. No air resistance
6. Dry friction only

MODEL: dv/dt = μ·g·sign(V_belt - v)"""
        
        self.ax3.text(0.02, 0.98, hypotheses_text, transform=self.ax3.transAxes,
                     fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round', 
                     facecolor='wheat', alpha=0.5))
        self.ax3.legend(loc='lower right')
        self.ax3.set_xlim(-1, self.sim.V_belt + 1)
        self.ax3.set_ylim(-12, 12)
        
    def update(self, frame, trajectory):
        """Update animation for each frame"""
        t = trajectory['t'][frame]
        x = trajectory['x'][frame]
        v = trajectory['v'][frame]
        
        # Update block position
        self.block.set_x(x - 0.4)
        self.block_text.set_position((x, 0.35))
        
        # Update velocity arrow
        if self.velocity_arrow:
            self.velocity_arrow.remove()
        
        arrow_length = np.clip(v * 0.3, -1, 1)
        if abs(v) > 0.01:
            self.velocity_arrow = self.ax1.arrow(x, 0.25, arrow_length, 0,
                                                   head_width=0.1, head_length=0.15, 
                                                   fc='red', ec='red', alpha=0.8)
            self.ax1.text(x + arrow_length/2, 0.45, f'{v:.2f} m/s', 
                         ha='center', fontsize=8, color='red')
        
        # Update time plot
        idx = frame + 1
        self.velocity_line.set_data(trajectory['t'][:idx], trajectory['v'][:idx])
        self.position_line.set_data(trajectory['t'][:idx], trajectory['x'][:idx])
        
        # Update phase plot
        self.phase_line.set_data(trajectory['v'][:idx], 
                                 self.sim.mu * self.sim.g * np.sign(self.sim.V_belt - trajectory['v'][:idx]))
        self.phase_point.set_data([v], [self.sim.mu * self.sim.g * np.sign(self.sim.V_belt - v)])
        
        # Update titles with current info
        self.ax1.set_title(f'Time: {t:.2f}s | Position: {x:.2f}m | Velocity: {v:.2f}m/s', 
                          fontsize=10, fontweight='bold')
        
        # Rescale axes dynamically
        self.ax2.set_xlim(0, max(trajectory['t'][:idx]) + 0.5)
        self.ax2_pos.set_xlim(0, max(trajectory['t'][:idx]) + 0.5)
        
        return [self.block, self.velocity_line, self.position_line, self.phase_line, self.phase_point]
    
    def animate(self, trajectory, interval=30):
        """Run the animation"""
        anim = FuncAnimation(self.fig, self.update, fargs=(trajectory,),
                            frames=len(trajectory['t']), interval=interval, 
                            blit=False, repeat=True)
        plt.tight_layout()
        plt.show()
        return anim

def print_simulation_info(sim):
    """Print simulation information and results"""
    print("\n" + "="*70)
    print("CONVEYOR BELT SIMULATION - MATHEMATICAL MODELING")
    print("="*70)
    
    print("\n📐 MODEL HYPOTHESES:")
    print("-" * 40)
    print("1. ✓ Block treated as point mass (no rotational inertia)")
    print("2. ✓ Belt moves at constant velocity")
    print("3. ✓ Coulomb friction model (μ constant)")
    print("4. ✓ Rigid surfaces (no deformation)")
    print("5. ✓ Negligible air resistance")
    print("6. ✓ Dry friction only (no lubrication)")
    
    print("\n📊 MATHEMATICAL EQUATIONS:")
    print("-" * 40)
    print("Equation of motion:  m·dv/dt = μ·m·g·sign(V_belt - v)")
    print("Simplified:          dv/dt = μ·g·sign(V_belt - v)")
    print("State space:         dx/dt = v,  dv/dt = μ·g·sign(V_belt - v)")
    print(f"\nParameters:")
    print(f"  • Friction coefficient (μ) = {sim.mu}")
    print(f"  • Belt velocity (V_belt) = {sim.V_belt} m/s")
    print(f"  • Gravity (g) = {sim.g} m/s²")
    print(f"  • Mass (m) = {sim.m} kg (cancels out in acceleration)")
    
    print("\n⏰ CHARACTERISTIC TIMES:")
    print("-" * 40)
    tau = max(abs(sim.V_belt), 0.01) / (sim.mu * sim.g)
    print(f"Characteristic time (τ) = V_belt/(μ·g) = {tau:.3f} seconds")
    print("The block approaches belt velocity exponentially with this time constant")
    
    print("\n🎯 KEY PREDICTIONS:")
    print("-" * 40)
    print("• Block accelerates until reaching belt velocity")
    print("• No overshoot (friction always opposes relative motion)")
    print("• Steady-state reached asymptotically")
    print("• Energy dissipated as heat through friction")
    
    return tau

def main():
    """Main function to run the simulation"""
    # Simulation parameters
    MU = 0.3          # Friction coefficient (rubber on metal)
    V_BELT = 1.5      # Belt velocity (m/s)
    MASS = 1.0        # Mass (kg)
    GRAVITY = 9.81    # Gravity (m/s²)
    DURATION = 8.0    # Simulation duration (seconds)
    
    # Create simulation
    sim = ConveyorBeltSimulation(mu=MU, V_belt=V_BELT, m=MASS, g=GRAVITY)
    
    # Print simulation information and get tau
    tau = print_simulation_info(sim)
    
    # Run simulation
    print("\n🔄 Running numerical simulation...")
    trajectory = sim.simulate(t_span=(0, DURATION))
    
    # Display results
    print("\n📈 SIMULATION RESULTS:")
    print("-" * 40)
    print(f"Final time: {trajectory['t'][-1]:.2f} s")
    print(f"Final position: {trajectory['x'][-1]:.2f} m")
    print(f"Final velocity: {trajectory['v'][-1]:.2f} m/s")
    print(f"Belt velocity: {sim.V_belt:.2f} m/s")
    print(f"Velocity error: {abs(trajectory['v'][-1] - sim.V_belt):.4f} m/s")
    
    # Analytical validation
    v_analytical = sim.analytical_velocity(trajectory['t'])
    max_error = np.max(np.abs(trajectory['v'] - v_analytical))
    print(f"Numerical vs analytical error: {max_error:.6f} m/s")
    
    # Check steady state
    steady_state_idx = np.argmin(np.abs(trajectory['v'] - sim.V_belt))
    steady_state_time = trajectory['t'][steady_state_idx]
    print(f"Time to approach belt velocity: {steady_state_time:.2f} s (≈ {steady_state_time/tau:.1f}τ)")
    
    # Create visualization
    print("\n🎬 Starting animation... Close the plot window to exit.")
    viz = ConveyorBeltVisualization(sim, belt_length=12)
    viz.animate(trajectory, interval=30)
    
    print("\n✅ Simulation complete!")

if __name__ == "__main__":
    main()