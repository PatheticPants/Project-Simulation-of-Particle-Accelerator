import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import odeint
from matplotlib import gridspec

class ProtonSimulation:
    def __init__(self, Q, D, d0, theta0):
        self.k = 8.99e9
        self.q = 1.60e-19
        self.m = 1.67e-27
        self.Q = Q
        self.D = D
        self.d0 = d0
        self.theta0 = np.radians(theta0)
        self.x0 = d0 * np.cos(self.theta0)
        self.y0 = d0 * np.sin(self.theta0)
        self.vx0 = self.vy0 = 0
    
    def electric_field(self, x, y):
        r1 = np.sqrt(x**2 + y**2)
        Ex1 = self.k * self.Q * x / (r1**3)
        Ey1 = self.k * self.Q * y / (r1**3)
        
        r2 = np.sqrt((x - self.D)**2 + y**2)
        Ex2 = -self.k * self.Q * (x - self.D) / (r2**3)
        Ey2 = -self.k * self.Q * y / (r2**3)
        
        return Ex1 + Ex2, Ey1 + Ey2
    
    def derivatives(self, state, t):
        x, y, vx, vy = state
        distance_to_negative = np.sqrt((x - self.D)**2 + y**2)
        if distance_to_negative < 0.0001:
            return [0, 0, 0, 0]
            
        Ex, Ey = self.electric_field(x, y)
        ax = (self.q * Ex) / self.m
        ay = (self.q * Ey) / self.m
        return [vx, vy, ax, ay]
    
    def simulate(self, t_max, num_points=1000):
        t = np.linspace(0, t_max, num_points)
        initial_state = [self.x0, self.y0, self.vx0, self.vy0]
        solution = odeint(self.derivatives, initial_state, t)
        
        x = solution[:, 0]
        y = solution[:, 1]
        vx = solution[:, 2]
        vy = solution[:, 3]
        
        ax = np.gradient(vx, t)
        ay = np.gradient(vy, t)
        
        distance = np.sqrt(x**2 + y**2)
        angle = np.degrees(np.arctan2(y, x))
        
        Ex0, Ey0 = self.electric_field(self.x0, self.y0)
        self.initial_force = abs(self.q * np.sqrt(Ex0**2 + Ey0**2))
        
        return t, distance, angle, x, y, vx, vy, ax, ay
    
    def create_info_text(self):
        return (f'System Parameters:\n'
                f'Fixed charge (Q): {self.Q:.2e} C\n'
                f'Charge separation (D): {self.D:.2e} m\n'
                f'Initial distance (d0): {self.d0:.2e} m\n'
                f'Initial angle: {np.degrees(self.theta0):.1f}°\n\n'
                f'Initial Force: {self.initial_force:.2e} N')

    def visualize(self, t, distance, angle, x, y, vx, vy, ax, ay, interval=50):
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 5, height_ratios=[2, 1])
        
        ax_traj = fig.add_subplot(gs[0, :])
        ax_dist = fig.add_subplot(gs[1, 0])
        ax_angle = fig.add_subplot(gs[1, 1])
        ax_vel = fig.add_subplot(gs[1, 2])
        ax_accel = fig.add_subplot(gs[1, 3])
        ax_info = fig.add_subplot(gs[1, 4])
        
        ax_traj.plot(0, 0, 'b+', markersize=10, label='Positive charge')
        ax_traj.plot(self.D, 0, 'r+', markersize=10, label='Negative charge')
        
        trajectory, = ax_traj.plot([], [], 'g-', alpha=0.5)
        proton = ax_traj.scatter([x[0]], [y[0]], c='g', s=50, label='Proton')
        
        max_x = max(max(abs(x)), self.D)
        max_y = max(abs(y))
        padding = 0.1 * max(max_x, max_y)
        ax_traj.set_xlim(min(min(x), 0) - padding, max(max_x, self.D) + padding)
        ax_traj.set_ylim(-max_y - padding, max_y + padding)
        ax_traj.set_xlabel('X position (m)')
        ax_traj.set_ylabel('Y position (m)')
        ax_traj.set_title('Proton Motion in Electric Field')
        ax_traj.grid(True)
        ax_traj.legend()
        
        ax_dist.plot(t, distance, 'b-')
        ax_dist.set_xlabel('Time (s)')
        ax_dist.set_ylabel('Distance (m)')
        ax_dist.grid(True)
        
        ax_angle.plot(t, angle, 'r-')
        ax_angle.set_xlabel('Time (s)')
        ax_angle.set_ylabel('Angle (degrees)')
        ax_angle.grid(True)
        
        velocity = np.sqrt(vx**2 + vy**2)
        ax_vel.plot(t, velocity, 'g-')
        ax_vel.set_xlabel('Time (s)')
        ax_vel.set_ylabel('|Velocity| (m/s)')
        ax_vel.grid(True)
        
        acceleration = np.sqrt(ax**2 + ay**2)
        ax_accel.plot(t, acceleration, 'm-')
        ax_accel.set_xlabel('Time (s)')
        ax_accel.set_ylabel('|Acceleration| (m/s²)')
        ax_accel.grid(True)
        
        time_line_dist, = ax_dist.plot([t[0], t[0]], ax_dist.get_ylim(), 'k-', linewidth=2)
        time_line_angle, = ax_angle.plot([t[0], t[0]], ax_angle.get_ylim(), 'k-', linewidth=2)
        time_line_vel, = ax_vel.plot([t[0], t[0]], ax_vel.get_ylim(), 'k-', linewidth=2)
        time_line_accel, = ax_accel.plot([t[0], t[0]], ax_accel.get_ylim(), 'k-', linewidth=2)
        
        ax_info.axis('off')
        ax_info.text(0.5, 0.5, self.create_info_text(),
                    fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='black', alpha=0.7),
                    ha='center',
                    va='center')

        def update(frame):
            trajectory.set_data(x[:frame+1], y[:frame+1])
            proton.set_offsets(np.column_stack((x[frame], y[frame])))
            current_time = t[frame]
            time_line_dist.set_xdata([current_time, current_time])
            time_line_angle.set_xdata([current_time, current_time])
            time_line_vel.set_xdata([current_time, current_time])
            time_line_accel.set_xdata([current_time, current_time])
            return trajectory, proton, time_line_dist, time_line_angle, time_line_vel, time_line_accel
        
        anim = animation.FuncAnimation(
            fig, update, frames=len(t),
            interval=interval, blit=True
        )
        
        plt.tight_layout()
        plt.show()
        return anim

if __name__ == "__main__":
    Q = 1e-6
    D = 0.6
    d0 = 0.3
    theta0 = 18
    
    sim = ProtonSimulation(Q, D, d0, theta0)
    t, distance, angle, x, y, vx, vy, ax, ay = sim.simulate(t_max=1e-6)
    anim = sim.visualize(t, distance, angle, x, y, vx, vy, ax, ay, interval=30)