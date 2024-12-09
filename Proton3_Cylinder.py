import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as mtick
from scipy.integrate import odeint, quad
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec

class CylinderProtonSimulation:
    def __init__(self, z0, radius, length, charge_density, mass=1.67e-27):
        self.k = 8.99e9
        self.m = mass
        self.z0 = z0
        self.radius = radius
        self.length = length
        self.charge_density = charge_density
        self.x0 = self.y0 = self.vx0 = self.vy0 = self.vz0 = 0
    
    def electric_field(self, x, y, z):
        if abs(x) < 1e-10 and abs(y) < 1e-10:
            def integrand(z_prime):
                R = self.radius
                dz = z - z_prime
                return R * dz / (R**2 + dz**2)**(3/2)
            
            z_min, z_max = -self.length/2, self.length/2
            Ez, _ = quad(integrand, z_min, z_max)
            Ez *= 2 * np.pi * self.radius * self.charge_density * self.k
            return 0, 0, Ez
        
        else:
            theta_points, z_points = 20, 20
            theta = np.linspace(0, 2*np.pi, theta_points)
            z_prime = np.linspace(-self.length/2, self.length/2, z_points)
            dtheta = 2*np.pi/theta_points
            dz = self.length/z_points
            
            Ex = Ey = Ez = 0
            for t in theta:
                for zp in z_prime:
                    xs = self.radius * np.cos(t)
                    ys = self.radius * np.sin(t)
                    dx, dy, dz = x - xs, y - ys, z - zp
                    r3 = (dx**2 + dy**2 + dz**2)**(3/2)
                    dq = self.charge_density * self.radius * dtheta * dz
                    Ex += self.k * dq * dx / r3
                    Ey += self.k * dq * dy / r3
                    Ez += self.k * dq * dz / r3
            return Ex, Ey, Ez

    def derivatives(self, state, t):
        x, y, z, vx, vy, vz = state
        Ex, Ey, Ez = self.electric_field(x, y, z)
        q = 1.60e-19
        ax = (q * Ex) / self.m
        ay = (q * Ey) / self.m
        az = (q * Ez) / self.m
        return [vx, vy, vz, ax, ay, az]

    def simulate(self, t_max, num_points=1000):
        t = np.linspace(0, t_max, num_points)
        initial_state = [self.x0, self.y0, self.z0, self.vx0, self.vy0, self.vz0]
        
        Ex, Ey, Ez = self.electric_field(self.x0, self.y0, self.z0)
        q = 1.60e-19
        self.initial_force = abs(q * np.sqrt(Ex**2 + Ey**2 + Ez**2))
        
        solution = odeint(self.derivatives, initial_state, t)
        x, y, z = solution[:, 0], solution[:, 1], solution[:, 2]
        vx, vy, vz = solution[:, 3], solution[:, 4], solution[:, 5]
        
        ax = np.gradient(vx, t)
        ay = np.gradient(vy, t)
        az = np.gradient(vz, t)
        
        return t, x, y, z, vx, vy, vz, ax, ay, az

    def create_info_text(self):
        return (f'Initial Conditions:\n'
                f'Z position: {self.z0:.2e} m\n'
                f'Initial velocity: {self.vz0:.2e} m/s\n'
                f'Proton mass: {self.m:.2e} kg\n\n'
                f'Cylinder Parameters:\n'
                f'Radius: {self.radius:.2e} m\n'
                f'Length: {self.length:.2e} m\n'
                f'Charge density: {self.charge_density:.2e} C/m²\n\n'
                f'Initial Force: {self.initial_force:.2e} N')

    def visualize(self, t, x, y, z, vx, vy, vz, ax, ay, az, interval=50):
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 4, height_ratios=[2, 1])
        ax_3d = fig.add_subplot(gs[0, :], projection='3d')
        ax_pos = fig.add_subplot(gs[1, 0])
        ax_vel = fig.add_subplot(gs[1, 1])
        ax_acc = fig.add_subplot(gs[1, 2])
        ax_info = fig.add_subplot(gs[1, 3])

        # Plot cylinder with reduced resolution
        theta = np.linspace(0, 2*np.pi, 10)
        z_cyl = np.linspace(-self.length/2, self.length/2, 8)
        theta, z_cyl = np.meshgrid(theta, z_cyl)
        x_cyl = self.radius * np.cos(theta)
        y_cyl = self.radius * np.sin(theta)
        ax_3d.plot_surface(x_cyl, y_cyl, z_cyl, alpha=0.2, color='r')

        # Initialize trajectory and proton
        trajectory, = ax_3d.plot([], [], [], 'b-', label='Trajectory')
        proton = ax_3d.scatter([x[0]], [y[0]], [z[0]], color='r', s=100, label='Proton')
        ax_3d.scatter(0, 0, self.z0, color='g', marker='o', label='Initial position')

        max_val = max(max(abs(x)), max(abs(y)), max(abs(z)), self.radius)
        ax_3d.set_xlim([-max_val, max_val])
        ax_3d.set_ylim([-max_val, max_val])
        ax_3d.set_zlim([min(z)-0.1, max(z)+0.1])
        ax_3d.set_xlabel('X (m)')
        ax_3d.set_ylabel('Y (m)')
        ax_3d.set_zlabel('Z (m)')
        ax_3d.legend()

        # Plot time series data
        ax_pos.plot(t, z, 'b-')
        ax_pos.set_xlabel('Time (s)')
        ax_pos.set_ylabel('Position (m)')
        ax_pos.grid(True)

        v_total = np.sqrt(vx**2 + vy**2 + vz**2)
        ax_vel.plot(t, v_total, 'g-')
        ax_vel.yaxis.set_major_formatter(mtick.ScalarFormatter())
        ax_vel.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax_vel.set_xlabel('Time (s)')
        ax_vel.set_ylabel('|Velocity|(m/s)')
        ax_vel.grid(True)

        a_total = np.sqrt(ax**2 + ay**2 + az**2)
        ax_acc.plot(t, a_total, 'r-')
        ax_acc.set_xlabel('Time (s)')
        ax_acc.set_ylabel('|Acceleration| (m/s²)')
        ax_acc.grid(True)

        # Add info text
        ax_info.axis('off')
        ax_info.text(0.5, 0.5, self.create_info_text(), 
                    fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='black', alpha=0.7),
                    ha='center',
                    va='center')

        # Add time indicator lines
        time_line_pos, = ax_pos.plot([], [], 'k-', linewidth=2)
        time_line_vel, = ax_vel.plot([], [], 'k-', linewidth=2)
        time_line_acc, = ax_acc.plot([], [], 'k-', linewidth=2)

        def init():
            trajectory.set_data([], [])
            trajectory.set_3d_properties([])
            time_line_pos.set_data([], [])
            time_line_vel.set_data([], [])
            time_line_acc.set_data([], [])
            return trajectory, proton, time_line_pos, time_line_vel, time_line_acc

        def update(frame):
            trajectory.set_data(x[:frame], y[:frame])
            trajectory.set_3d_properties(z[:frame])
            proton._offsets3d = (np.array([x[frame]]), np.array([y[frame]]), np.array([z[frame]]))
            current_time = t[frame]
            time_line_pos.set_data([current_time, current_time], ax_pos.get_ylim())
            time_line_vel.set_data([current_time, current_time], ax_vel.get_ylim())
            time_line_acc.set_data([current_time, current_time], ax_acc.get_ylim())
            return trajectory, proton, time_line_pos, time_line_vel, time_line_acc

        anim = animation.FuncAnimation(
            fig, update, frames=len(t),
            init_func=init, interval=interval, blit=False
        )

        plt.tight_layout()
        plt.show()
        return anim

if __name__ == "__main__":
    z0 = 0.02
    radius = 0.05
    length = 0.1
    charge_density = 1e-6
    proton_mass = 1.67e-27
    
    sim = CylinderProtonSimulation(z0, radius, length, charge_density, proton_mass)
    t, x, y, z, vx, vy, vz, ax, ay, az = sim.simulate(t_max=2e-6)
    anim = sim.visualize(t, x, y, z, vx, vy, vz, ax, ay, az, interval=50)