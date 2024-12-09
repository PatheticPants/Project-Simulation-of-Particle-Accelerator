import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as mtick
from scipy.integrate import odeint, quad
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec

class CylinderChainSimulation:
    def __init__(self, z0, radius, lengths, charge_densities, spacings, mass=1.67e-27):
        self.k = 8.99e9
        self.m = mass
        self.z0 = z0
        self.radius = radius
        self.lengths = np.array(lengths)
        self.densities = np.array(charge_densities)
        self.spacings = np.array(spacings)
        self.cylinder_positions = self._calculate_cylinder_positions()
        self.x0 = self.y0 = self.vx0 = self.vy0 = self.vz0 = 0
        
    def _calculate_cylinder_positions(self):
        positions = np.zeros(len(self.lengths))
        current_pos = 0
        positions[0] = current_pos + self.lengths[0]/2
        current_pos += self.lengths[0]
        
        for i in range(1, len(self.lengths)):
            current_pos += self.spacings[i-1]
            positions[i] = current_pos + self.lengths[i]/2
            current_pos += self.lengths[i]
        return positions

    def electric_field_single_cylinder(self, x, y, z, center_z, length, density):
        z_shifted = z - center_z
        if abs(x) < 1e-10 and abs(y) < 1e-10:
            def integrand(z_prime):
                R = self.radius
                dz = z_shifted - z_prime
                return R * dz / (R**2 + dz**2)**(3/2)
            
            z_min, z_max = -length/2, length/2
            Ez, _ = quad(integrand, z_min, z_max)
            Ez *= 2 * np.pi * self.radius * density * self.k
            return 0, 0, Ez
        
        else:
            theta_points, z_points = 20, 20
            theta = np.linspace(0, 2*np.pi, theta_points)
            z_prime = np.linspace(-length/2, length/2, z_points)
            dtheta = 2*np.pi/theta_points
            dz = length/z_points
            
            Ex = Ey = Ez = 0
            for t in theta:
                for zp in z_prime:
                    xs = self.radius * np.cos(t)
                    ys = self.radius * np.sin(t)
                    dx, dy, dz = x - xs, y - ys, z_shifted - zp
                    r3 = (dx**2 + dy**2 + dz**2)**(3/2)
                    dq = density * self.radius * dtheta * dz
                    Ex += self.k * dq * dx / r3
                    Ey += self.k * dq * dy / r3
                    Ez += self.k * dq * dz / r3
            return Ex, Ey, Ez
    
    def electric_field(self, x, y, z):
        Ex_total = Ey_total = Ez_total = 0
        for i in range(len(self.lengths)):
            Ex, Ey, Ez = self.electric_field_single_cylinder(
                x, y, z,
                self.cylinder_positions[i],
                self.lengths[i],
                self.densities[i]
            )
            Ex_total += Ex
            Ey_total += Ey
            Ez_total += Ez
        return Ex_total, Ey_total, Ez_total

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
        lengths_str = '\n'.join([f'Length {i+1}: {l:.2e} m' for i, l in enumerate(self.lengths)])
        densities_str = '\n'.join([f'Density {i+1}: {d:.2e} C/m²' for i, d in enumerate(self.densities)])
        spacings_str = '\n'.join([f'Spacing {i+1}-{i+2}: {s:.2e} m' for i, s in enumerate(self.spacings)])
        
        return (f'Initial Conditions:\n'
                f'Z position: {self.z0:.2e} m\n'
                f'Initial velocity: {self.vz0:.2e} m/s\n'
                f'Proton mass: {self.m:.2e} kg\n\n'
                f'Cylinder Parameters:\n'
                f'Radius: {self.radius:.2e} m\n'
                f'{lengths_str}\n\n'
                f'Charge Densities:\n'
                f'{densities_str}\n\n'
                f'Cylinder Spacings:\n'
                f'{spacings_str}\n\n'
                f'Initial Force: {self.initial_force:.2e} N')

    def visualize(self, t, x, y, z, vx, vy, vz, ax, ay, az, interval=50):
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 4, height_ratios=[2, 1])
        ax_3d = fig.add_subplot(gs[0, :], projection='3d')
        ax_pos = fig.add_subplot(gs[1, 0])
        ax_vel = fig.add_subplot(gs[1, 1])
        ax_acc = fig.add_subplot(gs[1, 2])
        ax_info = fig.add_subplot(gs[1, 3])

        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.lengths)))
        for i, (center_z, length) in enumerate(zip(self.cylinder_positions, self.lengths)):
            theta = np.linspace(0, 2*np.pi, 10)
            z_cyl = np.linspace(center_z - length/2, center_z + length/2, 8)
            theta, z_cyl = np.meshgrid(theta, z_cyl)
            x_cyl = self.radius * np.cos(theta)
            y_cyl = self.radius * np.sin(theta)
            ax_3d.plot_surface(x_cyl, y_cyl, z_cyl, alpha=0.2, color=colors[i])

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

        ax_pos.plot(t, z, 'b-')
        ax_pos.set_xlabel('Time (s)')
        ax_pos.set_ylabel('Position (m)')
        ax_pos.grid(True)

        v_total = np.sqrt(vx**2 + vy**2 + vz**2)
        ax_vel.plot(t, v_total, 'g-')
        ax_vel.yaxis.set_major_formatter(mtick.ScalarFormatter())
        ax_vel.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax_vel.set_xlabel('Time (s)')
        ax_vel.set_ylabel('|Velocity| (m/s)')
        ax_vel.grid(True)

        a_total = np.sqrt(ax**2 + ay**2 + az**2)
        ax_acc.plot(t, a_total, 'r-')
        ax_acc.set_xlabel('Time (s)')
        ax_acc.set_ylabel('|Acceleration| (m/s²)')
        ax_acc.grid(True)

        ax_info.axis('off')
        ax_info.text(0.5, 0.5, self.create_info_text(), 
                    fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='black', alpha=0.7),
                    ha='center',
                    va='center')

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
    z0 = 0.43
    radius = 0.05
    lengths = [0.1, 0.05, 0.1]
    charge_densities = [5e-6, 1e-6, 5e-6]
    spacings = [0.2, 0.2]
    proton_mass = 1.67e-27
    
    sim = CylinderChainSimulation(z0, radius, lengths, charge_densities, spacings, proton_mass)
    t, x, y, z, vx, vy, vz, ax, ay, az = sim.simulate(t_max=10e-6)
    anim = sim.visualize(t, x, y, z, vx, vy, vz, ax, ay, az, interval=50)