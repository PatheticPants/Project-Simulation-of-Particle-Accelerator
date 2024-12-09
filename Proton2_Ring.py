import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as mtick
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec

class RingProtonSimulation:
    def __init__(self, z0, r, charge_density, mass=1.67e-27):
        self.k = 8.99e9
        self.m = mass
        self.z0 = z0
        self.r = r
        self.charge_density = charge_density
        self.total_charge = 2 * np.pi * r * charge_density
        self.x0 = self.y0 = self.vx0 = self.vy0 = self.vz0 = 0

    def electric_field(self, x, y, z):
        if abs(x) < 1e-10 and abs(y) < 1e-10:
            Ez = (self.k * self.total_charge * z) / \
                 (2 * np.pi * (self.r**2 + z**2)**(3/2))
            return 0, 0, Ez
            
        theta = np.linspace(0, 2*np.pi, 100)
        dx = self.r * np.cos(theta)[:, np.newaxis] - x
        dy = self.r * np.sin(theta)[:, np.newaxis] - y
        dz = np.zeros_like(dx) - z
        r_cubed = (dx**2 + dy**2 + dz**2)**(3/2)
        dq = self.charge_density * self.r * 2 * np.pi / len(theta)
        Ex = -np.sum(self.k * dq * dx / r_cubed)
        Ey = -np.sum(self.k * dq * dy / r_cubed)
        Ez = -np.sum(self.k * dq * dz / r_cubed)
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
        return (f'System Parameters:\n'
                f'Ring radius: {self.r:.2e} m\n'
                f'Charge density: {self.charge_density:.2e} C/m\n'
                f'Total charge: {self.total_charge:.2e} C\n'
                f'Initial z position: {self.z0:.2e} m\n'
                f'Proton mass: {self.m:.2e} kg\n\n'
                f'Initial Force: {self.initial_force:.2e} N')

    def visualize(self, t, x, y, z, vx, vy, vz, ax, ay, az, interval=50):
        stride = max(1, len(t) // 200)
        t = t[::stride]
        x, y, z = x[::stride], y[::stride], z[::stride]
        vx, vy, vz = vx[::stride], vy[::stride], vz[::stride]
        ax, ay, az = ax[::stride], ay[::stride], az[::stride]
        
        velocity = np.sqrt(vx**2 + vy**2 + vz**2)
        acceleration = np.sqrt(ax**2 + ay**2 + az**2)
        
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 4, height_ratios=[2, 1])
        
        ax_traj = fig.add_subplot(gs[0, :], projection='3d')
        ax_pos = fig.add_subplot(gs[1, 0])
        ax_vel = fig.add_subplot(gs[1, 1])
        ax_acc = fig.add_subplot(gs[1, 2])
        ax_info = fig.add_subplot(gs[1, 3])
        
        theta = np.linspace(0, 2*np.pi, 20)
        ring_x = self.r * np.cos(theta)
        ring_y = self.r * np.sin(theta)
        ring_z = np.zeros_like(theta)
        ax_traj.plot(ring_x, ring_y, ring_z, 'r-', linewidth=2, label='Charged ring')
        
        trajectory, = ax_traj.plot([], [], [], 'b-', label='Trajectory', lw=1)
        proton, = ax_traj.plot([], [], [], 'mo', markersize=8, label='Proton')
        
        max_val = max(max(abs(x)), max(abs(y)), max(abs(z)), self.r)
        ax_traj.set_xlim([-max_val*1.2, max_val*1.2])
        ax_traj.set_ylim([-max_val*1.2, max_val*1.2])
        ax_traj.set_zlim([min(min(z), -max_val), max(max(z), max_val)])
        ax_traj.set_xlabel('X (m)')
        ax_traj.set_ylabel('Y (m)')
        ax_traj.set_zlabel('Z (m)')
        ax_traj.legend()

        ax_pos.plot(t, z, 'b-')
        ax_pos.set_xlabel('Time (s)')
        ax_pos.set_ylabel('Position (m)')
        ax_pos.grid(True)
        
        ax_vel.plot(t, velocity, 'g-')
        ax_vel.yaxis.set_major_formatter(mtick.ScalarFormatter())
        ax_vel.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax_vel.set_xlabel('Time (s)')
        ax_vel.set_ylabel('|Velocity| (m/s)')
        ax_vel.grid(True)
        
        ax_acc.plot(t, acceleration, 'r-')
        ax_acc.set_xlabel('Time (s)')
        ax_acc.set_ylabel('|Acceleration| (m/s²)')
        ax_acc.grid(True)
        
        time_line_pos = ax_pos.axvline(x=t[0], color='k', ls='--')
        time_line_vel = ax_vel.axvline(x=t[0], color='k', ls='--')
        time_line_acc = ax_acc.axvline(x=t[0], color='k', ls='--')
        
        ax_info.axis('off')
        ax_info.text(0.05, 0.5, self.create_info_text(),
                    fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='black', alpha=0.7),
                    va='center')

        trajectory_data = np.array([x, y, z])
        
        def update(frame):
            trajectory.set_data(trajectory_data[0, :frame], trajectory_data[1, :frame])
            trajectory.set_3d_properties(trajectory_data[2, :frame])
            
            proton.set_data([x[frame]], [y[frame]])
            proton.set_3d_properties([z[frame]])
            
            time_line_pos.set_xdata([t[frame], t[frame]])
            time_line_vel.set_xdata([t[frame], t[frame]])
            time_line_acc.set_xdata([t[frame], t[frame]])
            
            if frame % 5 == 0:
                ax_traj.view_init(30, frame/2)
            
            return trajectory, proton, time_line_pos, time_line_vel, time_line_acc

        anim = animation.FuncAnimation(
            fig, update, frames=len(t),
            interval=interval, blit=True,
            cache_frame_data=False
        )

        plt.tight_layout()
        return anim

if __name__ == "__main__":
    z0 = 0.1
    ring_radius = 0.5
    charge_density = 1e-6
    proton_mass = 1.67e-26
    
    sim = RingProtonSimulation(z0, ring_radius, charge_density, proton_mass)
    t, x, y, z, vx, vy, vz, ax, ay, az = sim.simulate(t_max=10e-6)
    anim = sim.visualize(t, x, y, z, vx, vy, vz, ax, ay, az, interval=30)
    plt.show()