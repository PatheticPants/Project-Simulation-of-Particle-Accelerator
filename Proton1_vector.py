import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class VectorFieldVisualization:
    def __init__(self, Q, D):
        self.k = 8.99e9
        self.Q = Q
        self.D = D
        
    def electric_field(self, x, y):
        r1_squared = x**2 + y**2
        r1 = np.sqrt(r1_squared)
        Ex1 = self.k * self.Q * x / (r1_squared * r1)
        Ey1 = self.k * self.Q * y / (r1_squared * r1)
        
        x_shifted = x - self.D
        r2_squared = x_shifted**2 + y**2
        r2 = np.sqrt(r2_squared)
        Ex2 = -self.k * self.Q * x_shifted / (r2_squared * r2)
        Ey2 = -self.k * self.Q * y / (r2_squared * r2)
        
        return Ex1 + Ex2, Ey1 + Ey2
    
    def plot_vector_field(self, grid_points=20, plot_size=None):
        if plot_size is None:
            plot_size = self.D * 1.5
            
        x = np.linspace(-plot_size, plot_size + self.D, grid_points)
        y = np.linspace(-plot_size, plot_size, grid_points)
        X, Y = np.meshgrid(x, y)
        
        Ex, Ey = self.electric_field(X, Y)
        E_magnitude = np.sqrt(Ex**2 + Ey**2)
        Ex_norm = Ex / E_magnitude
        Ey_norm = Ey / E_magnitude
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        quiver = ax.quiver(X, Y, Ex_norm, Ey_norm, E_magnitude,
                          cmap='viridis', scale=25, width=0.003)
        
        positive = Circle((0, 0), plot_size/20, color='red', label='Positive charge')
        negative = Circle((self.D, 0), plot_size/20, color='blue', label='Negative charge')
        ax.add_patch(positive)
        ax.add_patch(negative)
        
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title('Electric Field Vector Plot')
        ax.legend()
        
        plt.colorbar(quiver, label='Field strength (V/m)')
        plt.tight_layout()
        
        return fig, ax

if __name__ == "__main__":
    Q = 1e-6
    D = 0.6
    viz = VectorFieldVisualization(Q, D)
    fig, ax = viz.plot_vector_field(grid_points=25)
    plt.show()