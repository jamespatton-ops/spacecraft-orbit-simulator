import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class SpacecraftSimulator:
    def __init__(self):
        # Constants
        self.mu_earth = 3.986e14  # Earth's gravitational parameter, m^3/s^2
        self.R_earth = 6371e3     # meters

        # Initial equinoctial orbital elements
        self.state = {
            'p': 667800.0,  # semi-latus rectum, meters
            'f': 0.001,     # eccentricity component f
            'g': 0.001,     # eccentricity component g (was missing)
            'h': 0.02,      # inclination component h
            'k': 0.01,      # inclination component k
            'L': 0.0,       # true longitude, radians
            'mass': 1000.0  # kg
        }

        # Engine parameters
        self.thrust = 1000.0  # Newtons
        self.isp = 300.0      # seconds
        self.g0 = 9.81        # m/s^2

        # Storage for plotting
        self.history = {key: [] for key in self.state}
        self.history['time'] = []

    def compute_auxiliary_variables(self):
        """Compute helper variables needed for the equinoctial derivative equations"""
        f = self.state['f']
        g = self.state['g']
        h = self.state['h']
        k = self.state['k']
        L = self.state['L']

        w = 1.0 + f * np.cos(L) + g * np.sin(L)
        s2 = 1.0 + h**2 + k**2

        return w, s2

    def propagate_orbit(self, alpha_r: float, alpha_t: float, alpha_n: float, dt: float):
        """Propagate the equinoctial elements one time step using a simple thrust model"""
        p = float(self.state['p'])
        f = float(self.state['f'])
        g = float(self.state['g'])
        h = float(self.state['h'])
        k = float(self.state['k'])
        L = float(self.state['L'])
        mass = float(self.state['mass'])

        w, s2 = self.compute_auxiliary_variables()

        # Thrust to mass ratio
        T_over_m = self.thrust / mass

        # Common prefactor
        sqrt_p_over_mu = np.sqrt(p / self.mu_earth)

        # Derivatives based on equinoctial form (simplified)
        p_dot = sqrt_p_over_mu * (2.0 * p / w) * T_over_m * alpha_t

        f_dot = sqrt_p_over_mu * (
            np.sin(L) * T_over_m * alpha_r +
            (((1.0 + w) * np.cos(L) + f) * T_over_m * alpha_t) / w -
            (h * np.sin(L) - k * np.cos(L)) * (g / w) * T_over_m * alpha_n
        )

        g_dot = sqrt_p_over_mu * (
            -np.cos(L) * T_over_m * alpha_r +
            (((1.0 + w) * np.sin(L) + g) * T_over_m * alpha_t) / w +
            (h * np.sin(L) - k * np.cos(L)) * (f / w) * T_over_m * alpha_n
        )

        h_dot = sqrt_p_over_mu * (s2 * np.cos(L) / (2.0 * w)) * T_over_m * alpha_n
        k_dot = sqrt_p_over_mu * (s2 * np.sin(L) / (2.0 * w)) * T_over_m * alpha_n

        L_dot = np.sqrt(self.mu_earth / p) * (w**2 / p) + sqrt_p_over_mu * (h * np.sin(L) - k * np.cos(L)) * T_over_m * alpha_n / w

        # Mass consumption (rocket equation, simple constant thrust)
        mass_dot = -self.thrust / (self.isp * self.g0)

        # Update state
        self.state['p'] += p_dot * dt
        self.state['f'] += f_dot * dt
        self.state['g'] += g_dot * dt
        self.state['h'] += h_dot * dt
        self.state['k'] += k_dot * dt
        self.state['L'] += L_dot * dt
        self.state['mass'] += mass_dot * dt

        return [p_dot, f_dot, g_dot, h_dot, k_dot, L_dot, mass_dot]

    def convert_to_cartesian(self):
        """Convert equinoctial elements to Cartesian coordinates (simplified)"""
        p = self.state['p']
        f = self.state['f']
        g = self.state['g']
        h = self.state['h']
        k = self.state['k']
        L = self.state['L']

        # Convert to classical orbital elements (approximate)
        ecc = np.sqrt(f**2 + g**2)
        inc = 2.0 * np.arctan(np.sqrt(h**2 + k**2))

        # Simplified conversion for visualization (assumes reference frame aligned)
        r = p / (1.0 + ecc * np.cos(L))
        x = r * np.cos(L)
        y = r * np.sin(L) * np.cos(inc)
        z = r * np.sin(L) * np.sin(inc)

        return x, y, z

    def run_mission(self, mission_duration: float = 3600.0, dt: float = 10.0):
        """Run a complete mission simulation"""
        print("Starting spacecraft simulation...")
        print(f"Initial orbit - p: {self.state['p']/1000:.1f} km, f: {self.state['f']:.3f}, h: {self.state['h']:.3f}")

        t = 0.0
        while t < mission_duration and self.state['mass'] > 800.0:
            # Mission phases - different control strategies
            if t < 600.0:  # Phase 1: Circularize (use tangential thrust)
                alpha_r, alpha_t, alpha_n = 0.0, 1.0, 0.0
            elif t < 1200.0:  # Phase 2: Plane change (use normal thrust)
                alpha_r, alpha_t, alpha_n = 0.0, 0.0, 0.5
            else:  # Phase 3: Fine-tuning
                alpha_r, alpha_t, alpha_n = 0.1, 0.2, 0.1

            # Store current state
            for key in self.state:
                self.history[key].append(self.state[key])
            self.history['time'].append(t)

            # Propagate
            derivatives = self.propagate_orbit(alpha_r, alpha_t, alpha_n, dt)
            t += dt

            # Progress indicator
            if int(t) % 600 == 0:
                print(f"Time: {int(t)}s - Mass: {self.state['mass']:.1f} kg - p: {self.state['p']/1000:.1f} km")

    def plot_results(self):
        """Plot the simulation results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        elements = ['p', 'f', 'g', 'h', 'k', 'mass']
        titles = ['Semi-latus rectum (p)', 'Eccentricity (f)', 'Eccentricity (g)',
                  'Inclination (h)', 'Inclination (k)', 'Mass']
        units = ['m', '', '', '', '', 'kg']

        for i, (element, title, unit) in enumerate(zip(elements, titles, units)):
            ax = axes[i // 3, i % 3]
            ax.plot(self.history['time'], self.history[element])
            ax.set_title(title)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(f'{element} ({unit})')
            ax.grid(True)

        plt.tight_layout()
        plt.show()

        # 3D orbit plot
        self.plot_3d_orbit()

    def plot_3d_orbit(self):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        x_vals, y_vals, z_vals = [], [], []
        # guard against empty history
        n_points = max(1, len(self.history['p']))
        step = max(1, n_points // 200)  # limit plotted points for performance
        for i in range(0, len(self.history['p']), step):
            temp_state = {key: self.history[key][i] for key in ['p', 'f', 'g', 'h', 'k', 'L']}
            self.state.update(temp_state)
            x, y, z = self.convert_to_cartesian()
            x_vals.append(x / 1000.0)  # km
            y_vals.append(y / 1000.0)
            z_vals.append(z / 1000.0)

        if x_vals:
            ax.plot(x_vals, y_vals, z_vals, 'b-', alpha=0.7, label='Orbit')
            ax.scatter(x_vals[0], y_vals[0], z_vals[0], c='g', s=100, label='Start')
            ax.scatter(x_vals[-1], y_vals[-1], z_vals[-1], c='r', s=100, label='End')

        # Earth sphere
        u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
        x_earth = (self.R_earth / 1000.0) * np.cos(u) * np.sin(v)
        y_earth = (self.R_earth / 1000.0) * np.sin(u) * np.sin(v)
        z_earth = (self.R_earth / 1000.0) * np.cos(v)
        ax.plot_wireframe(x_earth, y_earth, z_earth, color='lightblue', alpha=0.3)

        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        ax.set_title('Spacecraft Orbit')
        ax.legend()
        plt.show()


if __name__ == "__main__":
    sim = SpacecraftSimulator()
    sim.run_mission(mission_duration=1800.0, dt=10.0)
    sim.plot_results()

    print("\nMission Summary:")
    print(f"Final mass: {sim.state['mass']:.1f} kg")
    print(f"Final orbit shape (f,g): ({sim.state['f']:.4f}, {sim.state['g']:.4f})")
    print(f"Final orbit tilt (h,k): ({sim.state['h']:.4f}, {sim.state['k']:.4f})")

# -------------------------
# Optional examples / helpers (clean, not duplicated)
# -------------------------
def custom_control_strategy(time, state):
    """Design your own controller (returns alpha_r, alpha_t, alpha_n)"""
    if state['f'] > 0.02:  # If too elliptical
        return 0.0, 0.8, 0.0  # Use tangential thrust
    elif state['h'] > 0.03:  # If too tilted
        return 0.0, 0.0, 0.6  # Use normal thrust
    else:
        return 0.1, 0.1, 0.1  # Fine tuning

# Example: run additional missions without duplicating or misnaming classes
if False:  # flip to True to run example additional missions
    sim2 = SpacecraftSimulator()
    sim2.state.update({'p': 667800.0, 'mass': 1500.0})
    sim2.run_mission(mission_duration=1200.0, dt=10.0)
    sim2.plot_results()

    sim3 = SpacecraftSimulator()
    sim3.state.update({'f': 0.1, 'g': 0.05})
    sim3.run_mission(mission_duration=1200.0, dt=10.0)
    sim3.plot_results()