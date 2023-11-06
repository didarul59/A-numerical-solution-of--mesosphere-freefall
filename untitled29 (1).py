"""**Calculation of Gravity**"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
g0 = 9.81  # m/s^2
Re = 6.371 * 10**6  # m

# Function to calculate acceleration due to gravity at altitude
def gravity_at_altitude(altitude):
    return g0 * (Re / (Re + altitude))**2

# Altitude range
altitudes = np.arange(0, 50000, 1000)

# Calculate gravity at each altitude
gravity_values = [gravity_at_altitude(altitude) for altitude in altitudes]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(altitudes, gravity_values, label='Gravity at Altitude')
plt.xlabel('Altitude (meters)')
plt.ylabel('Gravity (m/s^2)')
plt.title('Acceleration Due to Gravity vs Altitude')
#plt.grid(True)
plt.legend()
plt.show()

"""**Calculation of pressure**"""

import numpy as np
from matplotlib import pyplot as plt

p0 = 101300  # atmospheric pressure at sea level in Pa
H = 7400     # scale height in meters

P_mmHg = []
Z = []

for z in range(0, 50000):
    p_pa = p0 * np.exp(-(z/H))
    p_mmHg = p_pa * 0.00750062  # convert Pa to mmHg
    Z.append(z)
    P_mmHg.append(p_mmHg)


plt.figure(figsize=(10, 6))
plt.plot(Z, P_mmHg)
plt.xlabel('Altitude (meters)')
plt.ylabel('Pressure (mmHg)')
plt.title('Exponential Decay of Pressure')
#plt.grid()
plt.show()

"""**Calculation of air density**"""

import numpy as np
import matplotlib.pyplot as plt

# Function to calculate air density based on pressure
def calculate_air_density(pressure_mmHg, temperature_K=273):
    # Conversion factor from mmHg to Pa
    pressure_Pa = pressure_mmHg * 133.322

    # Ideal gas constant (approximately)
    R = 287.05  # J/(kg·K)

    # Calculate air density
    air_density = pressure_Pa / (R * temperature_K)

    return air_density

# Altitude range
altitudes = np.arange(0, 50000, 500)

# Pressure at sea level and altitude
pressure_sea_level_mmHg = 760
pressures = [pressure_sea_level_mmHg * np.exp(-altitude / 7400) for altitude in altitudes]

# Calculate air density at each altitude
densities = [calculate_air_density(pressure, 273) for pressure in pressures]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(altitudes, densities)
plt.xlabel('Altitude (meters)')
plt.ylabel('Air Density (kg/m³)')
plt.title('Air Density vs Altitude')
#plt.legend()
#plt.grid()
plt.show()

"""**Calculation Of Drag coefficient**"""

import numpy as np
from matplotlib import pyplot as plt

p0 = 101300  # atmospheric pressure at sea level in Pa
H = 7400     # scale height in meters
Z_max = 50000  # maximum altitude in meters

# Function to calculate drag coefficient based on altitude
def calculate_drag_coefficient(z):
    # Example: linear relationship (you can modify this based on your model)
    return 0.1 + 0.00002 * z

Z = np.arange(0, Z_max, 100)
CD = [calculate_drag_coefficient(z) for z in Z]

plt.figure(figsize=(10, 6))
plt.plot(Z, CD)
plt.xlabel('Altitude (meters)')
plt.ylabel('Drag Coefficient')
plt.title('Variation of Drag Coefficient with Altitude')
#plt.grid()
plt.show()

"""**Calculation Of Velocity**"""

import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp

# Function to calculate u(z, a, z0, lambda)
def u(z, a, z0, lambd, terms=10000):
    term1 = mp.sqrt(a)
    term2 = mp.exp(-a/2 * mp.exp(-z/lambd))
    term3 = mp.sqrt((z0 - z)/lambd + mp.nsum(lambda n: (a**n * (mp.exp(-n*z/lambd) - mp.exp(-n*z0/lambd)))/(n*mp.factorial(n)), [1, terms]))

    return float(term1 * term2 * term3)

# Constants
a = 46.686
z0 = 50000
lambda_val = 7.46 * 10**3

# Altitude range
altitudes = np.arange(0, z0, 1000)

# Calculate u for each altitude
u_values = [u(z, a, z0, lambda_val) for z in altitudes]

# Calculate (1 - z/z0) for x-axis
x_values = 1 - altitudes / z0

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x_values, u_values, label='$u(z, a, z_0, \\lambda)$', color='blue')
plt.xlabel('$(1 - \\frac{z}{z_0})$')
plt.ylabel('$u(z, a, z_0, \\lambda)$')
plt.title('Function: $u(z, a, z_0, \\lambda)$ vs $(1 - \\frac{z}{z_0})$')
#plt.grid(linewidth=0.5, linestyle='--', alpha=0.7)
plt.legend()
plt.show()

max(u_values)

import numpy as np

# Example lists of density and drag coefficient
density_values = densities  # replace with your density values
cd_values = CD  # replace with your drag coefficient values

# Constants
mass = 190  # replace with your object's mass
gravity = gravity_values
cross_sectional_area = 1.0  # replace with your reference area

# Calculate terminal velocities for each combination of density and drag coefficient
terminal_velocities = []

for density in density_values:
    for cd in cd_values:
        terminal_velocity = np.sqrt(2 * mass * gravity / (density * cross_sectional_area * cd))
        terminal_velocities.append(terminal_velocity)

# Print or use the terminal velocities as needed
print("Terminal Velocities:", terminal_velocities)

# Get the index of the maximum value
max_index = np.argmax(u_values)

# Print the result
print("Index of the maximum value:", max_index)

# Example list of 50000 values
terminal_velocitie = terminal_velocities  # replace with your actual list

# Select values at intervals of 1000
new_list = [terminal_velocitie[i] for i in range(0, len(terminal_velocitie), 1000)]

value_at_index_32 = new_list[32]

average_value = sum(value_at_index_32) / len(value_at_index_32)

# Print the result
print("Average value:", average_value)

max_value = max(u_values)
highest_velocity = max_value * average_value
print("The highest velocity is:", highest_velocity,"m/s")

# Constants
a = 46.686
z0 = 50000
lambda_val = 7.46 * 10**3
vmax = 660.1226176366592
zmax = lambda_val * np.log((vmax/average_value)**2)
print("altitude for maximum velocity:", zmax,"m")

"""**Calculation Of g-force**"""

def calculate_gravitational_acceleration(v_max, z_max):
    g = v_max**2 / (2 * z_max)
    return g

# Calculate gravitational acceleration
gravity = calculate_gravitational_acceleration(vmax, zmax)

# Display the result
print("Gravitational acceleration (g):", gravity)

import numpy as np
import matplotlib.pyplot as plt

# Constants
mass = 190  # mass of the body in kg
initial_altitude = 50000  # initial altitude in meters
final_altitude = 32000  # final altitude in meters
gravity_values = np.linspace(9.8, 9.7, 100)  # list of gravity values
drag_coefficient_values = CD  # list of drag coefficient values
initial_velocity = 0  # initial velocity in m/s
terminal_velocity = 660  # terminal velocity in m/s

# Function to calculate air resistance force
def air_resistance_force(velocity, drag_coefficient):
    return -0.5 * drag_coefficient * velocity**2

# Function to calculate acceleration including air resistance
def acceleration(velocity, gravity, drag_coefficient):
    return gravity + (air_resistance_force(velocity, drag_coefficient) / mass)

# Euler's method for numerical integration
def euler_method(dt, num_steps, gravity, drag_coefficient):
    time = np.zeros(num_steps)
    velocity = np.zeros(num_steps)
    altitude = np.zeros(num_steps)

    velocity[0] = initial_velocity
    altitude[0] = initial_altitude

    for i in range(1, num_steps):
        time[i] = time[i-1] + dt
        velocity[i] = velocity[i-1] + acceleration(velocity[i-1], gravity, drag_coefficient) * dt
        altitude[i] = altitude[i-1] - velocity[i-1] * dt  # negative sign for downward motion

        # Break the loop if terminal velocity is reached or final altitude is reached
        if velocity[i] >= terminal_velocity or altitude[i] <= final_altitude:
            break

    return time[:i+1], altitude[:i+1]

# Simulation parameters
dt = 0.1  # time step in seconds
num_steps = 1000

# Calculate mean altitude vs. time for all combinations of gravity and drag coefficient
mean_altitude = np.zeros(num_steps)

for gravity, drag_coefficient in zip(gravity_values, drag_coefficient_values):
    time, altitude = euler_method(dt, num_steps, gravity, drag_coefficient)
    mean_altitude += altitude

mean_altitude /= len(gravity_values)

# Plot the mean altitude vs. time
plt.figure(figsize=(10, 6))
plt.plot(time, mean_altitude, label='Mean Altitude')
plt.xlabel('Time (seconds)')
plt.ylabel('Altitude (meters)')
plt.legend()
plt.grid(True)
plt.title('Mean Altitude vs. Time for Falling Body with Varying Gravity and Drag Coefficient')
plt.show()