import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

DPI = 48 # Dots Per Pixel (own screen)
SIZE = 256 / DPI
BLADES = 5
RPM = 10
FPS = 30
FRAMES = 64


# polar plot, figsize means the PLT window size, plot is a 2D array of values
first_fig, axis = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(SIZE, SIZE))
plot, = plt.plot([], [], '-')
plt.grid(color='grey', linestyle='--', linewidth=0.5)

# X values on a plot to be used to calcute r (Y)
# 1000 - number of points - 0.0001, 0.0002, .... (above 1000 doesnt change anything)
x = np.linspace(0, 2*np.pi, 1000)

def draw_propeller():
    r = np.sin(BLADES * x + (np.pi / RPM))
    plot.set_data(x, r)
    plt.savefig("propeller")

def animate_propeler(frame):
    r = np.sin(BLADES * x + (frame * np.pi / RPM))
    plot.set_data(x, r)

draw_propeller()

FuncAnimation(first_fig, animate_propeler, np.arange(-FRAMES / 2 , FRAMES / 2, 1)).save(
    "propeller_animation.gif", writer=PillowWriter(fps=FPS))
plt.show()