import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

# DPI - how many pixels are in 1 inch
DPI = 48 # Dots Per Pixel (own screen)
SIZE = 256 / DPI# SIZE = 256 / DPI
BLADES = 200
RPM = 10
FPS = 30 
FRAMES = 64 # M ---  -M/2, M/2
JUMP = 16
temp = 0


# polar plot, figsize means the PLT window size, plot is a 2D array of values
first_fig, axis = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(SIZE, SIZE))
plot, = plt.plot([], [], '-')
plt.grid(color='grey', linestyle='--', linewidth=0.5)

# X values on a plot to be used to calcute r (Y)
# 1000 - number of points - 0.0001, 0.0002, .... (above 1000 doesnt change anything)
x = np.linspace(0, 2*np.pi, 1000)

def draw_propeller():
    # calculation of r
    r = np.sin(BLADES * x + (np.pi / RPM))

    # setting the points on plot
    plot.set_data(x, r)
    plt.savefig("propeller")

def animate_propeller(frame):
    r = np.sin(BLADES * x + (frame * np.pi / RPM))
    plot.set_data(x, r)

# Shutter Effect
shutter_effect = Image.new('RGB', (512, 512))
def shutter_effect(frame):
    r = np.sin(BLADES * x + (frame * np.pi / RPM))
    plot.set_data(x, r)

    global temp
    plt.savefig('frame.png')
    shutter_effect.paste(Image.open('frame.png').crop((0, 
        temp * JUMP, 512, temp * JUMP + JUMP)), (0, temp * JUMP))
    temp += 1

#draw_propeller()
#shutter_effect()

# first_fig - polar axis / plot
# animate_propeller - function that will be called %FRAMES times
# np.range(....) - args of the function (animate_propeller)
# FRAMES - M = <-64, 64>
# FPS - number of frames per second showed in the .gif file
FuncAnimation(first_fig, animate_propeller, np.arange(-FRAMES / 2 , FRAMES / 2, 1)).save(
    "propeller_animation.gif", writer=PillowWriter(fps=FPS))

# shows the result - an image or animation
plt.show()