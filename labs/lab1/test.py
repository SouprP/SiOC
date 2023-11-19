import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image


# polar plot, figsize means the window size
first_fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(5.3, 5.3))
l, = plt.plot([], [], '-')
plt.grid(color='grey', linestyle='--', linewidth=0.5)

shutter_effect = Image.new('RGB', (512, 512))

jump = 4
temp = 0
blades = 3
rpm = 1000

x = np.linspace(0, 2 * np.pi, 1000)

def update(m):
    # 5 - nubmer of blades
    r = np.sin(blades * x + (m * np.pi) / 10)
    l.set_data(x, r)

    global temp
    plt.savefig('frame.png')
    shutter_effect.paste(Image.open('frame.png').crop((0, temp * jump, 512, temp * jump + jump)), (0, temp * jump))
    temp += 1

#update(np.arange(-64 / 2, 64 / 2, 1))

FuncAnimation(first_fig, update, np.arange(-64 / 2, 64 / 2, 1)).save("3wings.gif", writer=PillowWriter(fps=25))
shutter_effect.save("shutter_effect.png")