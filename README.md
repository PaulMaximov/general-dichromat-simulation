# Python utilities to simulate color perception of generalized dichromats
## A generalized dichromat is defined by the confusion line in the LMS color space. The confusion line direction is defined by a hue angle in degrees. 0° corresponds to a real protanope, 120° -- to a real deuteranope, 240° - to a real tritanope, but you can use arbitraty angles to simulate unreal dicromats.
### The example below illustrates the use of functions defined in the file dichromat_simulation_twoplanes.py. One of the two fixed planes are used as a dichromat color surface. The plane is selected to maximize the angle between the confusion line and the plane.

#### 
import PIL.Image

filename = 'source.png'

im = np.asarray(PIL.Image.open(filename), dtype=np.float64) / 255

angle = 40

simmatr = get_simulation_matrix(angle)

ims = simulate_sRGB(simmatr, im)

PIL.Image.fromarray(np.asarray(ims * 255, dtype=np.uint8)).save('result.png')
