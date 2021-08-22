import mouse
import time


if True:
    x0 = 151
    y0 = 123
    while True:
        mouse.move(x0, y0, absolute=True, duration=0.2, steps_per_second=120.0)
        print(x0,y0)
        time.sleep(0.5)
        x0 = x0 + 40
        y0 = y0 + 20
        
    