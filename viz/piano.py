import gizeh as gz
import numpy as np

width = 700
height = int(width * 0.65)
surface = gz.Surface(width=width, height=height, bg_color=(1,1,1))

# white keys
white_count = 7
white_size = (width / white_count, height)
for i in range(white_count):
    w, h = white_size
    rect = gz.rectangle(lx=w-5, ly=h-5,
        xy=[(i + 0.5) * w, h / 2],
        stroke=(0.8, 0.8, 0.8), stroke_width=2)
    rect.draw(surface)

# black keys
for i in (0, 1, 3, 4, 5):
    w, h = white_size[0] * 0.85, (white_size[1] * 0.65)
    rect = gz.rectangle(lx=w-5, ly=h-5,
        xy=[(i + 0.5 + 0.5) * white_size[0], h / 2],
        fill=(0.1, 0.1, 0.1))
    rect.draw(surface)

for i, label in enumerate(('C', 'D', 'E', 'F', 'G', 'A', 'B')):
    text = gz.text(label, 'Helvetica', white_size[0]*0.4,
        xy=[(i + 0.5) * white_size[0], height * 0.85])
    text.draw(surface)

for i, label in ((0, 'Db'), (1, 'Eb'), (3, 'Gb'), (4, 'Ab'), (5, 'Bb')):
    text = gz.text(label, 'Helvetica', white_size[0]*0.35,
        xy=[(i + 0.5 + 0.5) * white_size[0], height * 0.55],
        fill=(1,1,1))
    text.draw(surface)

surface.write_to_png('piano-octave.png')
