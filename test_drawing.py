import pickle
from main import draw_img, get_gcode, send_gcode

all = []

with open('last_trajectory.lst', 'rb') as f:
    all = pickle.load(f)
    
gcode = get_gcode(all)  
send_gcode(gcode)