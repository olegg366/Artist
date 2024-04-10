from interface import App
from multiprocessing import Process
import pyautogui as pg
from time import sleep

def upd():
    global app
    
    while True: 
        app.update()


if __name__ == '__main__':

    app = App(lambda x: x)

    try:
        x, y = pg.position()
        x -= app.fr_draw.winfo_x()
        y -= app.fr_draw.winfo_y() + 25
        app.set_start(x, y)
        while True:
            x, y = pg.position()
            x -= app.fr_draw.winfo_x()
            y -= app.fr_draw.winfo_y() + 25
            if len(app.line_points) <= 2:
                app.set_start(x, y)
            else:
                app.draw_line(x, y)
            app.update()

    except Exception as e:
        print(e)

    # t.terminate()
