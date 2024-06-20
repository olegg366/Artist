import tkinter as tk
from tkinter.ttk import Progressbar
from tkinter.ttk import Style
from PIL import ImageTk, Image, ImageDraw
from webcolors import name_to_rgb
from multiprocessing import Process
from time import sleep
import cv2
import numpy as np

class App():
    def __init__(self):

        self.root = tk.Tk()
        self.root.attributes("-fullscreen", True)

        self.line_id = None
        self.line_points = []
        self.line_options = {'fill': 'black'}

        #конфигурируем панель управления
        self.fr_ctrl = tk.Frame(bg='#CFCFCF', width=100, height=100)
        self.fr_ctrl.pack(fill='both', side='left')

        #область рисования
        self.fr_draw = tk.Frame(width=100, height=200)
        self.fr_draw.pack(fill='both', expand=True)

        self.canvas = tk.Canvas(self.fr_draw, width=512, height=512)
        self.canvas.pack(fill='both', expand=True)

        self.canvas.bind('<Button-1>', self.set_start)
        self.canvas.bind('<B1-Motion>', self.draw_line)
        self.canvas.bind('<ButtonRelease-1>', lambda x: self.end_line())

        self.btfont = 'Arial 26'
        self.bth = 5
        self.btw = 13

        #кнопка очистки
        self.bt_del = tk.Button(self.fr_ctrl, 
                                text='Очистить', 
                                command=self.delete, 
                                relief='groove',
                                height=self.bth, 
                                width=self.btw, 
                                font=self.btfont)
        self.bt_del.pack(fill='both', side='top')

        #всплывающее окно с изменением цвета
        self.fr_clr_set = tk.Frame(relief='groove')
        self.clrs = ['red', 'green', 'blue', 'black', 'gray', 'yellow', 'brown', 'purple', 'cyan', 'white']
        self.rgbclrs = {x: name_to_rgb(x) for x in self.clrs}

        #всплывающее окно изменения толщины
        self.fr_wd_set = None

        #кнопка изменения толщины
        self.bt_set_wd = tk.Button(self.fr_ctrl, 
                                   text='Толщина', 
                                   command=self.build_wd_popup, 
                                   relief='groove', 
                                   height=self.bth, 
                                   width=self.btw, 
                                   font=self.btfont)
        self.bt_set_wd.pack(side='top', fill='both')

        #кнопка генерации
        self.bt_gen = tk.Button(self.fr_ctrl, 
                                text='Готово!', 
                                command=lambda: 1, 
                                height=self.bth, 
                                width=self.btw, 
                                font=self.btfont)
        self.bt_gen.pack(side='top', fill='both')
        
        self.status_drawing = tk.Frame(self.fr_ctrl, bg='red', width=250)
        self.status_drawing.pack(side='top', fill='both')
        self.now_clr = 'red'

        #удаление всплывающих окон
        self.root.bind('<Button-1>', lambda x: self.del_popups())

        #картинка, чтобы затем генерировать
        self.image = Image.new("RGB", (512, 512), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)
        
        self.points_image = ImageTk.PhotoImage(Image.fromarray(np.ones((320, 240))))
        self.points_image_panel = tk.Label(self.fr_ctrl, image=self.points_image)
        self.points_image_panel.pack(side='bottom', fill='both')
        
        #предыдущие высота и ширина canvas
        self.prevw = 0
        self.prevh = 0
        
        self.fr_status = tk.Frame(self.fr_ctrl)
        self.fr_status.pack(side='left', fill='both', expand=True)
        
        self.text_lb = tk.Label(self.fr_status, relief='groove')
        
        self.fr_progressbar = tk.Frame(self.fr_status, height=60)
        
        self.progressval = 0
        self.progressmax = 50
        
        self.style = Style(self.root)
        self.style.layout('text.Horizontal.TProgressbar',
                    [('Horizontal.Progressbar.trough',
                    {'children': [('Horizontal.Progressbar.pbar',
                                    {'side': 'left', 'sticky': 'ns'})],
                        'sticky': 'nswe'}),
                    ('Horizontal.Progressbar.label', {'sticky': ''})])
        self.style.configure('text.Horizontal.TProgressbar', text=f'0/{self.progressmax}', background='yellow', font='Montserrat 20')

        self.progressbar = Progressbar(self.fr_progressbar, style='text.Horizontal.TProgressbar', length=200, maximum=self.progressmax)
        self.lb_progressbar = tk.Label(self.fr_progressbar, text='Идет обработка, подождите...', font='Times 18')

        self.actions = []

    def del_popups(self):
        if self.fr_wd_set is not None:
            self.fr_wd_set.place_forget()
            self.fr_wd_set.destroy()
            self.fr_wd_set = None
        self.fr_clr_set.place_forget()
        
    def change_status(self):
        if self.now_clr == 'red':
            self.now_clr = 'green'
        elif self.now_clr == 'green':
            self.now_clr = 'yellow'
        else:
            self.now_clr = 'red'
        self.status_drawing.configure(bg=self.now_clr)
        
    def progressbar_step(self, amount):        
        self.progressval = (self.progressval + amount) % (self.progressmax + 1)
        self.style.configure('text.Horizontal.TProgressbar', text=f'{self.progressval}/{self.progressmax}')
        self.progressbar.step(amount)
        
    def setup_progressbar(self):
        self.text_lb.pack_forget()
    
        
        self.fr_progressbar.pack(anchor='s', fill='x', expand=True, padx=15)
        self.fr_progressbar.pack_propagate(False)
        
        self.progressbar.pack(fill='both', expand=True)
        self.lb_progressbar.pack()
        
    def build_wd_popup(self):
        self.fr_wd_set = tk.Frame(relief='groove', borderwidth=5, width=100)
        for i in range(2, 15, 2):
            cv = tk.Frame(self.fr_wd_set, width=self.bt_set_wd.winfo_width() - 10, height=40, relief='groove', border=5)
            line = tk.Frame(cv, height=i, width=self.bt_set_wd.winfo_width() - 30, bg=self.line_options['fill'])
            cv.bind('<Button-1>', lambda x, k = i: self.change_width(k))
            cv.grid(row=i // 2, column=0)
            line.bind('<Button-1>', lambda x, k = i: self.change_width(k))
            line.pack(padx=10, pady=10)
        self.fr_wd_set.place(x=self.bt_del.winfo_width() + self.bt_set_clr.winfo_width(), 
                             y=self.fr_ctrl.winfo_height())

    def set_start(self, cords):
        if isinstance(cords, tk.Event):
            x, y = cords.x, cords.y
        else:
            x, y = cords
        self.line_points.append((x, y))

    def draw_line(self, cords):
        if isinstance(cords, tk.Event):
            x, y = cords.x, cords.y
        else:
            x, y = cords
        self.line_points.append((x, y))
        if self.line_id is not None:
            self.canvas.delete(self.line_id)
        self.line_id = self.canvas.create_line(self.line_points, **self.line_options)
        self.draw.line(self.line_points, **self.line_options)
        self.actions.append(lambda: self.draw.line(self.line_points, **self.line_options))

    def end_line(self):
        self.line_points.clear()
        self.line_id = None

    def delete(self):
        self.end_line()
        self.canvas.delete("all")
        self.image = Image.new("RGB", (self.canvas.winfo_width(), self.canvas.winfo_height()), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)
        self.actions.clear()

    def set_color(self, x):
        self.line_options['fill'] = self.clrs[x]
        self.fr_clr_set.place_forget()

    def change_width(self, x):
        self.line_options['width'] = x
        self.fr_wd_set.place_forget()

    def update(self, image: Image = None):
        if image is not None:
            image = Image.fromarray(image.astype('uint8'))
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            self.points_image = ImageTk.PhotoImage(image)
            self.points_image_panel.configure(image=self.points_image)
        self.root.update()
        h, w = self.canvas.winfo_height(), self.canvas.winfo_width()
        if w != self.prevw or h != self.prevh:
            self.image = self.image.resize((w, h))
            self.draw = ImageDraw.Draw(self.image)
            self.prevh = h
            self.prevw = w
            for action in self.actions:
                action()
                
    def print_text(self, text):
        self.fr_progressbar.pack_forget()
        self.lb_progressbar.pack_forget()
        self.text_lb.pack(side='left', expand=True, fill='both')
        self.text_lb.configure(text=text, font=self.btfont)
                
    def remove_img(self):
        self.image_panel.pack_forget()
    
    def display(self, img: Image):
        img = img.resize((img.size[0] * 2, img.size[1] * 2))
        self.display_img = ImageTk.PhotoImage(img)
        self.image_panel = tk.Label(self.canvas, image=self.display_img)
        self.image_panel.pack(side="bottom", fill="both", expand="yes")
from skimage.transform import resize
if __name__ == '__main__':
    vid = cv2.VideoCapture(0)
    app = App()
    app.setup_progressbar()
    while True:
        res, img = vid.read()
        if not res:
            break
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        app.progressbar_step(1)
        app.update(resize(img, (img.shape[0] // 2, img.shape[1] // 2)) * 255)
    vid.release()
    