import tkinter as tk
from PIL import ImageTk, Image, ImageDraw
from webcolors import name_to_rgb
from multiprocessing import Process
from time import sleep

class App():
    def __init__(self, get_img_func):

        self.root = tk.Tk()

        self.line_id = None
        self.line_points = []
        self.line_options = {'fill': 'black'}

        self.get_img = get_img_func

        #конфигурируем панель управления
        self.fr_ctrl = tk.Frame(bg='#CFCFCF', width=100, height=100)
        self.fr_ctrl.pack(fill='both')

        #область рисования
        self.fr_draw = tk.Frame(width=100, height=200)
        self.fr_draw.pack(fill='both', expand=True)

        self.canvas = tk.Canvas(self.fr_draw, width=512, height=512)
        self.canvas.pack(fill='both', expand=True)

        # self.canvas.bind('<Button-1>', self.set_start)
        # self.canvas.bind('<B1-Motion>', self.draw_line)
        # self.canvas.bind('<ButtonRelease-1>', self.end_line)

        self.btfont = 'Times 20'
        self.bth = 5
        self.btw = 10

        #кнопка очистки
        self.bt_del = tk.Button(self.fr_ctrl, 
                                text='Очистить', 
                                command=self.delete, 
                                relief='groove',
                                height=self.bth, 
                                width=self.btw, 
                                font=self.btfont)
        self.bt_del.pack(fill='both', side='left')

        #всплывающее окно с изменением цвета
        self.fr_clr_set = tk.Frame(relief='groove')
        self.clrs = ['red', 'green', 'blue', 'black', 'gray', 'yellow', 'brown', 'purple', 'cyan', 'white']
        self.rgbclrs = {x: name_to_rgb(x) for x in self.clrs}
        for a in range(2):
            self.fr_clr_set.rowconfigure(a, minsize=50)
            for b in range(5):
                self.fr_clr_set.columnconfigure(b, minsize=50)
                newfr = tk.Frame(master=self.fr_clr_set, bg=self.clrs[a * 5 + b], width=50, height=50, relief='groove')
                newfr.bind('<Button-1>', lambda x, k = a * 5 + b: self.set_color(k))
                newfr.grid(row=a, column=b, padx=5, pady=5)

        #кнопка изменения цвета
        self.bt_set_clr = tk.Button(self.fr_ctrl, 
                                    text='Цвет', 
                                    command=lambda: self.fr_clr_set.place(x=self.bt_del.winfo_width(), 
                                                                          y=self.fr_ctrl.winfo_height()), 
                                    relief='groove', 
                                    height=self.bth, 
                                    width=self.btw, 
                                    font=self.btfont)
        self.bt_set_clr.pack(side='left', fill='both')

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
        self.bt_set_wd.pack(side='left', fill='both')

        #кнопка генерации
        self.bt_gen = tk.Button(self.fr_ctrl, 
                                text='Готово!', 
                                command=self.gen, 
                                height=self.bth, 
                                width=self.btw, 
                                font=self.btfont)
        self.bt_gen.pack(side='left', fill='both')

        #удаление всплывающих окон
        self.root.bind('<Button-1>', lambda x: self.del_popups())

        #картинка, чтобы затем генерировать
        self.image = Image.new("RGB", (512, 512), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)

        #предыдущие высота и ширина canvas
        self.prevw = 0
        self.prevh = 0

        self.actions = []

    def del_popups(self):
        if self.fr_wd_set is not None:
            self.fr_wd_set.place_forget()
            self.fr_wd_set.destroy()
            self.fr_wd_set = None
        self.fr_clr_set.place_forget()

    def build_wd_popup(self):
        self.fr_wd_set = tk.Frame(relief='groove', borderwidth=5, width=100)
        self.fr_wd_set.columnconfigure(0, minsize=15)
        for i in range(2, 15, 2):
            self.fr_wd_set.rowconfigure(i // 2, minsize=20)
            cv = tk.Frame(self.fr_wd_set, height=40, width=200, borderwidth=2)
            line = tk.Frame(cv, height=i, width=180, bg=self.line_options['fill'])
            line.bind('<Button-1>', lambda x, k = i: self.change_width(k))
            line.pack()
            cv.bind('<Button-1>', lambda x, k = i: self.change_width(k))
            cv.grid(row=i // 2, column=0)
        self.fr_wd_set.place(x=self.bt_del.winfo_width() + self.bt_set_clr.winfo_width(), 
                             y=self.fr_ctrl.winfo_height())

    def set_start(self, x, y):
        self.line_points.append((x, y))

    def draw_line(self, x, y):
        self.line_points.append((x, y))
        if self.line_id is not None:
            self.canvas.delete(self.line_id)
        self.line_id = self.canvas.create_line(self.line_points, **self.line_options)
        self.draw.line(self.line_points, **self.line_options)
        self.actions.append(lambda: self.draw.line(self.line_points, **self.line_options))
        # print(self.canvas.find_all())

    def end_line(self):
        self.line_points.clear()
        self.line_id = None

    def delete(self):
        self.end_line()
        self.canvas.delete("all")
        self.image = Image.new("RGB", (self.canvas.winfo_width(), self.canvas.winfo_height()), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)
        self.actions.clear()
        print(self.canvas.find_all())

    def set_color(self, x):
        self.line_options['fill'] = self.clrs[x]
        self.fr_clr_set.place_forget()
        self.end_line()

    def change_width(self, x):
        self.line_options['width'] = x
        self.fr_wd_set.place_forget()
        self.end_line()

    def update(self):
        self.root.update()
        w, h = self.canvas.winfo_height(), self.canvas.winfo_width()
        if w != self.prevw or h != self.prevh:
            self.image = self.image.resize((w, h))
            self.draw = ImageDraw.Draw(self.image)
            self.prevh = h
            self.prevw = w
            for action in self.actions:
                action()
    
    def gen(self):
        img = self.get_img(self.image)
        self.display_img = ImageTk.PhotoImage(img)
        panel = tk.Label(self.canvas, image=self.display_img)
        panel.pack(side="bottom", fill="both", expand="yes")

if __name__ == '__main__':
    app = App(lambda x: Image.open('img.png'))
    app.root.mainloop()