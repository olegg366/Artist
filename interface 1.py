import tkinter as tk
from tkinter.ttk import Progressbar, Style
from PIL import ImageTk, Image, ImageDraw
from webcolors import name_to_rgb
from multiprocessing import Process
from time import sleep
from textwrap import wrap
from skimage.transform import resize
import cv2
import numpy as np

# Размеры изображения
IMAGE_HEIGHT = 300
IMAGE_WIDTH = 400

class RoundedFrame(tk.Canvas):
    """
    Класс для создания круглой рамки с текстом.
    """
    def __init__(self, 
                 master=None, 
                 text:str="", 
                 radius=35, 
                 btnforeground="#000000", 
                 btnpressclr="#d2d6d3", 
                 font='Jost 20', 
                 btnbackground="#ffffff", 
                 btncontour='',
                 btncontourwidth=0,
                 clicked=None, 
                 click=True, 
                 *args, **kwargs):
        super(RoundedFrame, self).__init__(master, *args, **kwargs)
        self.config(bg=self.master["bg"])
        self.btnbackground = btnbackground
        self.btnforeground = btnforeground
        self.font = font
        self.btnpressclr = btnpressclr
        self.clicked = clicked
        self.click = click

        self.radius = radius        
        
        self.rect = self.round_rectangle(0, 0, 0, 0, 
                                         tags="button", 
                                         radius=radius, 
                                         fill=btnbackground, 
                                         width=btncontourwidth,
                                         outline=btncontour)
        self.text = self.create_text(0, 0, text=text, tags="button", fill=btnforeground, font=font, justify="center")

        self.tag_bind("button", "<ButtonPress>", self.border)
        self.tag_bind("button", "<ButtonRelease>", self.border)
        self.bind("<Configure>", self.resize)
        
        text_rect = self.bbox(self.text)
        if int(self["width"]) < text_rect[2]-text_rect[0]:
            self["width"] = (text_rect[2]-text_rect[0]) + 10
        
        if int(self["height"]) < text_rect[3]-text_rect[1]:
            self["height"] = (text_rect[3]-text_rect[1]) + 10
          
    def round_rectangle(self, x1, y1, x2, y2, radius=25, update=False, **kwargs): 
        """
        Создает или обновляет круглую рамку.
        """
        points = [x1+radius, y1,
                x1+radius, y1,
                x2-radius, y1,
                x2-radius, y1,
                x2, y1,
                x2, y1+radius,
                x2, y1+radius,
                x2, y2-radius,
                x2, y2-radius,
                x2, y2,
                x2-radius, y2,
                x2-radius, y2,
                x1+radius, y2,
                x1+radius, y2,
                x1, y2,
                x1, y2-radius,
                x1, y2-radius,
                x1, y1+radius,
                x1, y1+radius,
                x1, y1]

        if not update:
            return self.create_polygon(points, smooth=True, **kwargs)
        else:
            self.coords(self.rect, points)

    def resize(self, event):
        """
        Изменяет размер рамки при изменении размеров окна.
        """
        text_bbox = self.bbox(self.text)

        if self.radius > event.width or self.radius > event.height:
            radius = min((event.width, event.height))
        else:
            radius = self.radius

        width, height = event.width, event.height

        if event.width < text_bbox[2]-text_bbox[0]:
            width = text_bbox[2]-text_bbox[0] + 30
        
        if event.height < text_bbox[3]-text_bbox[1]:  
            height = text_bbox[3]-text_bbox[1] + 30
        
        self.round_rectangle(5, 5, width-5, height-5, radius, update=True)

        bbox = self.bbox(self.rect)

        x = ((bbox[2]-bbox[0])/2) - ((text_bbox[2]-text_bbox[0])/2)
        y = ((bbox[3]-bbox[1])/2) - ((text_bbox[3]-text_bbox[1])/2)

        self.moveto(self.text, x, y)

    def border(self, event):
        """
        Изменяет цвет рамки при нажатии и отпускании кнопки.
        """
        if event.type == "4":
            if self.click:
                self.itemconfig(self.rect, fill=self.btnpressclr)
            if self.clicked is not None:
                self.clicked()
        else:
            self.itemconfig(self.rect, fill=self.btnbackground)
    def change_color(self, clr):
        """
        Изменяет цвет рамки.
        """
        self.itemconfig(self.rect, fill=clr)
        self.btnbackground = clr
        
    def change_text(self, text):
        """
        Изменяет текст внутри рамки.
        """
        self.itemconfig(self.text, text=text)
        
        bbox = self.bbox(self.text)
        w = bbox[2] - bbox[0]
        if w > self.winfo_width():
            average_char_width = w / len(text)
            chars_per_line = int(self.winfo_width() / average_char_width)
            while w > self.winfo_width():  
                wrapped_text = '\n'.join(wrap(text, chars_per_line))
                self.itemconfig(self.text, text=wrapped_text)
                self.update()
                chars_per_line -= 1
                bbox = self.bbox(self.text)
                w = bbox[2] - bbox[0]
        

def add_corners(im, rad):
    """
    Добавляет закругленные углы к изображению.
    """
    circle = Image.new('L', (rad * 2, rad * 2), 0)
    draw = ImageDraw.Draw(circle)
    draw.ellipse((0, 0, rad * 2, rad * 2), fill=255)
    alpha = Image.new('L', im.size, "white")
    w, h = im.size
    alpha.paste(circle.crop((0, 0, rad, rad)), (0, 0))
    alpha.paste(circle.crop((0, rad, rad, rad * 2)), (0, h - rad))
    alpha.paste(circle.crop((rad, 0, rad * 2, rad)), (w - rad, 0))
    alpha.paste(circle.crop((rad, rad, rad * 2, rad * 2)), (w - rad, h - rad))
    im.putalpha(alpha)
    return im
class App():
    """
    Основной класс приложения.
    """
    def __init__(self):
        self.root = tk.Tk()
        self.root.attributes("-fullscreen", True)  # Устанавливаем полноэкранный режим

        self.line_id = None  # Идентификатор текущей линии
        self.line_points = []  # Список точек текущей линии
        self.line_options = {'fill': 'black', 'width': 10}  # Опции для рисования линии

        # Конфигурируем панель управления
        self.fr_ctrl = tk.Frame(width=self.root.winfo_width(), height=100)
        self.fr_ctrl.pack(side='left', fill='y')

        # Область рисования
        self.fr_draw = tk.Frame(width=100, height=200)
        self.fr_draw.pack(fill='both', expand=True)

        self.canvas = tk.Canvas(self.fr_draw, width=512, height=512)
        self.canvas.pack(fill='both', expand=True)

        # Привязываем события к канвасу
        self.canvas.bind('<Button-1>', self.set_start)  # Начало рисования линии
        self.canvas.bind('<B1-Motion>', self.draw_line)  # Процесс рисования линии
        self.canvas.bind('<ButtonRelease-1>', lambda x: self.end_line())  # Конец рисования линии

        self.btfont = 'Jost 50 bold'  # Шрифт для кнопок
        self.bth = 150  # Высота кнопок
        self.btw = 20  # Ширина кнопок
        self.pad = 10  # Отступы
        self.btclr = '#6d2222'  # Цвет кнопок
        self.btpress = '#e77774'  # Цвет кнопок при нажатии
        self.bttextclr = 'white'  # Цвет текста на кнопках
        
        self.image_panel = tk.Label(self.canvas)  # Панель для отображения изображений

        self.points_image = ImageTk.PhotoImage(Image.fromarray(np.ones((IMAGE_HEIGHT, IMAGE_WIDTH))))
        self.points_image_panel = tk.Label(self.fr_ctrl, image=self.points_image)
        self.points_image_panel.pack(side='top', fill='both', pady=self.pad)
        
        self.status_drawing = RoundedFrame(master=self.fr_ctrl, 
                                            width=self.btw, 
                                            height=self.bth - 25, 
                                            btnbackground='red', 
                                            bg='red', 
                                            click=False)
        self.status_drawing.pack(side='top', fill='both', pady=self.pad)
        self.now_clr = 'red'  # Текущий цвет статуса
        
        self.fr_status = tk.Frame(self.fr_ctrl)
        self.fr_status.pack(side='top', fill='both', pady=self.pad)

        # Кнопка очистки
        self.bt_del = RoundedFrame(self.fr_ctrl,
                                text='Очистить', 
                                clicked=self.delete,
                                btnbackground=self.btclr,
                                btnforeground=self.bttextclr,
                                btnpressclr=self.btpress,
                                font=self.btfont,
                                height=self.bth,
                                width=self.btw)
        self.bt_del.pack(fill='both', side='top', pady=self.pad)

        # Всплывающее окно с изменением цвета
        self.clrs = ['red', 'green', 'blue', 'black', 'gray', 'yellow', 'brown', 'purple', 'cyan', 'white']
        self.rgbclrs = {x: name_to_rgb(x) for x in self.clrs}

        # Кнопка изменения толщины
        self.bt_set_wd = RoundedFrame(self.fr_ctrl, 
                                       text='Толщина', 
                                       clicked=self.setup_wd_popup, 
                                       relief='groove', 
                                       height=self.bth, 
                                       btnpressclr=self.btpress,
                                       btnforeground=self.bttextclr,
                                       width=self.btw, 
                                       btnbackground=self.btclr,
                                       font=self.btfont)
        self.bt_set_wd.pack(side='top', fill='both', pady=self.pad)
        
        self.fr_wd_set = RoundedFrame(btncontour='#a72525', btncontourwidth=5, width=200, click=False)
        
        for i in range(4, 15, 2):
            cv = tk.Canvas(self.fr_wd_set, width=200, height=40, bg='white')
            line = tk.Frame(cv, height=i, width=190, bg=self.line_options['fill'])
            cv.bind('<Button-1>', lambda x, k = i: self.change_width(k))
            cv.grid(row=i // 2, column=0, padx=10, pady=12)
            line.bind('<Button-1>', lambda x, k = i: self.change_width(k))
            line.pack(padx=5, pady=10)

        # Кнопка генерации
        self.bt_gen = RoundedFrame(self.fr_ctrl,
                                    btnbackground=self.btclr,
                                    btnforeground=self.bttextclr,
                                    clicked=self.gen,
                                    text='Готово!', 
                                    height=self.bth, 
                                    btnpressclr=self.btpress,
                                    width=self.btw, 
                                    font="Jost 55 bold")
        self.bt_gen.pack(side='top', fill='both', pady=self.pad)

        # Картинка, чтобы затем генерировать
        self.image = Image.new("RGB", (512, 512), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)
        
        # Предыдущие высота и ширина canvas
        self.prevw = 0
        self.prevh = 0
        
        self.fr_progressbar = tk.Frame(self.fr_status, height=60)
        
        self.progressval = 0  # Текущее значение прогресс-бара
        self.progressmax = 50  # Максимальное значение прогресс-бара
        
        self.style = Style(self.root)
        self.style.layout('text.Horizontal.TProgressbar',
                    [('Horizontal.Progressbar.trough',
                    {'children': [('Horizontal.Progressbar.pbar',
                                    {'side': 'left', 'sticky': 'ns'})],
                        'sticky': 'nswe'}),
                    ('Horizontal.Progressbar.label', {'sticky': ''})])
        self.style.configure('text.Horizontal.TProgressbar', text=f'0/{self.progressmax}', background='cyan', font='Jost 20')

        self.progressbar = Progressbar(self.fr_progressbar, style='text.Horizontal.TProgressbar', length=200, maximum=self.progressmax)
        self.lb_progressbar = tk.Label(self.fr_progressbar, text='Идет обработка, подождите...', font='Jost 16')
        
        self.flag_recognition = 0  # Флаг для распознавания
        self.flag_answer = 0  # Флаг для ответа
        
        self.bt_yes = RoundedFrame(self.fr_status, 
                                   text="Да", 
                                   clicked=self.rec,
                                   click=True,
                                   font="Jost 30",
                                   btnbackground=self.btclr,
                                   btnforeground='white')
        
        self.bt_no = RoundedFrame(self.fr_status, 
                                  text="Нет", 
                                  font="Jost 30",
                                  clicked=self.nrec, 
                                  click=True,
                                  btnbackground=self.btclr,
                                  btnforeground='white')
        
        self.actions = []  # Список действий для отмены
        self.flag_generate = 0  # Флаг для генерации
        
        # Привязываем события к удалению всплывающих окон
        self.bt_del.bind('<Button-1>', lambda x: self.del_popups())
        self.bt_gen.bind('<Button-1>', lambda x: self.del_popups())
        self.fr_ctrl.bind('<Button-1>', lambda x: self.del_popups())
        self.fr_status.bind('<Button-1>', lambda x: self.del_popups())
        self.points_image_panel.bind('<Button-1>', lambda x: self.del_popups())
        self.status_drawing.bind('<Button-1>', lambda x: self.del_popups())
        
        self.print_instructions()  # Выводим инструкции
    
    def setup_wd_popup(self):
        """
        Показывает всплывающее окно для изменения толщины линии.
        """
        self.fr_wd_set.place(x=self.bt_set_wd.winfo_width() + 5, y=self.bt_set_wd.winfo_y() - 100)

    def del_popups(self):
        """
        Скрывает все всплывающие окна.
        """
        self.fr_wd_set.place_forget()
        
    def gen(self):
        """
        Устанавливает флаг для генерации.
        """
        self.flag_generate = 1
    
    def rec(self):
        """
        Устанавливает флаг для распознавания и ответа.
        """
        self.flag_recognition = 1
        self.flag_answer = 1
    
    def nrec(self):
        """
        Сбрасывает флаг для распознавания и устанавливает флаг для ответа.
        """
        self.flag_recognition = 0
        self.flag_answer = 1
        
    def print_instructions(self):
        """
        Выводит инструкции на экран.
        """
        text = ["- начать/закончить", "- перемещать курсор", "- рисовать", "- очистить все"]
        imgs_names = ["thumb_up.png", "point_up.png", "click.png", "open.png"]
        self.instruction_frame = tk.Frame(self.canvas)
        self.instruction_frame.pack(fill='both', padx=100)
        self.signs = []
        for idx, sentence in enumerate(text):
            label = tk.Label(self.instruction_frame, text=sentence, font="Jost 50")
            self.signs.append(ImageTk.PhotoImage(Image.open('images/' + imgs_names[idx])))
            image = tk.Label(self.instruction_frame, image=self.signs[-1])
            
            image.grid(row=idx, column=0)
            label.grid(row=idx, column=1, sticky='w')
        
    def remove_instructions(self):
        """
        Удаляет инструкции с экрана.
        """
        self.instruction_frame.pack_forget()
        self.root.update()
        
    def change_status(self):
        """
        Изменяет цвет статуса.
        """
        if self.now_clr == 'red':
            self.now_clr = 'green'
        elif self.now_clr == 'green':
            self.now_clr = 'yellow'
        else:
            self.now_clr = 'red'
        self.status_drawing.change_color(self.now_clr)
        self.root.update()
        
    def progressbar_step(self, amount):        
        """
        Обновляет прогресс-бар.
        """
        self.progressval = (self.progressval + amount) % (self.progressmax + 1)
        if (self.progressval == 0):
            self.progressval = 1
        self.style.configure('text.Horizontal.TProgressbar', text=f'{self.progressval}/{self.progressmax}')
        self.progressbar.step(amount)
        self.root.update()
        
    def check_recognition(self):
        """
        Проверяет статус распознавания и отображает кнопки "Да" и "Нет".
        """
        self.root.update()
        self.bt_yes.configure(width=self.fr_ctrl.winfo_width() // 2 - 5, height=self.bt_gen.winfo_height() // 2)
        self.bt_no.configure(width=self.fr_ctrl.winfo_width() // 2 - 5, height=self.bt_gen.winfo_height() // 2)
        self.bt_yes.pack(side='left')
        self.bt_no.pack(side='left')
        
    def setup_progressbar(self):
        """
        Настраивает прогресс-бар.
        """
        self.bt_yes.pack_forget()
        self.bt_no.pack_forget()
        
        self.fr_progressbar.pack(anchor='s', fill='x', padx=15, pady=2)
        self.fr_progressbar.pack_propagate(False)
        
        self.progressbar.pack(fill='both')
        self.lb_progressbar.pack()
        self.lb_progressbar.pack()
        
        self.root.update()

    def set_start(self, cords):
        """
        Устанавливает начальную точку линии.
        """
        if isinstance(cords, tk.Event):
            x, y = cords.x, cords.y
        else:
            x, y = cords
        self.line_points.append((x, y))
        self.del_popups()

    def draw_line(self, cords):
        """
        Рисует линию на канвасе.
        """
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
        """
        Завершает рисование линии.
        """
        self.line_points.clear()
        self.line_id = None

    def delete(self):
        """
        Очищает канвас.
        """
        self.end_line()
        self.canvas.delete("all")
        self.image = Image.new("RGB", (self.canvas.winfo_width(), self.canvas.winfo_height()), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)
        self.actions.clear()

    def change_width(self, x):
        """
        Изменяет толщину линии.
        """
        self.line_options['width'] = x
        self.fr_wd_set.place_forget()

    def update(self, image: Image = None):
        """
        Обновляет изображение на панели и канвасе.
        """
        if image is not None:
            image = Image.fromarray(image.astype('uint8'))
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image = add_corners(image, 35)
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
        """
        Выводит текст на панель статуса.
        """
        self.status_drawing.change_text(text)
        self.root.update()
                
    def remove_img(self):
        """
        Удаляет изображение с панели.
        """
        self.image_panel.pack_forget()
    
    def display(self, img: Image):
        """
        Отображает изображение на панели.
        """
        img = img.resize((img.size[0] * 2, img.size[1] * 2))
        self.display_img = ImageTk.PhotoImage(img)
        self.image_panel.configure(image=self.display_img)
        self.image_panel.pack(side="bottom", fill="both", expand="yes")
        self.root.update()

if __name__ == '__main__':
    vid = cv2.VideoCapture(2)
    app = App()
    # app.setup_progressbar()
    app.check_recognition()
    while True:
        res, img = vid.read()
        if not res:
            break
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # app.progressbar_step(1)
        app.change_status()
        # app.print_text("Вы сказали: я нарисовал апельсиновое облако")
        print(app.flag_recognition)
        app.update(resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH)) * 255)
    vid.release()