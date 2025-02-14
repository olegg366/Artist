import tkinter as tk
from tkinter import font

root = tk.Tk()
fonts = font.families()
print("Доступные шрифты:", fonts)
root.destroy()