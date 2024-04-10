import win32clipboard as wc
from io import BytesIO
from imageio import imread
from PIL import Image

def send_to_clipboard(clip_type, data):
    wc.OpenClipboard()
    wc.EmptyClipboard()
    wc.SetClipboardData(clip_type, data)
    wc.CloseClipboard()

img = Image.fromarray(imread('img.png'))
output = BytesIO()
img.convert("RGB").save(output, "BMP")
data = output.getvalue()[14:]
output.close()
send_to_clipboard(wc.CF_DIB, data)