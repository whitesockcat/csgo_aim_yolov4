import win32gui,win32api
import time

wdname = 'Counter-Strike: Global Offensive'

hwnd = win32gui.FindWindow(0, wdname) # 父句柄
#hwnd1 = win32gui.FindWindowEx(hwnd, None,'类名称', None) # 目标子句柄
windowRec = win32gui.GetWindowRect(hwnd) # 目标子句柄窗口的坐标

while True:
    tempt = win32api.GetCursorPos() # 记录鼠标所处位置的坐标
    x = tempt[0]-windowRec[0] # 计算相对x坐标
    y = tempt[1]-windowRec[1] # 计算相对y坐标
    print(x,y)
    time.sleep(0.5) # 每0.5s输出一次
