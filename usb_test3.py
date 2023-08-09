import win32api
import win32con


# 获取USB设备路径
def get_usb_path():
    for i in range(win32api.GetLastErrorCode()):
        # 获取错误信息
        error = win32api.FormatMessage(win32api.GetLastError())
        # 如果错误信息包含"USB"则说明是USB设备错误
        if "USB" in error:
            # 获取USB设备路径
            device_path = win32api.GetWindowText(win32api.FindWindow(None, "USBSTOR"))
            return device_path

        # 断开USB设备


def disconnect_usb():
    device_path = get_usb_path()
    if device_path:
        win32api.SendMessage(device_path, win32con.WM_CLOSE, 0, 0)

    # 连接USB设备


def connect_usb():
    device_path = get_usb_path()
    if device_path:
        win32api.ShellExecute(None, "open", device_path, None, None, win32con.SW_SHOWNORMAL)