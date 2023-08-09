import win32com.client
import time

def find_usb_device(device_id):
    wmi = win32com.client.GetObject("winmgmts:")
    devices = wmi.InstancesOf("Win32_USBControllerDevice")
    for device in devices:
        print(device.Dependent)
        if device_id in device.Dependent:
            print(device.Dependent.split("=")[1].strip('"'))
            return device.Dependent.split("=")[1].strip('"')
    return None

def unplug_and_replug_usb_device(device_id):
    device_path = find_usb_device(device_id)
    device_path = eval(repr(device_path).replace('\\\\', '\\'))
    print(device_path)
    if not device_path:
        print("Error: USB device not found.")
        return False

    try:
        # Unplug the USB device
        with open(device_path + "\\Device Parameters\\DeviceStatus", "w") as f:
            f.write("Disable")

        time.sleep(2)  # Wait to ensure the device is properly unplugged

        # Replug the USB device
        with open(device_path + "\\Device Parameters\\DeviceStatus", "w") as f:
            f.write("Enable")

        time.sleep(2)  # Wait to ensure the device is properly replugged

        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

# Example usage: Replace "DEVICE_ID_TO_UNPLUG" with the actual device ID you want to unplug/replug
device_id_to_unplug = "VID_0781&PID_5571"
unplug_and_replug_usb_device(device_id_to_unplug)
