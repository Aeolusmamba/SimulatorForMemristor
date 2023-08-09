import usb.core
import usb.util
# import usb.backend.openusb
import usb.backend.libusb1


def connect_usb_device(vendor_id, product_id):
    # Find the USB device using the vendor ID and product ID
    device = usb.core.find(idVendor=vendor_id, idProduct=product_id, backend=usb.backend.libusb1.get_backend())

    if device is None:
        raise ValueError("USB device not found.")
    # print(device)

    # try:
    #     # Detach the device from the kernel driver if it's already attached
    #     if device.is_kernel_driver_active(0):
    #         device.detach_kernel_driver(0)
    #
    #     # Set the active configuration (it's usually the first one)
    #     device.set_configuration()
    #
    #     # Claim the interface
    #     usb.util.claim_interface(device, 0)
    #
    #     print("USB device connected.")
    # except Exception as e:
    #     print(f"Error connecting USB device: {e}")

    # Detach the device from the kernel driver if it's already attached
    if device.is_kernel_driver_active(0):
        device.detach_kernel_driver(0)

    # Set the active configuration (it's usually the first one)
    device.set_configuration()

    # Claim the interface
    usb.util.claim_interface(device, 0)

    print("USB device connected.")

def disconnect_usb_device(vendor_id, product_id):
    # Find the USB device using the vendor ID and product ID
    device = usb.core.find(idVendor=vendor_id, idProduct=product_id, backend=usb.backend.libusb1.get_backend())

    if device is None:
        raise ValueError("USB device not found.")

    # try:
    #     # Release the claimed interface
    #     usb.util.release_interface(device, 0)
    #
    #     # Reattach the kernel driver
    #     device.attach_kernel_driver(0)
    #
    #     print("USB device disconnected.")
    # except Exception as e:
    #     print(f"Error disconnecting USB device: {e}")

    # Release the claimed interface
    usb.util.release_interface(device, 0)

    # Reattach the kernel driver
    # device.attach_kernel_driver(0)

    print("USB device disconnected.")

if __name__ == "__main__":
    # Replace these values with your USB device's vendor ID and product ID
    # vendor_id = 0x0BDA
    # product_id = 0x565A
    # vendor_id = 0x346D
    # product_id = 0x5678
    vendor_id = 0x0781
    product_id = 0x5571

    # Set the backend to 'libusb' for Windows
    # usb.core.find = usb.backend.find(find_library=lambda x: 'winusb' if 'win' in x else None)

    # try:
    #     connect_usb_device(vendor_id, product_id)
    #     # Your code to interact with the connected USB device can go here
    # except Exception as e:
    #     print(f"Error: {e}")
    # finally:
    #     disconnect_usb_device(vendor_id, product_id)
    connect_usb_device(vendor_id, product_id)
    disconnect_usb_device(vendor_id, product_id)
