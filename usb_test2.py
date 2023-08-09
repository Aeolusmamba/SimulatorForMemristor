import usb

def connect_usb_device(vendor_id, product_id):
    """Connects a USB device with the given ID."""
    dev = usb.core.find(idVendor=vendor_id, idProduct=product_id)
    dev.open()

def disconnect_usb_device(vendor_id, product_id):
    """Disconnects a USB device with the given ID."""
    dev = usb.core.find(idVendor=vendor_id, idProduct=product_id)
    dev.close()

if __name__ == "__main__":
    vendor_id = 0x0781
    product_id = 0x5571
    connect_usb_device(vendor_id, product_id)
    print("USB device connected")
    disconnect_usb_device(vendor_id, product_id)
    print("USB device disconnected")
