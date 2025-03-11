## Install YOLOv5

```
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

## Run detection program on pi5
```
cd Documents/yolo-zoneCount
source ./yolov5-venv/bin/activate
python yolo_webcam.py
```

## Creating symlinks for cameras on pi5

Follow these steps to create the symlinks using Udev rules:

1. Plug in cameras
2. Create a dump with the information about all devices by running `udevadm info --export-db > file.txt`. This will create a `file.txt` file with information about all devices.
3. Find the `ID_MODEL` and `ID_SERIAL` attributes of the cameras. To do this you can search for the camera's manufacturer or model.
4. Create the file for the Udev rules: create a file in `/etc/udev/rules.d/`, for example, `70-persistent-capture-devices.rules`.
5. Open the file you just created as root (otherwise you wont be able to save it), this can be done by running `sudo nano filename.rules` for example.
6. Create the Udev rules, the rules should look like this:
    ```
    SUBSYSTEM=="video4linux", ENV{ID_MODEL}=="replace with model for zone A", ENV{ID_SERIAL}=="replace with serial for zone A", RUN+="/usr/bin/at -M -f /home/robotoo/symlinkzonea.sh now"
    SUBSYSTEM=="video4linux", ENV{ID_MODEL}=="replace with model for zone B", ENV{ID_SERIAL}=="replace with serial for zone B", RUN+="/usr/bin/at -M -f /home/robotoo/symlinkzoneb.sh now"
    SUBSYSTEM=="video4linux", ENV{ID_MODEL}=="replace with model for zone C", ENV{ID_SERIAL}=="replace with serial for zone C", RUN+="/usr/bin/at -M -f /home/robotoo/symlinkzonec.sh now"
    ```
    For example:
   ```
   SUBSYSTEM=="video4linux", ENV{ID_MODEL}=="Logitech_BRIO", ENV{ID_SERIAL}=="046d_Logitech_BRIO_B5317ACD", RUN+="/usr/bin/at -M -f /home/robotoo/symlinkzonea.sh now"
   SUBSYSTEM=="video4linux", ENV{ID_MODEL}=="Trust_USB_Camera", ENV{ID_SERIAL}=="Sunplus_IT_Co_Trust_USB_Camera_20200707002", RUN+="/usr/bin/at -M -f /home/robotoo/symlinkzoneb.sh now"
   SUBSYSTEM=="video4linux", ENV{ID_MODEL}=="Logitech_BRIO", ENV{ID_SERIAL}=="046d_Logitech_BRIO_17E63C5B", RUN+="/usr/bin/at -M -f /home/robotoo/symlinkzonec.sh now"
   ```
   Note: be sure that you have the `at` command available. If it isn't available, install it using `sudo apt-get install at`
7. Make the `symlinkzonea.sh`, `symlinkzoneb.sh` and `symlinkzonec.sh` files in the `/home/robotoo` directory.
8. Use the following bash script template for the sh files. This is the script for zone A with the `046d_Logitech_BRIO_B5317ACD` serial id from the example in step 6:
   ```
   #!/bin/bash
   DEVICE_PATH="/dev/v4l/by-id/usb-046d_Logitech_BRIO_17E63C5B-video-index0"
   SYMLINK_PATH="/dev/camerazonea"

   # Wait for the device to appear
   while [ ! -e $DEVICE_PATH ]
   do
     sleep 1
   done

   # Get the actual device path
   ACTUAL_DEVICE_PATH=$(readlink -f $DEVICE_PATH)

   # Create the symlink
   ln -sf $ACTUAL_DEVICE_PATH $SYMLINK_PATH
   ```
9. Make the scripts executable by running `chmod +x /home/robotoo/symlinkzonea.sh` (repeat for b and c).
10. Reload the udev rules by running this command:
    ```
    sudo udevadm control --reload-rules && sudo udevadm trigger
    ```
11. Try running the `yolo_webcam.py` script to check if the symlinks work.

If the symlinks aren't working, you can these things:

1. Add your user to the video group:
   ```
   sudo usermod -a -G video robotoo
   ```
2. Change the permissions of the `/dev/video*` devices (this is a last resort, it will change the permissions of the `/dev/video*` devices to allow all users to access them.
   ```
   sudo chmod a+rw /dev/video*
   ```
   This change is temporary and will be reset when the devices are reconnected or the system is rebooted.

Here are some of the debugging steps we used to create the symlinks for the first time:

- Use udevadm to test the rules:
  1. Run `lsof /dev/video*` to check what the device paths of the cameras are.
  2. Use udevadm to test the rules:
     ```
     sudo udevadm test $(udevadm info -q path -n /dev/your_device)
     ```
     This command will simulate the addition of the device and show you what rules are applied.
- Use udevadm to check whether the symlinks exist:
  1. Using `camerazonea` for example:
     ```
     udevadm info -q path -n /dev/camerazonea
     ```
     If this outputs `Unknown device "/dev/camerazonea": No such device` then the symlink does not work.

## References
https://blog.roboflow.com/how-to-count-objects-in-a-zone/
 https://github.com/weirros/yolov5_wi_pi4 

https://medium.com/@elvenkim1/how-to-run-yolov5-successfully-on-raspberry-pi-3f90db518e25
