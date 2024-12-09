tabloo selfdriving robot readme

on Linux machine:
https://www.youtube.com/watch?v=XKIm_R_rIeQ

#installing everything needed for python

sudo apt install python3 //installing python
sudo apt install python3-pip //installing pip
sudo apt install python3-venv //installing python virtual environments


#setting up the virtual environment

python3 -m venv --system-site-packages "environment_name" //creating a virtual environment in the current directory with the name "environment_name"
source "environment _name"/bin/activate //jumping into the created virtual environment ("environment_name") will show up at the left of the command line


#making sure the package manager is up to date once jumped into the virtual environment

sudo apt update //update packages
sudo apt upgrade //upgrading packages
sudo apt install python3-pip -y //install pip (automatically skipping confirmation prompts) 
pip install -U pip //upgrading pip


#installing the ultralytics package (yolo, OpenCV...) in the virtual environment (you have to be jumped into the virtual environment

pip install ultralytics[export] //installing ultralytics library with extra libraries to export YOLO models to different formats

if this installation gives the following error: 
AttributeError: module 'pkgutil' has no attribute 'ImpImporter'. Did you mean: 'zipimporter'?
you have to make a python virtual environment with a previous version of python (3.11)

deactivate //deactivate the current virtual environment
rm -rf "environment_name" //remove the current virtual environment

sudo add-apt-repository ppa:deadsnakes/ppa //adding the deadsnakes ppa to avoid python3.11 installation errors
sudo apt update //update package list
sudo apt install python3.11 python3.11-venv python3.11-dev //installing python3.11 including virtual environments etc
python3.11 -m venv "environment_name" //making the python3.11 virtual environment, make sure to choose a different name if you did not delete the 3.12 environment.
source "environment_name"/bin/activate //activate the virtual environment
python3.11 -m ensurepip --upgrade //install pip
pip install -U pip //check installation
pip install ultralytics[export] //installing ultralytics libraries including dependencies


#install ssh server to ssh into the pi from an external device

sudo apt install openssh-server //install ssh server on the pi
systemctl status ssh //checking if the server is running
ip a //getting the ip address of the pi

(on the external machine): 
ssh "ip_address" //ssh into the pi (you have to enter the password of the pi)
exit //exit the ssh session


#installing supervision

pip install supervision //do this while in virtual environment, supervision is used to draw polygons onto the image feed