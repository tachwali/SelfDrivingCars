#!/usr/bin/env sh
pip uninstall --yes Pillow
pip install Pillow
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch