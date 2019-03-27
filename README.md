# LINKS:

Github: https://github.com/Apiquet/Visual_recognition

My website: https://anthonypiquet.wordpress.com/2019/02/10/visual-recognition/


# DESCRIPTION:

It's just the beginning of the project. I'm participating at the EPFL Robotics Competition (15 students selected, separated in group of 3) where we must build a standalone robot 
from scratch which recognizes any kind of bottles, catch it, then bring it to a recycling bin. It must overcome obstacles like rocks, take ramps, run over grass, etc. 

I'm in charge of the collector (front of the robot to catch and store bottles) and of the bottle recognition. I spent only few days on the bottle recognition because the priority
is to have a built robot, then I will move forward to the bottle recognition. So far, the bottle recognition takes around 0.6s with the lowest detection efficiency provided by the
library I'm using, that's not enough as we need a frequency of at least 1Hz so 1 detection every 0.1s. I will move to OpenCV and build something faster as soon as the robot is
built (in 2 weeks maximum). I will have a big recognition project to show at this time if this project caught your attention. So far, it's mainly the use of a library without
any algorithmic skill shown. I put this project in case you are interested in seeing the future project in few weeks.



# HOW TO RUN:

You may need to install the package: imageai
	

Simply run: main.py

Then, once your camera is turned on, put any object in front of it and press S key to take a screenshot and to display the object detection result.
Press Q key to exit the script.