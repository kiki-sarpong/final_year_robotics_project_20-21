# Final year robotics project 20-21

## Final year project topic: Vision System For Warehouse Robotics

Final project ROS and Gazebo files<br/>

# Brief project description:<br/>

A Centralized system using multiple stereo cameras placed on the ceiling of a building looking down to give real-time mapping of an indoor environment<br/>
An image from each stereo camera is stitched across multiple cameras to get a bird's eye view overview of the entire environment.<br/>
A depth segmentation is ran to separate free pathways from obstructed pathways.<br/>
The segmented pathways in the multiple images are stitched together to get a complete outline of free space in the indoor environment.<br/>
A path planning algorithm is ran to plan paths for the warehouse robots to move materials from one location to the other.<br/>
This project is done entirely in simulation due to COVID
Project will be simulated on ROS and Gazebo. <br/><br/>

# Warehouse environment <br/>

# Process:<br/>
- Stereo camera setup and calibration; setting cameras in warehouse environment.
- Depth segmentation based on environment and camera height.
- Finding homography between images of the different camera setups for image stitching
