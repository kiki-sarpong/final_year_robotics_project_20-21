# Final year robotics project 20-21

## Final year project topic: Vision System For Warehouse Robotics

Final project ROS and Gazebo files<br/>

# Brief project description:<br/>

A Centralized system using multiple stereo cameras placed on the ceiling of a warehouse looking down to give real-time mapping of an indoor environment<br/>
An image from each stereo camera is stitched across multiple cameras to get a bird's eye view overview of the entire environment.<br/>
A depth segmentation is ran to separate free pathways from obstructed pathways.<br/>
The segmented pathways in the multiple images are stitched together to get a complete outline of free space in the indoor environment.<br/>
A path planning algorithm is ran to plan paths for the warehouse robots to move materials from one location to the other.<br/>
This project is done entirely in simulation due to COVID
Project will be simulated on ROS and Gazebo. <br/><br/>

# Warehouse environment <br/>
![image](https://user-images.githubusercontent.com/17696533/115128526-55f14a80-9fac-11eb-8f2a-5a5d5418afd9.png)

![image](https://user-images.githubusercontent.com/17696533/115128531-5be72b80-9fac-11eb-9bc2-9e00e76177a9.png)

# Progress as of now
# Process:<br/>
- Stereo camera setup and calibration; setting cameras in warehouse environment.<br/>
![image](https://user-images.githubusercontent.com/17696533/115128844-b84b4a80-9fae-11eb-80f8-0c590d70603b.png)
<br/>

- Depth segmentation based on environment and camera height.<br/>
![image](https://user-images.githubusercontent.com/17696533/115128745-014ecf00-9fae-11eb-9fc8-6a55b6080a24.png)
<br/>

- Finding homography between images of the different camera setups for image stitching
![image](https://user-images.githubusercontent.com/17696533/115128892-06604e00-9faf-11eb-9af2-3cf35aa44027.png)
<br/>

- Path generation.
![image](https://user-images.githubusercontent.com/17696533/115129037-45db6a00-9fb0-11eb-9500-0758228fd73b.png)


