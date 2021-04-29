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

![warehouse](https://user-images.githubusercontent.com/17696533/116504020-714e3680-a885-11eb-96c3-52d7c082bdf2.PNG)

# Progress as of now
# Process:<br/>
- Stereo camera setup and calibration; setting cameras in warehouse environment.<br/>
![image](https://user-images.githubusercontent.com/17696533/115128844-b84b4a80-9fae-11eb-80f8-0c590d70603b.png)
<br/>

- Depth segmentation based on environment and camera height.<br/>
![stereo_2](https://user-images.githubusercontent.com/17696533/116504262-e6217080-a885-11eb-8322-dadc6b919961.PNG)
![image](https://user-images.githubusercontent.com/17696533/115128745-014ecf00-9fae-11eb-9fc8-6a55b6080a24.png)
<br/>

- Finding homography between images of the different camera setups for image stitching
![image](https://user-images.githubusercontent.com/17696533/116504129-a78bb600-a885-11eb-8b02-2e27261ed68d.png)

- Mask stitching comes out a bit noisy and will need additional noise reduction.
![image](https://user-images.githubusercontent.com/17696533/115617226-277eb280-a2bf-11eb-9ef9-8874efe0558a.png)<br/>

- Path generation. Image dimensionality is reduced for path planning to speed up process.
![image](https://user-images.githubusercontent.com/17696533/115617718-bc81ab80-a2bf-11eb-84a2-08d6256c8697.png)
![image](https://user-images.githubusercontent.com/17696533/115618107-3c0f7a80-a2c0-11eb-8e06-e896ed7aacdb.png)
![image](https://user-images.githubusercontent.com/17696533/116504798-2fbe8b00-a887-11eb-8bb0-ea0a0320dcb7.png)
![image](https://user-images.githubusercontent.com/17696533/116504669-d9e9e300-a886-11eb-88b1-38fd3e8b9444.png)


