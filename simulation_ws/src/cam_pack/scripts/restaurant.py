from sensor_msgs.msg import Image
import cv2
import numpy as np
import rospy
import sys
from cv_bridge import CvBridge, CvBridgeError
import time

class image_stitch(object):
    def __init__(self):
        self.camera1 = None
        self.camera2 = None
        self.camera3 = None
        self.camera4 = None
        self.camera5 = None
        self.br = CvBridge()
        self.scale_x = 41.5      #1 meter in real world coordinates
        self.scale_y = 41         #1 meter in real world coordinates
        self.img_h= 2000
        self.img_w = 2000
        self.masked_imgs = []
        self.base_image = np.zeros((self.img_h,self.img_w,3),dtype=np.uint8)   #create image
        # camera positions
        # cam positions must be in same order used in xacro or urdf file
        self.camera_positions = np.array( [(4, 4), (-4, 4), (-4, -4), (4, -4), (0, 0)])  # camera locations in xacro files(joints)
        # the top of the camera faces the x direction in gazebo
        # multiply by -1 and fliplr to align image and camera
        self.cam_locations = -1*self.camera_positions*(self.scale_x,self.scale_y) + (self.img_w/2,self.img_h/2)
        self.cam_locations = np.fliplr(self.cam_locations)


        self.K = np.array([[269.7977676690669, 0.0, 539.4244485376425],
                        [0.0, 269.7474290400939, 359.5466822104538],
                        [0.0, 0.0, 1.0]])
        self.D = np.array([[0.08470212784201793], [0.00840973220058839], [-0.003227013248923447], [0.0031067477482508775]])

        #publishers
        self.pub = rospy.Publisher("/camera/read_image",Image,queue_size=10)

        #subscribers
        self.sub_camera1 = rospy.Subscriber("/restaurant/camera1/image_raw",Image,self.callback_camera1)
        self.sub_camera2 = rospy.Subscriber("/restaurant/camera2/image_raw",Image,self.callback_camera2)
        self.sub_camera3 = rospy.Subscriber("/restaurant/camera3/image_raw",Image,self.callback_camera3)
        self.sub_camera4 = rospy.Subscriber("/restaurant/camera4/image_raw",Image,self.callback_camera4)
        self.sub_camera5 = rospy.Subscriber("/restaurant/camera5/image_raw",Image,self.callback_camera5)


    def callback_camera1(self,img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera1 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError, e:
            rospy.logerr("CvBridge Error camera1: {0}".format(e))

    def callback_camera2(self,img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera2 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError, e:
            rospy.logerr("CvBridge Error camera2: {0}".format(e))

    def callback_camera3(self,img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera3 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError, e:
            rospy.logerr("CvBridge Error camera3: {0}".format(e))
    
    def callback_camera4(self,img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera4 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError, e:
            rospy.logerr("CvBridge Error camera4: {0}".format(e))

    def callback_camera5(self,img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera5 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError, e:
            rospy.logerr("CvBridge Error camera5: {0}".format(e))



    def undistort(self,img):
        balance = 0.0
        dim = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort

        # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.K, self.D, dim, np.eye(3), balance=balance)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), new_K, dim, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return undistorted_img

    def save_imgs(self,dist_camera1,dist_camera2,dist_camera3,dist_camera4,dist_camera5):
        cv2.imwrite("factory_imgs/camera1.png", dist_camera1)
        cv2.imwrite("factory_imgs/camera2.png", dist_camera2)
        cv2.imwrite("factory_imgs/camera3.png", dist_camera3)
        cv2.imwrite("factory_imgs/camera4.png", dist_camera4)
        cv2.imwrite("factory_imgs/camera5.png", dist_camera5)

    def create_full_mask(self):
        hold_imgs = []
        cam = self.masked_imgs[0] ### any img can be cam since you just need the shape and they all have the same shape

        # make sure cameras and camera locations are aligned properly.

        for (px,py),cam in zip(self.cam_locations,self.masked_imgs):
            base_copy = self.base_image.copy()
            offset_x = int(px - cam.shape[1] / 2)  # figure out the offset to get where the image is placed
            offset_y = int(py - cam.shape[0] / 2)
            y1, y2 = offset_y, offset_y + cam.shape[0]
            x1, x2 = offset_x, offset_x + cam.shape[1]
            base_copy[y1:y2, x1:x2] = cam
            hold_imgs.append(base_copy)

        add_imgs = 0
        for img in hold_imgs:
            add_imgs += img

        # get the just the scaled image
        x1 = int(self.cam_locations[0, 0] - cam.shape[1] / 2)
        y1 = int(self.cam_locations[0, 1] - cam.shape[0] / 2)
        x2 = int(self.cam_locations[2, 0] + cam.shape[1] / 2)
        y2 = int(self.cam_locations[2, 1] + cam.shape[0] / 2)

        final_img = add_imgs[y1:y2, x1:x2]
        print("Final image shape is ",final_img.shape)
        return final_img





    def image_analyis(self):
        for px, py in self.cam_locations:    # plot cam_locations base image
            px, py = int(px), int(py)
            cv2.circle(self.base_image, (px, py), 10, (0, 255, 0), -1)

        for num in range(len(self.cam_locations)):     #get masked image
            path = "segmented_imgs/"+"camera"+str(num+1)+".png"
            img = cv2.imread(path)
            self.masked_imgs.append(img)


        mask_path = "segmented_imgs/camera1.png"
        mask1 = cv2.imread(mask_path)
        mask1 = cv2.bitwise_not(mask1)
        while not rospy.is_shutdown():
            #put all the images in an array
            active = np.array([(self.camera1 > 0),(self.camera2 > 0) , (self.camera3 > 0),
            (self.camera4 > 0), (self.camera5 > 0)])

            cond = active.shape    # get the shape... if an image isn't present len(cond) = 1
            print(cond)
            if len(cond) == 4:
                camera1_img = self.camera1
                camera2_img = self.camera2
                camera3_img = self.camera3
                camera4_img = self.camera4
                camera5_img = self.camera5

                #undistort images
                dist_camera1 = self.undistort(camera1_img)
                dist_camera2 = self.undistort(camera2_img)
                dist_camera3 = self.undistort(camera3_img)
                dist_camera4 = self.undistort(camera4_img)
                dist_camera5 = self.undistort(camera5_img)

                # self.save_imgs(dist_camera1,dist_camera2,dist_camera3,dist_camera4,dist_camera5)
                # print("done")
                # break



                # mask_path = "segmented_imgs/camera5.png"
                # mask1 = cv2.imread(mask_path)
                # # mask1 = cv2.bitwise_not(mask1)
                mask_cam = cv2.bitwise_and(mask1,dist_camera5)
                start = time.time()
                final_img = self.create_full_mask()
                end = time.time()
                print(end-start,"This is the time the code takes to run")


                # final_img = self.base_image     #final image to be shown
                # final_img = cv2.resize(final_img,(720,640))
                try:
                    self.pub.publish(self.br.cv2_to_imgmsg(final_img))
                except CvBridgeError, e:
                    rospy.logerr("CvBridge Error: {0}".format(e))

                # cv2.imshow("camera_feed",final_img)
                # if cv2.waitKey(1) & 0XFF ==ord("q"):
                #     break

            # self.rate.sleep()
    
def main(args):
    rospy.init_node('cam_node', anonymous=True)
    restaurant = image_stitch()
    restaurant.image_analyis()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)