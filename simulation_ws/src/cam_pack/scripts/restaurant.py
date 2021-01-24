from sensor_msgs.msg import Image
import cv2
import numpy as np
import rospy
import sys
from cv_bridge import CvBridge, CvBridgeError

class image_stitch(object):
    def __init__(self):
        self.camera1 = None
        self.camera2 = None
        self.camera3 = None
        self.camera4 = None
        self.camera5 = None


        self.br = CvBridge()

        global K1,D1
        K1 = np.array([[269.7977676690669, 0.0, 539.4244485376425],  
                        [0.0, 269.7474290400939, 359.5466822104538],
                        [0.0, 0.0, 1.0]])
        D1 = np.array([[0.08470212784201793], [0.00840973220058839], [-0.003227013248923447], [0.0031067477482508775]])

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



    def undistort(self,img,K,D):
        balance = 0.0
        dim = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort

        # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, dim, np.eye(3), balance=balance)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, dim, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return undistorted_img




    def image_analyis(self):

        while not rospy.is_shutdown():
            #put all the images in an array
            # active = np.array([(self.camera1 > 0),(self.camera2 > 0) , (self.camera3 > 0),
            # (self.camera4 > 0), (self.camera5 > 0),(self.camera6 > 0),(self.camera7 > 0) , (self.camera8 > 0),
            # (self.camera4 > 0), (self.camera5 > 0)])

            active = np.array([(self.camera5 > 0)])
            cond = active.shape    # get the shape... if an image isn't prest len(cond) = 1
            if len(cond) == 4:
                # camera1_img = self.camera1
                # camera2_img = self.camera2
                # camera3_img = self.camera3
                # camera4_img = self.camera4
                camera5_img = self.camera5

                
                #undistort images
                # dist_camera1 = self.undistort(camera1_img,K1,D1)
                # dist_camera2 = self.undistort(camera2_img,K1,D1)
                # dist_camera3 = self.undistort(camera3_img,K1,D1)
                # dist_camera4 = self.undistort(camera4_img,K1,D1)
                dist_camera5 = self.undistort(camera5_img,K1,D1)


                # cv2.imwrite("factory_imgs/camera1.png",dist_camera1)
                # cv2.imwrite("factory_imgs/camera2.png",dist_camera2)
                # cv2.imwrite("factory_imgs/camera3.png",dist_camera3)
                # cv2.imwrite("factory_imgs/camera4.png",dist_camera4)
                # cv2.imwrite("factory_imgs/camera5.png",dist_camera5)

                cv2.imwrite("scaling_2_img.png", dist_camera5)


                print("done")
                break

                final_img = dist_camera5        #final image to be shown
                # final_img = cv2.resize(final_img,(720,640))
                try:
                    self.pub.publish(self.br.cv2_to_imgmsg(final_img))
                except CvBridgeError, e:
                    rospy.logerr("CvBridge Error: {0}".format(e))

                cv2.imshow("camera_feed",final_img)
                if cv2.waitKey(1) & 0XFF ==ord("q"):
                    break
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