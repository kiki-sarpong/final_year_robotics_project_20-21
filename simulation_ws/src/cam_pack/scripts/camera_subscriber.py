from sensor_msgs.msg import Image
import cv2
import numpy as np
import rospy
import sys
from cv_bridge import CvBridge, CvBridgeError

class read_cam(object):
    def __init__(self):
        self.image1 = None
        self.image6 = None
        self.br = CvBridge()
        # self.loop_rate = rospy.Rate(1)
        # self.rate = rospy.Rate(1)

        #publishers
        self.pub = rospy.Publisher("/camera/read_cam_image",Image,queue_size=10)

        #subscribers
        self.sub1 = rospy.Subscriber("/warehouse_cam/camera1/image_raw",Image,self.callback1)   #cam1
        self.sub6 = rospy.Subscriber("/warehouse_cam/camera6/image_raw",Image,self.callback6) #camera6 is adjacent to camera1
    
    def callback1(self,img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.image1 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError, e:
            rospy.logerr("CvBridge Error cam1: {0}".format(e))

    
    def callback6(self,img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.image6 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError, e:
            rospy.logerr("CvBridge Error cam6: {0}".format(e))

    def image_analyis(self):
        # rospy.loginfo("Timing images")
        #rospy.spin()
       
        while not rospy.is_shutdown():
            if self.image1 is not None: 
                # cv2.circle(self.image,(100,100),6,(0,0,255))      #draw a random circle for image manipulation test
                gray_img1 = cv2.cvtColor(self.image1,cv2.COLOR_BGR2GRAY)
                gray_img6 = cv2.cvtColor(self.image1,cv2.COLOR_BGR2GRAY)

                orb = cv2.ORB_create()
                kp1, des1 = orb.detectAndCompute(gray_img1, None)
                kp6, des6 = orb.detectAndCompute(gray_img6, None)

                # matcher takes normType, which is set to cv2.NORM_L2 for SIFT and SURF, cv2.NORM_HAMMING for ORB, FAST and BRIEF
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des6)
                matches = sorted(matches, key=lambda x: x.distance)
                match_img = cv2.drawMatches(gray_img1, kp1, gray_img6, kp6, matches[:20], None)



                final_img = match_img           #final image to be shown
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
    warehouse_cam = read_cam()
    warehouse_cam.image_analyis()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
                
