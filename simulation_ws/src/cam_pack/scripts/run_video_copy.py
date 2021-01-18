from sensor_msgs.msg import Image
import cv2
import numpy as np
import rospy
import sys
from cv_bridge import CvBridge, CvBridgeError

class read_cam(object):
    def __init__(self):
        self.image = None
        self.br = CvBridge()
        # self.loop_rate = rospy.Rate(1)
        # self.rate = rospy.Rate(1)

        #publishers
        self.pub = rospy.Publisher("/camera/read_cam_image",Image,queue_size=10)

        #subscribers
        self.sub = rospy.Subscriber("/factory_cam/camera1/image_raw",Image,self.callback)
    
    def callback(self,img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.image = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError, e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    def image_analyis(self):
        # rospy.loginfo("Timing images")
        #rospy.spin()
       
        while not rospy.is_shutdown():
            if self.image is not None:
                # cv2.circle(self.image,(100,100),6,(0,0,255))      #draw a random circle for image manipulation test
                gray_img = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
                sift = cv2.xfeatures2d.SIFT_create()
                kp,des = sift.detectAndCompute(gray_img,None)
                kp_img = cv2.drawKeypoints(self.image,kp,None,color=(0,0,255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                
                

                





                final_img = kp_img           #final image to be shown
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
    factory_cam = read_cam()
    factory_cam.image_analyis()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
                
