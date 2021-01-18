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
        self.camera6 = None
        self.camera7 = None
        self.camera8 = None
        self.camera9 = None
        self.camera10 = None
        self.camera11 = None
        self.camera12 = None
        self.camera13 = None
        self.camera14 = None
        self.camera15 = None
        self.br = CvBridge()
        global K1,D1
        K1 = np.array([[269.7977676690669, 0.0, 539.4244485376425],  
                        [0.0, 269.7474290400939, 359.5466822104538],
                        [0.0, 0.0, 1.0]])
        D1 = np.array([[0.08470212784201793], [0.00840973220058839], [-0.003227013248923447], [0.0031067477482508775]])
        
        self.K = np.array([[270.1132079536561, 0.0, 539.4278422993344], 
                            [0.0, 270.12435793063867, 359.65772231968793],
                            [0.0, 0.0, 1.0]])

        self.D = np.array([[0.0838650200838641], [0.004369973071477834], [0.006062524916889637], [-0.0019177722768309521]])
     
        #publishers
        self.pub = rospy.Publisher("/camera/read_image",Image,queue_size=10)

        #subscribers
        self.sub_camera1 = rospy.Subscriber("/factory_cam/camera1/image_raw",Image,self.callback_camera1)   
        self.sub_camera2 = rospy.Subscriber("/factory_cam/camera2/image_raw",Image,self.callback_camera2)    
        self.sub_camera3 = rospy.Subscriber("/factory_cam/camera3/image_raw",Image,self.callback_camera3)
        self.sub_camera4 = rospy.Subscriber("/factory_cam/camera4/image_raw",Image,self.callback_camera4) 
        self.sub_camera5 = rospy.Subscriber("/factory_cam/camera5/image_raw",Image,self.callback_camera5)   
        self.sub_camera6 = rospy.Subscriber("/factory_cam/camera6/image_raw",Image,self.callback_camera6) 
        self.sub_camera7 = rospy.Subscriber("/factory_cam/camera1/image_raw",Image,self.callback_camera7)   
        self.sub_camera8 = rospy.Subscriber("/factory_cam/camera8/image_raw",Image,self.callback_camera8)    
        self.sub_camera9 = rospy.Subscriber("/factory_cam/camera9/image_raw",Image,self.callback_camera9)
        self.sub_camera10 = rospy.Subscriber("/factory_cam/camera10/image_raw",Image,self.callback_camera10) 
        self.sub_camera11 = rospy.Subscriber("/factory_cam/camera11/image_raw",Image,self.callback_camera11)   
        self.sub_camera12 = rospy.Subscriber("/factory_cam/camera12/image_raw",Image,self.callback_camera12)  
        self.sub_camera13 = rospy.Subscriber("/factory_cam/camera13/image_raw",Image,self.callback_camera13) 
        self.sub_camera14 = rospy.Subscriber("/factory_cam/camera14/image_raw",Image,self.callback_camera14)   
        self.sub_camera15 = rospy.Subscriber("/factory_cam/camera15/image_raw",Image,self.callback_camera15)
    


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

    def callback_camera6(self,img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera6 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError, e:
            rospy.logerr("CvBridge Error camera6: {0}".format(e))

    def callback_camera7(self,img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera7 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError, e:
            rospy.logerr("CvBridge Error camera7: {0}".format(e))


    def callback_camera8(self,img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera8 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError, e:
            rospy.logerr("CvBridge Error camera8: {0}".format(e))

    def callback_camera9(self,img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera9 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError, e:
            rospy.logerr("CvBridge Error camera9: {0}".format(e))

    def callback_camera10(self,img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera10 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError, e:
            rospy.logerr("CvBridge Error camera10: {0}".format(e))

    def callback_camera11(self,img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera11 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError, e:
            rospy.logerr("CvBridge Error camera11: {0}".format(e))

    def callback_camera12(self,img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera12 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError, e:
            rospy.logerr("CvBridge Error camera12: {0}".format(e))

    def callback_camera13(self,img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera13 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError, e:
            rospy.logerr("CvBridge Error camera13: {0}".format(e))

    def callback_camera14(self,img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera14 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError, e:
            rospy.logerr("CvBridge Error camera14: {0}".format(e))

    def callback_camera15(self,img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera15 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError, e:
            rospy.logerr("CvBridge Error camera15: {0}".format(e))


    def undistort(self,img,K,D):
        balance = 0.0
        dim = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort

        # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, dim, np.eye(3), balance=balance)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, dim, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return undistorted_img



    def match(self,img1,img2):
        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)

        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])

        # cv2.drawMatchesKnn expects list of lists as matches.
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
        return img3,good


    

    def image_analyis(self):
        # rospy.loginfo("Timing images")
        #rospy.spin()
       
        while not rospy.is_shutdown():
            #put all the images in an array
            active = np.array([(self.camera1 > 0),(self.camera2 > 0) , (self.camera3 > 0), \
            (self.camera4 > 0), (self.camera5 > 0),(self.camera6 > 0),(self.camera7 > 0) , (self.camera8 > 0), \
            (self.camera4 > 0), (self.camera5 > 0),(self.camera6 > 0),(self.camera7 > 0) , (self.camera8 > 0), \
            (self.camera9 > 0), (self.camera10 > 0),(self.camera11 > 0),(self.camera12 > 0) , (self.camera13 > 0), \
            (self.camera14 > 0),(self.camera15 > 0)])
            
            cond = active.shape    # get the shape... if an image isn't prest len(cond) = 1
            if len(cond) == 4:
                camera1_img = self.camera1
                camera2_img = self.camera2
                camera3_img = self.camera3
                camera4_img = self.camera4
                camera5_img = self.camera5
                camera6_img = self.camera6
                camera7_img = self.camera7
                camera8_img = self.camera8
                camera9_img = self.camera9
                camera10_img = self.camera10
                camera11_img = self.camera11
                camera12_img = self.camera12
                camera13_img = self.camera13
                camera14_img = self.camera14
                camera15_img = self.camera15

                
                #undistort images
                dist_camera1 = self.undistort(camera1_img,K1,D1)
                dist_camera2 = self.undistort(camera2_img,K1,D1)
                dist_camera3 = self.undistort(camera3_img,K1,D1)
                dist_camera4 = self.undistort(camera4_img,K1,D1)
                dist_camera5 = self.undistort(camera5_img,K1,D1)
                dist_camera6 = self.undistort(camera6_img,K1,D1)
                dist_camera7 = self.undistort(camera7_img,K1,D1)
                dist_camera8 = self.undistort(camera8_img,K1,D1)
                dist_camera9 = self.undistort(camera9_img,K1,D1)
                dist_camera10 = self.undistort(camera10_img,K1,D1)
                dist_camera11 = self.undistort(camera11_img,K1,D1)
                dist_camera12= self.undistort(camera12_img,K1,D1)
                dist_camera13 = self.undistort(camera13_img,K1,D1)
                dist_camera14 = self.undistort(camera14_img,K1,D1)
                dist_camera15 = self.undistort(camera15_img,K1,D1)
                
                # print("start")
                #match images in along cameras
                # match_img,found_matches = self.match(dist_camera1,dist_camera2)
                # print(len(found_matches))

                cv2.imwrite("factory_imgs/camera1.png",dist_camera1)
                cv2.imwrite("factory_imgs/camera2.png",dist_camera2)
                # cv2.imwrite("factory_imgs/camera3.png",dist_camera3)
                # cv2.imwrite("factory_imgs/camera4.png",dist_camera4)
                # cv2.imwrite("factory_imgs/camera5.png",dist_camera5)
                # cv2.imwrite("factory_imgs/camera6.png",dist_camera6)
                # cv2.imwrite("factory_imgs/camera7.png",dist_camera7)
                # cv2.imwrite("factory_imgs/camera8.png",dist_camera8)
                # cv2.imwrite("factory_imgs/camera9.png",dist_camera9)
                # cv2.imwrite("factory_imgs/camera10.png",dist_camera10)
                # cv2.imwrite("factory_imgs/camera11.png",dist_camera11)
                # cv2.imwrite("factory_imgs/camera12.png",dist_camera12)
                # cv2.imwrite("factory_imgs/camera13.png",dist_camera13)
                # cv2.imwrite("factory_imgs/camera14.png",dist_camera14)
                # cv2.imwrite("factory_imgs/camera15.png",dist_camera15)
               


                print("done")
                break

                final_img = dist_camera8        #final image to be shown
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
    factory_cam = image_stitch()
    factory_cam.image_analyis()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)