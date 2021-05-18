from sensor_msgs.msg import Image
import cv2
import pickle
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
        self.camera16 = None
        self.camera17 = None
        self.camera18 = None
        self.camera19 = None
        self.camera20 = None
        self.camera21 = None
        self.camera22 = None
        self.camera23 = None
        self.camera24 = None

        self.br = CvBridge()

        global K1,D1
        K1 = np.array([[269.7977676690669, 0.0, 539.4244485376425],  
                        [0.0, 269.7474290400939, 359.5466822104538],
                        [0.0, 0.0, 1.0]])
        D1 = np.array([[0.08470212784201793], [0.00840973220058839], [-0.003227013248923447], [0.0031067477482508775]])

        # Cameras
        # Ready
        # to
        # use

        # Mls[[571.86553091   0.         479.2987878]
        # [0.
        # 571.85309943
        # 359.19367409]
        # [0.           0.           1.]]
        #
        # Mrs[[572.01807905   0.         479.5396884]
        # [0.
        # 572.0419174
        # 359.39838091]
        # [0.           0.           1.]]

        # Cameras
        # Ready
        # to
        # use
        # Mls[[572.17647393   0.         478.96299565]
        # [0.
        # 572.18235571
        # 359.12679712]
        # [0.           0.           1.]]
        #
        # Mrs[[571.47628153   0.         479.37913581]
        # [0.
        # 571.45085247
        # 359.15989811]
        # [0.           0.           1.]]

        #publishers
        self.pub = rospy.Publisher("/camera/read_image",Image,queue_size=10)

        #subscribers
        self.sub_camera1 = rospy.Subscriber("/warehouse/camera1/image_raw",Image,self.callback_camera1)
        self.sub_camera2 = rospy.Subscriber("/warehouse/camera2/image_raw",Image,self.callback_camera2)
        self.sub_camera3 = rospy.Subscriber("/warehouse/camera3/image_raw",Image,self.callback_camera3)
        self.sub_camera4 = rospy.Subscriber("/warehouse/camera4/image_raw",Image,self.callback_camera4)
        self.sub_camera5 = rospy.Subscriber("/warehouse/camera5/image_raw",Image,self.callback_camera5)
        self.sub_camera6 = rospy.Subscriber("/warehouse/camera6/image_raw",Image,self.callback_camera6)
        self.sub_camera7 = rospy.Subscriber("/warehouse/camera7/image_raw",Image,self.callback_camera7)
        self.sub_camera8 = rospy.Subscriber("/warehouse/camera8/image_raw",Image,self.callback_camera8)
        self.sub_camera9 = rospy.Subscriber("/warehouse/camera9/image_raw",Image,self.callback_camera9)
        self.sub_camera10 = rospy.Subscriber("/warehouse/camera10/image_raw",Image,self.callback_camera10)
        self.sub_camera11 = rospy.Subscriber("/warehouse/camera11/image_raw",Image,self.callback_camera11)
        self.sub_camera12 = rospy.Subscriber("/warehouse/camera12/image_raw",Image,self.callback_camera12)
        self.sub_camera13 = rospy.Subscriber("/warehouse/camera13/image_raw",Image,self.callback_camera13)
        self.sub_camera14 = rospy.Subscriber("/warehouse/camera14/image_raw",Image,self.callback_camera14)
        self.sub_camera15 = rospy.Subscriber("/warehouse/camera15/image_raw",Image,self.callback_camera15)
        self.sub_camera16 = rospy.Subscriber("/warehouse/camera16/image_raw", Image, self.callback_camera16)
        self.sub_camera17 = rospy.Subscriber("/warehouse/camera17/image_raw", Image, self.callback_camera17)
        self.sub_camera18 = rospy.Subscriber("/warehouse/camera18/image_raw", Image, self.callback_camera18)
        self.sub_camera19 = rospy.Subscriber("/warehouse/camera19/image_raw", Image, self.callback_camera19)
        self.sub_camera20 = rospy.Subscriber("/warehouse/camera20/image_raw", Image, self.callback_camera20)
        self.sub_camera21 = rospy.Subscriber("/warehouse/camera21/image_raw", Image, self.callback_camera21)
        self.sub_camera22 = rospy.Subscriber("/warehouse/camera22/image_raw", Image, self.callback_camera22)
        self.sub_camera23 = rospy.Subscriber("/warehouse/camera23/image_raw", Image, self.callback_camera23)
        self.sub_camera24 = rospy.Subscriber("/warehouse/camera24/image_raw", Image, self.callback_camera24)

    def callback_camera1(self, img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera1 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error camera1: {0}".format(e))

    def callback_camera2(self, img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera2 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error camera2: {0}".format(e))

    def callback_camera3(self, img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera3 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error camera3: {0}".format(e))
    
    def callback_camera4(self, img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera4 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error camera4: {0}".format(e))

    def callback_camera5(self, img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera5 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error camera5: {0}".format(e))

    def callback_camera6(self, img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera6 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error camera6: {0}".format(e))

    def callback_camera7(self, img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera7 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error camera7: {0}".format(e))

    def callback_camera8(self, img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera8 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error camera8: {0}".format(e))

    def callback_camera9(self, img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera9 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error camera9: {0}".format(e))

    def callback_camera10(self, img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera10 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error camera10: {0}".format(e))

    def callback_camera11(self, img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera11 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error camera11: {0}".format(e))

    def callback_camera12(self, img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera12 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error camera12: {0}".format(e))

    def callback_camera13(self, img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera13 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error camera13: {0}".format(e))

    def callback_camera14(self, img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera14 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error camera14: {0}".format(e))

    def callback_camera15(self, img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera15 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error camera15: {0}".format(e))

    def callback_camera16(self, img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera16 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error camera16: {0}".format(e))

    def callback_camera17(self, img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera17 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error camera17: {0}".format(e))

    def callback_camera18(self, img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera18 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error camera18: {0}".format(e))

    def callback_camera19(self, img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera19 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error camera19: {0}".format(e))

    def callback_camera20(self, img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera20 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error camera20: {0}".format(e))

    def callback_camera21(self, img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera21 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error camera21: {0}".format(e))

    def callback_camera22(self, img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera22 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError  as e:
            rospy.logerr("CvBridge Error camera22: {0}".format(e))

    def callback_camera23(self, img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera23 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error camera23: {0}".format(e))

    def callback_camera24(self, img_msg):
        # rospy.loginfo("reading video.....")
        try:
            self.camera24 = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error camera24: {0}".format(e))



    def nothing(self, x):
        pass



    def image_analyis(self):
        left_map = "left_map"
        right_map = "right_map"
        w_left = open(left_map, 'rb')  # Open the file
        w_right = open(right_map, 'rb')  # Open the file
        left_stereo_map = pickle.load(w_left)  # Assign the recreated object to bok
        right_stereo_map = pickle.load(w_right)  # Assign the recreated object to bok
        Left_Stereo_Map_x = left_stereo_map[0]
        Left_Stereo_Map_y = left_stereo_map[1]
        Right_Stereo_Map_x = right_stereo_map[0]
        Right_Stereo_Map_y = right_stereo_map[1]

        # cv_file = cv2.FileStorage("data/stereo_rectify_maps.xml", cv2.FILE_STORAGE_READ)
        # Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
        # Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
        # Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
        # Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
        # cv_file.release()

        kernel = np.ones((3, 3), np.uint8)


        cv_file = cv2.FileStorage("data/depth_estimation_params_py.xml", cv2.FILE_STORAGE_READ)
        # numDisparities = int(cv_file.getNode("numDisparities").real())
        # blockSize = int(cv_file.getNode("blockSize").real())
        # # preFilterType = int(cv_file.getNode("preFilterType").real())
        # # preFilterSize = int(cv_file.getNode("preFilterSize").real())
        # preFilterCap = int(cv_file.getNode("preFilterCap").real())
        # # textureThreshold = int(cv_file.getNode("textureThreshold").real())
        # uniquenessRatio = int(cv_file.getNode("uniquenessRatio").real())
        # speckleRange = int(cv_file.getNode("speckleRange").real())
        # speckleWindowSize = int(cv_file.getNode("speckleWindowSize").real())
        # disp12MaxDiff = int(cv_file.getNode("disp12MaxDiff").real())
        # minDisparity = int(cv_file.getNode("minDisparity").real())
        M = cv_file.getNode("M").real()
        cv_file.release()

        while not rospy.is_shutdown():
            #put all the factory_imgs in an array
            active = np.array([(self.camera1 > 0), (self.camera2 > 0)])

            cond = active.shape    # get the shape... if an image isn't prest len(cond) = 1
            if len(cond) == 4:

                camera1_img = self.camera1
                camera2_img = self.camera2
                # camera3_img = self.camera3
                # camera4_img = self.camera4
                # camera5_img = self.camera5
                # camera6_img = self.camera6
                # camera7_img = self.camera7
                # camera8_img = self.camera8
                # camera9_img = self.camera9
                # camera10_img = self.camera10
                # camera11_img = self.camera11
                # camera12_img = self.camera12
                # camera13_img = self.camera13
                # camera14_img = self.camera14
                # camera15_img = self.camera15
                # camera16_img = self.camera16
                # camera17_img = self.camera17
                # camera18_img = self.camera18
                # camera19_img = self.camera19
                # camera20_img = self.camera20
                # camera21_img = self.camera21
                # camera22_img = self.camera22
                # camera23_img = self.camera23
                # camera24_img = self.camera24

                # preFilterType =
                # preFilterSize =
                # textureThreshold =

                blockSize = 5
                preFilterCap = 0
                uniquenessRatio = 5
                speckleRange = 5
                speckleWindowSize = 5
                disp12MaxDiff = 20
                maxDisparity = 63
                minDisparity = -1
                numDisparities = maxDisparity - minDisparity
                P1 = 8 * 3 * blockSize ** 2
                P2 = 32 * 3 * blockSize ** 2

                # numDisparities = 48
                # blockSize = 15
                # preFilterCap = 5
                # uniquenessRatio = 15
                # speckleRange = 14
                # speckleWindowSize = 6
                # disp12MaxDiff = 10
                # minDisparity = 7
                # P1 = 5400
                # P2 = 21600
                # P1 = 8 * 3 * window_size ** 2,
                # P2 = 32 * 3 * window_size ** 2


                print("working")
                # Create StereoSGBM and prepare all parameters
                window_size = blockSize
                min_disp = minDisparity
                num_disp = numDisparities
                pre_cap = preFilterCap
                # pp1 = 8 * 3 * window_size ** 2
                # pp2 = 32 * 3 * window_size ** 2

                # print(minDisparity,numDisparities,blockSize,uniquenessRatio,preFilterCap,speckleWindowSize,speckleRange,disp12MaxDiff,pp1,pp2)

                stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                               numDisparities=num_disp,
                                               blockSize=window_size,
                                               uniquenessRatio=uniquenessRatio,
                                               preFilterCap=pre_cap,
                                               speckleWindowSize=speckleWindowSize,
                                               speckleRange=speckleRange,
                                               disp12MaxDiff=disp12MaxDiff,
                                               P1=P1,
                                               P2=P2)

                # Used for the filtered image
                stereoR = cv2.ximgproc.createRightMatcher(stereo)  # Create another stereo for right this time

                # WLS FILTER Parameters
                lmbda = 80000
                sigma = 1.8
                visual_multiplier = 1.0

                wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
                wls_filter.setLambda(lmbda)
                wls_filter.setSigmaColor(sigma)

                frameR = camera1_img
                frameL = camera2_img

                #
                # Left_nice = cv2.remap(frameL, left_stereo_map[0], left_stereo_map[1], cv2.INTER_LANCZOS4,cv2.BORDER_CONSTANT,0)  # Rectify the image using the kalibration parameters founds during the initialisation
                # Right_nice = cv2.remap(frameR, right_stereo_map[0], right_stereo_map[1], cv2.INTER_LANCZOS4,cv2.BORDER_CONSTANT, 0)

                Left_nice = cv2.remap(frameL, Left_Stereo_Map_x,Left_Stereo_Map_y, cv2.INTER_LANCZOS4,cv2.BORDER_CONSTANT,0)  # Rectify the image using the calibration parameters founds during the initialisation
                Right_nice = cv2.remap(frameR, Right_Stereo_Map_x, Right_Stereo_Map_y, cv2.INTER_LANCZOS4,cv2.BORDER_CONSTANT, 0)


                focal_length = np.load('camera_params/FocalLength.npy')
                print(focal_length,M)
                # Convert from color(BGR) to gray
                grayR = cv2.cvtColor(Right_nice, cv2.COLOR_BGR2GRAY)
                grayL = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2GRAY)

                # Compute the 2 factory_imgs for the Depth_image
                disp = stereo.compute(grayL, grayR)  # .astype(np.float32)/ 16
                dispL = disp
                dispR = stereoR.compute(grayR, grayL)
                dispL = np.int16(dispL)
                dispR = np.int16(dispR)

                # Using the WLS filter
                filteredImg1 = wls_filter.filter(dispL, grayL, None, dispR)
                filteredImg1 = cv2.normalize(src=filteredImg1, dst=filteredImg1, beta=0, alpha=255,norm_type=cv2.NORM_MINMAX);
                filteredImg1 = np.uint8(filteredImg1)


                # cv2.imshow('Disparity Map', filteredImg)
                disp = ((disp.astype(np.float32) / 16) - min_disp) / num_disp  # Calculation allowing us to have 0 for the most distant object able to detect

                ##    # Resize the image for faster executions
                ##    dispR= cv2.resize(disp,None,fx=0.7, fy=0.7, interpolation = cv2.INTER_AREA)

                # Filtering the Results with a closing filter
                closing = cv2.morphologyEx(disp, cv2.MORPH_CLOSE, kernel)  # Apply an morphological filter for closing little "black" holes in the picture(Remove noise)

                # Colors map
                dispc = (closing - closing.min()) * 255
                dispC = dispc.astype(np.uint8)  # Convert the type of the matrix from float32 to uint8, this way you can show the results with the function cv2.imshow()
                disp_Color = cv2.applyColorMap(dispC, cv2.COLORMAP_OCEAN)  # Change the Color of the Picture into an Ocean Color_Map
                filt_color1 = cv2.applyColorMap(filteredImg1, cv2.COLORMAP_OCEAN)

                final_img = dispC   #final image to be shown
                print(final_img.shape)
                # final_img = cv2.resize(final_img,(1080,640))
                try:
                    self.pub.publish(self.br.cv2_to_imgmsg(final_img))
                except CvBridgeError as e:
                    rospy.logerr("CvBridge Error: {0}".format(e))

                cv2.imshow("camera_feed",final_img)
                if cv2.waitKey(1) & 0XFF == ord("q"):
                    break
            # self.rate.sleep()
        # print("Saving depth estimation parameters ......")
        #
        # cv_file = cv2.FileStorage("data/depth_estimation_params_py.xml", cv2.FILE_STORAGE_WRITE)
        # cv_file.write("numDisparities", numDisparities)
        # cv_file.write("blockSize", blockSize)
        # # cv_file.write("preFilterType", preFilterType)
        # # cv_file.write("preFilterSize", preFilterSize)
        # # cv_file.write("preFilterCap", preFilterCap)
        # # cv_file.write("textureThreshold", textureThreshold)
        # cv_file.write("uniquenessRatio", uniquenessRatio)
        # cv_file.write("speckleRange", speckleRange)
        # cv_file.write("speckleWindowSize", speckleWindowSize)
        # cv_file.write("disp12MaxDiff", disp12MaxDiff)
        # cv_file.write("minDisparity", minDisparity)
        # cv_file.write("M", 39.075)
        # cv_file.release()
    
def main(args):
    rospy.init_node('cam_node', anonymous=True)
    warehouse = image_stitch()
    warehouse.image_analyis()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)