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
        #Q is from the calibration code.....stereorectify
        self.Q = np.array([[1.,      0.,       0.,   -477.64002609],
                          [0.,      1.,     0.,      -439.02316284],
                          [0.,       0.,        0.,     571.82637773],
                          [0.,       0.,          1.65088716, -0.]])
        global K1,D1
        K1 = np.array([[269.7977676690669, 0.0, 539.4244485376425],  
                        [0.0, 269.7474290400939, 359.5466822104538],
                        [0.0, 0.0, 1.0]])
        D1 = np.array([[0.08470212784201793], [0.00840973220058839], [-0.003227013248923447], [0.0031067477482508775]])

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


    def stitch_images(self,homography_list,img_array):
        b_y, b_x, stitched, translated_H = 0, 0, 0, 0
        a = img_array[0]
        for i, homography in enumerate(homography_list):
            b = img_array[i+1]

            # center of source image
            center_pt = [a.shape[1] // 2, a.shape[0] // 2, 1]

            # find where center of source image will be after warping without comepensating for any offset
            warped_pt = np.dot(homography, center_pt)
            warped_pt = [x / warped_pt[2] for x in warped_pt]
            # print("passing")
            inc = 3
            cond = True
            print("starting while")
            # while cond:
            # warping output image size # center of warping output image
            stitched_frame_size = tuple(int(inc * x) for x in a.shape[:2])
            w, h = stitched_frame_size
            if len(a.shape) == 3:
                im_copy = np.zeros((w, h, 3), np.uint8)
            else:
                im_copy = np.zeros((w, h, 1), np.uint8)

            # calculate offset for translation of warped image  ...find the offset to the center of the blank image
            x_offset = stitched_frame_size[0] / 2 - warped_pt[0]
            y_offset = stitched_frame_size[1] / 2 - warped_pt[1]

            # translation matrix
            Trans = np.array([[1, 0, x_offset], [0, 1, y_offset], [0, 0, 1]])
            # translate tomography matrix
            translated_H = np.dot(Trans, homography)

            # get the location to add the other stitched image
            b_x = int(x_offset)
            b_y = int(y_offset)
            # print("in loop")
            hold = False
            try:
                hold = False
                stitched = cv2.warpPerspective(a, translated_H, stitched_frame_size)
                if i == 0:
                    stitched[b_y:b.shape[0] + b_y, b_x:b.shape[1] + b_x] = b  # on the first image match use this
                else:
                    im_copy[b_y:b.shape[0] + b_y, b_x:b.shape[1] + b_x] = b  # check sizing using blank image to get the best image size
            except ValueError as e:
                inc += 3
                hold = True
            # if not hold:
            #     break


            hold_img = stitched[b_y:b.shape[0] + b_y, b_x:b.shape[1] + b_x]  # create roi
            # Now create a mask of logo and create its inverse mask also
            new2_gray = cv2.cvtColor(hold_img, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(new2_gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            # Now black-out the area of logo in ROI
            img1_bg = cv2.bitwise_and(hold_img, hold_img, mask=mask)

            # Take only region of logo from logo image.
            img2_fg = cv2.bitwise_and(b, b, mask=mask_inv)
            # Put logo in ROI and modify the main image

            dst = cv2.add(img1_bg, img2_fg)
            stitched[b_y:b.shape[0] + b_y, b_x:b.shape[1] + b_x] = dst

            # cropping function
            # 2d ..x,y format... clockwise..
            a_sizes = [[0, 0, 1],  # this should be the warped image
                       [a.shape[1], 0, 1],
                       [a.shape[1], a.shape[0], 1],
                       [0, a.shape[0], 1]]

            b_sizes = [[b_x, b_y],  # non warped image
                       [b.shape[1] + b_x, b_y],
                       [b.shape[1] + b_x, b.shape[0] + b_y],
                       [b_x, b.shape[0] + b_y]]

            new_pts_warped = []
            for pt in a_sizes:  # find the new warped location points of the warped image
                warped_pt = np.dot(translated_H, pt)
                warped_pt = [x / warped_pt[2] for x in warped_pt]
                new_pts_warped.append(warped_pt)

            new_pts_warped = np.array(new_pts_warped)[0:, 0:2]  # get the first 2 columns since the third column is just ones
            non_warped = np.array(b_sizes)

            all_pts = np.concatenate((new_pts_warped, non_warped))
            x = int(min(all_pts[:, 0]))  # get the min x
            xw = int(max(all_pts[:, 0]))  # get the max x
            y = int(min(all_pts[:, 1]))  # get the min y
            yh = int(max(all_pts[:, 1]))  # get the max y
            stitched = stitched[y:yh, x:xw]  # crop image
            print("updating")
            a = stitched    #assign final stitch to 'a'
        final_stitch = a
        print("done stitching")
        return final_stitch

    def nothing(self, x):
        pass

    def create_output(self, vertices, colors, filename):
        colors = colors.reshape(-1, 3)
        vertices = np.hstack([vertices.reshape(-1, 3), colors])

        ply_header = '''ply
    		format ascii 1.0
    		element vertex %(vert_num)d
    		property float x
    		property float y
    		property float z
    		property uchar red
    		property uchar green
    		property uchar blue
    		end_header
    		'''
        with open(filename, 'w') as f:
            f.write(ply_header % dict(vert_num=len(vertices)))
            np.savetxt(f, vertices, '%f %f %f %d %d %d')





    def image_analyis(self):
        mask_imgs_stitch, real_imgs_stitch = [], []

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

        homo1 = "homography_list_2"
        hm1 = open(homo1, 'rb')  # Open the file
        homography_list = pickle.load(hm1)  # Assign the recreated object

        w, h = 960, 720    #image shape
        kernel = np.ones((3, 3), np.uint8)        #kernel for closing filter
        # focal_length = np.load('camera_params/FocalLength.npy')
        focal_length = 10

        #Perspective transformation matrix
        #This transformation matrix is from the openCV documentation, didn't seem to work for me.
        Q = np.float32([[1, 0, 0, -w/2.0],
                        [0, 1, 0, h/2.0],
                        [0, 0, 0, -focal_length],
                        [0, 0, 1, 0]])
        #This transformation matrix is derived from Prof. Didier Stricker's power point presentation on computer vision.
        #Link : https://ags.cs.uni-kl.de/fileadmin/inf_ags/3dcv-ws14-15/3DCV_lec01_camera.pdf

        Q2 = np.float32([[1, 0, 0, 0],
                         [0, -1, 0, 0],
                         [0, 0, focal_length*0.05, 0], #Focal length multiplication obtained experimentally.
                         [0, 0, 0, 1]])

        # Define name for output file
        output_file = 'reconstructed.ply'

        while not rospy.is_shutdown():
            #put all the factory_imgs in an array
            active = np.array([(self.camera1 > 0), (self.camera2 > 0), (self.camera3 > 0),
            (self.camera4 > 0), (self.camera5 > 0), (self.camera6 > 0), (self.camera7 > 0), (self.camera8 > 0),
            (self.camera4 > 0), (self.camera5 > 0), (self.camera6 > 0), (self.camera7 > 0), (self.camera8 > 0),
            (self.camera9 > 0), (self.camera10 > 0), (self.camera11 > 0), (self.camera12 > 0), (self.camera13 > 0),
            (self.camera14 > 0), (self.camera15 > 0), (self.camera16 > 0), (self.camera17 > 0), (self.camera18 > 0),
            (self.camera19 > 0), (self.camera20 > 0), (self.camera21 > 0), (self.camera22 > 0), (self.camera23 > 0),
            (self.camera24 > 0)])

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
                camera16_img = self.camera16
                camera17_img = self.camera17
                camera18_img = self.camera18
                camera19_img = self.camera19
                camera20_img = self.camera20
                camera21_img = self.camera21
                camera22_img = self.camera22
                camera23_img = self.camera23
                camera24_img = self.camera24

                # stereo_cams = [(camera1_img, camera2_img), (camera3_img,camera4_img), (camera5_img, camera6_img),
                #                (camera7_img, camera8_img), (camera9_img, camera10_img), (camera11_img, camera12_img),
                #                (camera13_img, camera14_img), (camera15_img, camera16_img), (camera17_img, camera18_img),
                #                (camera19_img, camera20_img), (camera21_img, camera22_img), (camera24_img, camera24_img)]

                real_imgs_stitch = [camera1_img, camera3_img, camera5_img, camera7_img, camera9_img, camera11_img,
                                    camera13_img, camera15_img, camera17_img, camera19_img, camera21_img, camera23_img]

                # preFilterType =
                # preFilterSize =
                # textureThreshold =
                # blockSize = 5
                # preFilterCap = 5
                # uniquenessRatio = 10
                # speckleRange = 32
                # speckleWindowSize = 100
                # disp12MaxDiff = 5
                # maxDisparity = 130
                # minDisparity = 2
                # numDisparities = maxDisparity - minDisparity
                # P1 = 8 * 3 * blockSize ** 2
                # P2 = 32 * 3 * blockSize ** 2

                blockSize = 3
                preFilterCap = 5
                uniquenessRatio = 10
                speckleRange = 5
                speckleWindowSize = 100
                disp12MaxDiff = 2
                maxDisparity = 64
                minDisparity = 0
                numDisparities = maxDisparity - minDisparity
                P1 = 8 * 3 * blockSize ** 2
                P2 = 32 * 3 * blockSize ** 2


                print("working")
                # Create StereoSGBM and prepare all parameters
                window_size = blockSize
                min_disp = minDisparity
                num_disp = numDisparities
                pre_cap = preFilterCap

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

                # # Used for the filtered image
                stereoR = cv2.ximgproc.createRightMatcher(stereo)  # Create another stereo for right this time

                # WLS FILTER Parameters
                lmbda = 80000
                sigma = 1.5
                visual_multiplier = 1.0

                wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
                wls_filter.setLambda(lmbda)
                wls_filter.setSigmaColor(sigma)

                # for (imga, imgb) in stereo_cams:
                # frameR = imga
                # frameL = imgb

                frameR = camera21_img
                frameL = camera22_img

                Left_nice = cv2.remap(frameL, Left_Stereo_Map_x,Left_Stereo_Map_y, cv2.INTER_LANCZOS4,cv2.BORDER_CONSTANT,0)  # Rectify the image using the calibration parameters founds during the initialisation
                Right_nice = cv2.remap(frameR, Right_Stereo_Map_x, Right_Stereo_Map_y, cv2.INTER_LANCZOS4,cv2.BORDER_CONSTANT, 0)

                # Convert from color(BGR) to gray
                grayR = cv2.cvtColor(Right_nice, cv2.COLOR_BGR2GRAY)
                grayL = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2GRAY)

                # Compute the 2 factory_imgs for the Depth_image
                disp = stereo.compute(grayL, grayR)
                dispL = disp
                dispR = stereoR.compute(grayR, grayL)
                dispL = np.int16(dispL)
                dispR = np.int16(dispR)

                # Using the WLS filter
                filteredImg = wls_filter.filter(dispL, grayL, None, dispR)
                filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
                mask_min = 1   # Don't use zero as min
                mask_max = 15
                mask_temp = cv2.inRange(filteredImg, mask_min, mask_max)
                mask_temp = np.uint8(mask_temp)
                mask_temp = cv2.bitwise_not(mask_temp)
                filteredImg = np.uint8(filteredImg)


                # #Reproject points into 3D
                # points_3D = cv2.reprojectImageTo3D(disp, Q2)
                # #Get color points
                # colors = cv2.cvtColor(frameL, cv2.COLOR_BGR2RGB)    #Get rid of points with value 0 (i.e no depth)
                # mask_map = disp > disp.min()        #Mask colors and points.
                # output_points = points_3D[mask_map]
                # output_colors = colors[mask_map]
                #
                # self.create_output(output_points, output_colors, output_file)
                # print("reconstruction done")

                # cv2.imshow('Disparity Map', filteredImg)
                disp = ((disp.astype(np.float32) / 16) - min_disp) / num_disp  # Calculation allowing us to have 0 for the most distant object able to detect

                ##    # Resize the image for faster executions
                ##    dispR= cv2.resize(disp,None,fx=0.7, fy=0.7, interpolation = cv2.INTER_AREA)

                # Filtering the Results with a closing filter
                closing = cv2.morphologyEx(disp, cv2.MORPH_CLOSE, kernel)  # Apply an morphological filter for closing little "black" holes in the picture(Remove noise)

                # Colors map
                dispc = (closing - closing.min()) * 255
                dispC = dispc.astype(np.uint8)  # Convert the type of the matrix from float32 to uint8, this way you can show the results with the function cv2.imshow()

                disp_color = cv2.applyColorMap(dispC, cv2.COLORMAP_OCEAN)  # Change the Color of the Picture into an Ocean Color_Map
                filt_color = cv2.applyColorMap(filteredImg, cv2.COLORMAP_OCEAN)
                # mask_imgs_stitch.append(mask_temp)

                # print("starting stitch")
                # final_stitch = self.stitch_images(homography_list, real_imgs_stitch)


                # cv2.imwrite("save_imgs/camera1.png", real_imgs_stitch[0])
                # cv2.imwrite("save_imgs/camera2.png",  real_imgs_stitch[1])
                # cv2.imwrite("save_imgs/camera3.png",  real_imgs_stitch[2])
                # cv2.imwrite("save_imgs/camera4.png",  real_imgs_stitch[3])
                #
                # cv2.imwrite("save_imgs/mask1.png", mask_imgs_stitch[0])
                # cv2.imwrite("save_imgs/mask2.png", mask_imgs_stitch[1])
                # cv2.imwrite("save_imgs/mask3.png", mask_imgs_stitch[2])
                # cv2.imwrite("save_imgs/mask4.png", mask_imgs_stitch[3])


                final_img = filt_color  #final image to be shown
                # print(final_img.shape)
                final_img = cv2.resize(final_img,(1080,640))
                try:
                    self.pub.publish(self.br.cv2_to_imgmsg(final_img))
                except CvBridgeError as e:
                    rospy.logerr("CvBridge Error: {0}".format(e))

                cv2.imshow("camera_feed",final_img)
                if cv2.waitKey(1) & 0XFF == ord("q"):
                    break
            # self.rate.sleep()

    
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