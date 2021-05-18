from sensor_msgs.msg import Image
import cv2
import numpy as np
import rospy
import sys
import pickle
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


    def undistort(self,img,K,D):
        balance = 0.0
        dim = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort

        # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, dim, np.eye(3), balance=balance)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, dim, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return undistorted_img




    def image_analyis(self):
        cnt = 0
        # homo = "h_lateral"
        # hm = open(homo, 'rb')  # Open the file
        # homography = pickle.load(hm)  # Assign the recreated object to bok
        
        while not rospy.is_shutdown():
            #put all the factory_imgs in an array
            # active = np.array([(self.camera1 > 0),(self.camera2 > 0) , (self.camera3 > 0),
            # (self.camera4 > 0), (self.camera5 > 0),(self.camera6 > 0),(self.camera7 > 0) , (self.camera8 > 0),
            # (self.camera4 > 0), (self.camera5 > 0),(self.camera6 > 0),(self.camera7 > 0) , (self.camera8 > 0),
            # (self.camera9 > 0), (self.camera10 > 0),(self.camera11 > 0),(self.camera12 > 0) , (self.camera13 > 0),
            # (self.camera14 > 0),(self.camera15 > 0),(self.camera16 > 0),(self.camera17 > 0) , (self.camera18 > 0),
            # (self.camera19 > 0), (self.camera20 > 0),(self.camera21 > 0),(self.camera22 > 0) , (self.camera23 > 0), \
            # (self.camera24 > 0)])

            active = np.array([(self.camera1 > 0), (self.camera2 > 0)])

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


                
                #undistort factory_imgs
                # dist_camera1 = self.undistort(camera1_img,K1,D1)
                # dist_camera2 = self.undistort(camera2_img,K1,D1)
                # dist_camera3 = self.undistort(camera3_img,K1,D1)
                # dist_camera4 = self.undistort(camera4_img,K1,D1)
                # dist_camera5 = self.undistort(camera5_img,K1,D1)
                # dist_camera6 = self.undistort(camera6_img,K1,D1)
                # dist_camera7 = self.undistort(camera7_img,K1,D1)
                # dist_camera8 = self.undistort(camera8_img,K1,D1)
                # dist_camera9 = self.undistort(camera9_img,K1,D1)
                # dist_camera10 = self.undistort(camera10_img,K1,D1)
                # dist_camera11 = self.undistort(camera11_img,K1,D1)
                # dist_camera12= self.undistort(camera12_img,K1,D1)
                # dist_camera13 = self.undistort(camera13_img,K1,D1)
                # dist_camera14 = self.undistort(camera14_img,K1,D1)
                # dist_camera15 = self.undistort(camera15_img,K1,D1)
                # dist_camera16 = self.undistort(camera16_img, K1, D1)
                # dist_camera17 = self.undistort(camera17_img, K1, D1)
                # dist_camera18 = self.undistort(camera18_img, K1, D1)
                # dist_camera19 = self.undistort(camera19_img, K1, D1)
                # dist_camera20 = self.undistort(camera20_img, K1, D1)
                # dist_camera21 = self.undistort(camera21_img, K1, D1)
                # dist_camera22 = self.undistort(camera22_img, K1, D1)
                # dist_camera23 = self.undistort(camera23_img, K1, D1)
                # dist_camera24 = self.undistort(camera24_img, K1, D1)

                # print(cv2.__version__)
                passed,translated_H,stitched,b_x,b_y = 0,0,0,0,0
                # a = dist_camera1
                # b = dist_camera2

                # cameras = [dist_camera2]
                # imga = dist_camera1
                # print("starting")
                #
                # for imgb in cameras:
                homography, translated_H, stitched, b_x, b_y = 0, 0, 0, 0, 0
                # gray_img1 = cv2.cvtColor(dist_camera1, cv2.COLOR_BGR2GRAY)
                # gray_img2 = cv2.cvtColor(dist_camera2, cv2.COLOR_BGR2GRAY)
                # #
                # # # print("start")
                # # # Initiate SIFT detector
                # sift = cv2.xfeatures2d.SIFT_create()
                # a = gray_img1
                # b = gray_img2
                # #
                # # find the keypoints and descriptors with SIFT
                # kp1, des1 = sift.detectAndCompute(a, None)
                # kp2, des2 = sift.detectAndCompute(b, None)
                #
                # FLANN_INDEX_KDTREE = 0
                # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                # search_params = dict(checks=50)
                #
                # flann = cv2.FlannBasedMatcher(index_params, search_params)
                # matches = flann.knnMatch(des1, des2, k=2)
                #
                # # store all the good matches as per Lowe's ratio test.
                # good = []
                # for m, n in matches:
                #     if m.distance < 0.75 * n.distance:
                #         good.append(m)
                #
                # MIN_MATCH_COUNT = 10
                # ransac_thresh = 0.5
                # if len(good) > MIN_MATCH_COUNT:
                #     src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                #     dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                #
                #     E, mask = cv2.findEssentialMat(src_pts, dst_pts, K1, cv2.RANSAC, ransac_thresh, 1,
                #                                    None)  # apply ransac and get E matrix
                #     _, R, T, mask = cv2.recoverPose(E, src_pts, dst_pts, cameraMatrix=K1,
                #                                     mask=mask)  # get rotation and translation
                #     matchesMask = mask.ravel().tolist()
                #
                #     homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 6.0)
                #     matchesMask = mask.ravel().tolist()
                # else:
                #     print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
                #     matchesMask = None
                #
                # draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                #                    singlePointColor=None,
                #                    matchesMask=matchesMask,  # draw only inliers
                #                    flags=2)
                #
                # fimg = cv2.drawMatches(a, kp1, b, kp2, good, None, **draw_params)
                #
                #
                #
                #     # center of source image
                #     center_pt = [a.shape[1] // 2, a.shape[0] // 2, 1]
                #
                #     # find where center of source image will be after warping without comepensating for any offset
                #     warped_pt = np.dot(homography, center_pt)
                #     warped_pt = [x / warped_pt[2] for x in warped_pt]
                #     # print("passing")
                #     inc = 0.1
                #     cond = True
                #     while cond:
                #         # warping output image size # center of warping output image
                #         stitched_frame_size = tuple(int(inc * x) for x in a.shape)
                #         w, h = stitched_frame_size
                #         im_copy = np.zeros((w, h, 3), np.uint8)
                #
                #         # calculate offset for translation of warped image  ...find the offset to the center of the blank image
                #         x_offset = stitched_frame_size[0] / 2 - warped_pt[0]
                #         y_offset = stitched_frame_size[1] / 2 - warped_pt[1]
                #
                #         # translation matrix
                #         T = np.array([[1, 0, x_offset], [0, 1, y_offset], [0, 0, 1]])
                #         # translate tomography matrix
                #         translated_H = np.dot(T, homography)
                #
                #         # get the location to add the other stitched image
                #         b_x = int(x_offset)
                #         b_y = int(y_offset)
                #
                #         hold = False
                #         try:
                #             hold = False
                #             stitched = cv2.warpPerspective(imga, translated_H, stitched_frame_size)
                #             if imgb == dist_camera2:
                #                 stitched[b_y:b.shape[0]+b_y, b_x:b.shape[1]+b_x] = imgb  #on the first image match use this
                #             else:
                #                 passed = True
                #                 im_copy[b_y:b.shape[0] + b_y, b_x:b.shape[1] + b_x] = imgb  # check sizing using blank image to get the best image size
                #         except ValueError as e:
                #             inc += 3
                #             hold = True
                #         if not hold:
                #             break
                #
                #     print("passed")
                #
                #     if passed:
                #         hold_img = stitched[b_y:b.shape[0] + b_y, b_x:b.shape[1] + b_x]  # create roi
                #         # Now create a mask of logo and create its inverse mask also
                #         new2_gray = cv2.cvtColor(hold_img, cv2.COLOR_BGR2GRAY)
                #         ret, mask = cv2.threshold(new2_gray, 10, 255, cv2.THRESH_BINARY)
                #         mask_inv = cv2.bitwise_not(mask)
                #         # Now black-out the area of logo in ROI
                #         img1_bg = cv2.bitwise_and(hold_img, hold_img, mask=mask)
                #
                #         # Take only region of logo from logo image.
                #         img2_fg = cv2.bitwise_and(dist_camera2, dist_camera2, mask=mask_inv)
                #         # Put logo in ROI and modify the main image
                #         dst = cv2.add(img1_bg, img2_fg)
                #         stitched[b_y:b.shape[0] + b_y, b_x:b.shape[1] + b_x] = dst
                #
                #         # cropping function
                #         # 2d ..x,y format... clockwise..
                #         a_sizes = [[0, 0, 1],  # this should be the warped image
                #                    [a.shape[1], 0, 1],
                #                    [a.shape[1], a.shape[0], 1],
                #                    [0, a.shape[0], 1]]
                #
                #         b_sizes = [[b_x, b_y],  # non warped image
                #                    [b.shape[1] + b_x, b_y],
                #                    [b.shape[1] + b_x, b.shape[0] + b_y],
                #                    [b_x, b.shape[0] + b_y]]
                #
                #         new_pts_warped = []
                #         for pt in a_sizes:  # find the new warped location points of the warped image
                #             warped_pt = np.dot(translated_H, pt)
                #             warped_pt = [x / warped_pt[2] for x in warped_pt]
                #             new_pts_warped.append(warped_pt)
                #
                #         new_pts_warped = np.array(new_pts_warped)[0:, 0:2]  # get the first 2 columns since the third column is just ones
                #         non_warped = np.array(b_sizes)
                #
                #         all_pts = np.concatenate((new_pts_warped, non_warped))
                #         x = int(min(all_pts[:, 0]))  # get the min x
                #         xw = int(max(all_pts[:, 0]))  # get the max x
                #         y = int(min(all_pts[:, 1]))  # get the min y
                #         yh = int(max(all_pts[:, 1]))  # get the max y
                #         stitched = stitched[y:yh, x:xw]  # crop image
                #     imga = stitched

                if cv2.waitKey(1) & 0xFF == ord("s"):
                    # val = "h_lateral"
                    # # print(Left_Stereo_Map)
                    # left_map = open(val, "wb")  # Open the file
                    # pickle.dump(homography, left_map, protocol=2)  # Dump the dictionary bok, the first parameter into the file object w.
                    # left_map.close()
                    # print("done")
                    # name_1 = "calib_imgs/camera_right" + str(cnt) + ".png"
                    # name_2 = "calib_imgs/camera_left" + str(cnt) + ".png"
                    # cv2.imwrite(name_1, camera1_img)
                    # cv2.imwrite(name_2, camera2_img)
                    # print("taken" + str(cnt))
                    # cnt += 1

                    cv2.imwrite("factory_imgs/camera1.png", camera1_img)
                    # cv2.imwrite("factory_imgs/camera2.png", camera2_img)
                    cv2.imwrite("factory_imgs/camera3.png", camera3_img)
                    # cv2.imwrite("factory_imgs/camera4.png", camera4_img)
                    cv2.imwrite("factory_imgs/camera5.png", camera5_img)
                    # cv2.imwrite("factory_imgs/camera6.png", camera6_img)
                    cv2.imwrite("factory_imgs/camera7.png", camera7_img)
                    # cv2.imwrite("factory_imgs/camera8.png", camera8_img)
                    cv2.imwrite("factory_imgs/camera9.png", camera9_img)
                    # cv2.imwrite("factory_imgs/camera10.png", camera10_img)
                    cv2.imwrite("factory_imgs/camera11.png", camera11_img)
                    # cv2.imwrite("factory_imgs/camera12.png", camera12_img)
                    cv2.imwrite("factory_imgs/camera13.png", camera13_img)
                    # cv2.imwrite("factory_imgs/camera14.png", camera14_img)
                    cv2.imwrite("factory_imgs/camera15.png", camera15_img)
                    # cv2.imwrite("factory_imgs/camera16.png", camera16_img)
                    cv2.imwrite("factory_imgs/camera17.png", camera17_img)
                    # cv2.imwrite("factory_imgs/camera18.png", camera18_img)
                    cv2.imwrite("factory_imgs/camera19.png", camera19_img)
                    # cv2.imwrite("factory_imgs/camera20.png", camera20_img)
                    cv2.imwrite("factory_imgs/camera21.png", camera21_img)
                    # cv2.imwrite("factory_imgs/camera22.png", camera22_img)
                    cv2.imwrite("factory_imgs/camera23.png", camera23_img)
                    # cv2.imwrite("factory_imgs/camera24.png", camera24_img)
                    print("done")


                final_img = camera1_img    #final image to be shown
                # final_img = cv2.resize(final_img,(1080,640))
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