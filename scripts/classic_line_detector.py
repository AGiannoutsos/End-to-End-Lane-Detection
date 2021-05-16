import cv2 
import numpy as np
from google.colab.patches import cv2_imshow


class Classic_Line_Detector():
    def __init__(self,
                 low_threshold=30,
                 high_threshold=100,
                 gaussian_kernel=5,
                 triangle_top_factor=0.1):
        
        self.low_threshold  = low_threshold
        self.high_threshold = high_threshold
        self.gaussian_kernel = gaussian_kernel
        self.triangle_top_factor = triangle_top_factor
        self.title = "Low:%s High:%s" %(self.low_threshold, self.high_threshold)

    def get_output(self, image):
        return self(image)

    def make_points(self, image, average): 
        try:
            slope, y_int = average 
        except:
            slope, y_int = 1, 0
        y1 = image.shape[0]
        y2 = int(y1 * (2/5))
        x1 = int((y1 - y_int) // slope)
        x2 = int((y2 - y_int) // slope)
        return np.array([x1, y1, x2, y2])  

    def __call__(self, image):
        image_shape = image.shape

        # Greyscaling the iamge
        grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2_imshow(grey_image)

        # Gaussian smoothing
        blur_image = cv2.GaussianBlur(grey_image, (self.gaussian_kernel, self.gaussian_kernel),0)
        # cv2_imshow(blur_image)

        # Canny edge detection
        edges_image = cv2.Canny(blur_image, self.low_threshold, self.high_threshold)
        # cv2_imshow(edges_image)

        # Masking the image
        triangle_mask = np.zeros(image.shape[0:2], dtype="uint8")
        points = np.array( [[ (image_shape[1]/2, image_shape[0]*self.triangle_top_factor), (0, image_shape[0]), (image_shape[1], image_shape[0]) ]], dtype=np.int32)
        triangle_mask = cv2.fillPoly(triangle_mask, points, (255, 255, 255))
        # cv2_imshow(triangle_mask)
        
        # Masking
        masked_line_image = cv2.bitwise_and(edges_image, edges_image, mask=triangle_mask)
        # cv2_imshow(masked_line_image)

        # Hough transform
        rho = 1
        theta = np.pi/180
        threshold = 10
        min_line_length = 20
        max_line_gap = 30
        line_image = np.zeros(image.shape, dtype="uint8")

        # Run Hough on edge detected image
        lines = cv2.HoughLinesP(masked_line_image, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

        # Iterate over the output "lines"  and find the average slope of the lines
        left = []
        right = []
        for line in lines:
            for x1,y1,x2,y2 in line:
                x1, y1, x2, y2 = line.reshape(4)
                parameters = np.polyfit((x1, x2), (y1, y2), 1)
                slope = parameters[0]
                y_int = parameters[1]
                if slope < 0:
                    left.append((slope, y_int))
                else:
                    right.append((slope, y_int))
        
        # find the average slope and their lines
        right_avg = np.average(right, axis=0)
        left_avg = np.average(left, axis=0)
        left_line = self.make_points(image, left_avg)
        right_line = self.make_points(image, right_avg)
        # draw the average lines
        try:
            line_image = cv2.line(line_image, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0,0,255),10)
            line_image = cv2.line(line_image, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0,0,255),10)
        except:
            print("line not found")

        return line_image