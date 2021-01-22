from styx_msgs.msg import TrafficLight
import rospy
import numpy as np
import tensorflow as tf
from PIL import Image

SSD_GRAPH_FILE = '/home/workspace/data/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'
DETECT_TIME_DELAY = 0.1

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.prev_color = TrafficLight.UNKNOWN
        self.curr_color = TrafficLight.UNKNOWN
        self.curr_count = 0
        self.prev_log_time = rospy.get_time() - 1
        self.prev_detect_time = rospy.get_time() - DETECT_TIME_DELAY
        self.detection_graph = self.load_graph(SSD_GRAPH_FILE)
        
        # The input placeholder for the image.
        # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')

        # The classification of the object (integer id).
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        
        self.sess = tf.Session(graph=self.detection_graph)

    @staticmethod
    def load_graph(graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph

    @staticmethod
    def filter_boxes(min_score, boxes, scores, classes, cls=None):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score and (cls is None or classes[i] == cls):
                idxs.append(i)

        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    def pipeline_img(self, img, min_conf=0.8, cls=None):
        boxes, scores, classes = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
                                               feed_dict={self.image_tensor: np.expand_dims(img, 0)})
        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        # Filter boxes with a confidence score less than min_conf
        return self.filter_boxes(min_conf, boxes, scores, classes, cls)

    @staticmethod
    def to_image_coords(boxes, height, width):
        """
        The original box coordinate output is normalized, i.e [0, 1].

        This converts it back to the original coordinate based on the image
        size.
        """
        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width

        return box_coords

    @staticmethod
    def traffic_light_color_detector(img, box_coords, scores, color_thresh=225, l_tol=0.001, u_tol=0.9):
        """
        image -- BGR image
        """
        if len(scores) == 0:
            return TrafficLight.UNKNOWN, 0
        # index of the most confident bounding box
        b_idx = np.argmax(scores)
        y0 = int(box_coords[b_idx, 0])
        x0 = int(box_coords[b_idx, 1])
        y1 = int(box_coords[b_idx, 2])
        x1 = int(box_coords[b_idx, 3])
        crop_img = img[y0:y1, x0:x1, :]
        rospy.logwarn('TLClassifier.traffic_light_color_detector(): b_idx = %s, box_coords[b_idx] = %s'
                      % (b_idx, box_coords[b_idx]))
        r_mask = crop_img[:,:,2] > color_thresh
        g_mask = crop_img[:,:,1] > color_thresh
        r_sum = np.sum(crop_img[:, :, 2][r_mask])
        g_sum = np.sum(crop_img[:, :, 1][g_mask])
        
        if r_sum == 0 and g_sum == 0:
            return TrafficLight.UNKNOWN, np.nan
        
        r = min(r_sum, g_sum) / max(r_sum, g_sum)
        if r < l_tol:
            if r_sum < g_sum:
                return TrafficLight.GREEN, r
            else:
                return TrafficLight.RED, r

        if r > u_tol:
            return TrafficLight.YELLOW, r
        
        return TrafficLight.UNKNOWN, r
    
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if rospy.get_time() - self.prev_detect_time < DETECT_TIME_DELAY:
            return self.prev_color
        
        img = np.asarray(image)
        self.prev_detect_time = rospy.get_time()
        # cls==10 is for traffic light
        boxes, scores, classes = self.pipeline_img(img, min_conf=0.5, cls=10)
        height, width, _ = img.shape
        box_coords = self.to_image_coords(boxes, height, width)
        color, ratio = self.traffic_light_color_detector(img, box_coords, scores)
        if color == self.curr_color:
            self.curr_count += 1
        else:
            self.prev_color = self.curr_color
            self.curr_color = color
            self.curr_count = 0
        rospy.logwarn('TLClassifier.get_classification(): len(classes) = %d, prev_color = %s, curr_color = %s, curr_count = %s, ratio = %.4f'
                      % (len(classes), self.prev_color, self.curr_color, self.curr_count, ratio))
        #if len(scores) > 0:
        #    img = img[:,:,::-1]
        #    im = Image.fromarray(img)
        #    im.save("/home/workspace/data/images/img_%03d.jpeg" % self.img_count)
        #    self.img_count += 1
        ret_color = self.curr_color
        if self.prev_color == TrafficLight.RED and self.curr_color == TrafficLight.UNKNOWN and self.curr_count < 3:
            # Reduce false positives following a previous red light
            ret_color = TrafficLight.RED
        return ret_color
