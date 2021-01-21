from styx_msgs.msg import TrafficLight
import rospy
import numpy as np
import tensorflow as tf

SSD_GRAPH_FILE = '~/Downloads/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.prev_log_time = rospy.get_time() - 1
        self.detection_graph = self.load_graph(SSD_GRAPH_FILE)
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
        with tf.Session(graph=detection_graph) as sess:
            draw_img = Image.fromarray(img)
            boxes, scores, classes = self.sess.run([detection_boxes, detection_scores, detection_classes],
                                                   feed_dict={image_tensor: np.expand_dims(img, 0)})
            # Remove unnecessary dimensions
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)

            # Filter boxes with a confidence score less than min_conf
            boxes, scores, classes = self.filter_boxes(min_conf, boxes, scores, classes, cls)

            return scrores, classes

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        img = numpy.asarray(image)
        # cls==10 is for traffic light
        scrores, classes = self.pipeline_img(img, conf=0.5, cls=10)
        if rospy.get_time() - self.prev_log_time >= 1:
            self.prev_log_time = rospy.get_time()
            rospy.logwarn('TLClassifier.get_classification(): len(scrores) = %d' % len(scrores))
        return TrafficLight.UNKNOWN
