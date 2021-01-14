#!/usr/bin/env python

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 0.5
PUBLISH_DELAY = 0.5

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        self.final_waypoints_pub = rospy.Publisher('/final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.base_lane = None
        self.pose = None
        self.stopline_wp_idx = -1
        self.base_lane = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.prev_update_time = rospy.get_time() - PUBLISH_DELAY
        self.prev_log_time = rospy.get_time() - 1
        self.pose_prev_log_time = rospy.get_time() - 1
        self.traffic_prev_log_time = rospy.get_time() - 1
        self.pose_idx = 0
        self.pose_arr = []
        self.pose_time = []
        self.pose_x = 0
        self.pose_y = 0
        self.loop()

    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            #if self.pose and self.base_waypoints: # TBR
                # Get closest waypoint
                #closest_waypoint_idx = self.get_closest_waypoint_idx() # TBR
                #self.publish_waypoints_old(closest_waypoint_idx)
            if self.pose and self.base_lane:
                self.publish_waypoints()
            rate.sleep()

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        # Check if closest is ahead or hehind vehicle closest_coord = self.waypoints_2d[closest_idx]
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[(closest_idx-1) % len(self.waypoints_2d)]

        # Equation for hyperplane through closest_coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect-prev_vect, pos_vect-cl_vect)
        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx

    def publish_waypoints_old(self, closest_idx):
        if rospy.get_time() - self.prev_update_time < PUBLISH_DELAY:
            return
        self.prev_update_time = rospy.get_time()
        lane = Lane()
        lane.header = self.base_waypoints.header
        lane.waypoints = self.base_waypoints.waypoints[closest_idx: closest_idx + LOOKAHEAD_WPS]
        self.final_waypoints_pub.publish(lane)
        if rospy.get_time() - self.prev_log_time >= 1:
            self.prev_log_time = rospy.get_time()
            rospy.logwarn('WaypointUpdater.publish_waypoints(): len(self.base_waypoints.waypoints) = %s, len(lane.waypoints) = %s, closest_idx = %s'
                          % (len(self.base_waypoints.waypoints), len(lane.waypoints), closest_idx))

    def publish_waypoints(self):
        if rospy.get_time() - self.prev_update_time < PUBLISH_DELAY:
            return
        self.prev_update_time = rospy.get_time()
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)

    def generate_lane(self):
        lane = Lane()
        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_lane.waypoints[closest_idx:farthest_idx]
        use_base_wpts = farthest_idx <= self.stopline_wp_idx
        if farthest_idx >= len(self.base_lane.waypoints):
            farthest_idx %= len(self.base_lane.waypoints)
            base_waypoints += self.base_lane.waypoints[:farthest_idx]
            use_base_wpts = (farthest_idx <= self.stopline_wp_idx) and (self.stopline_wp_idx < closest_idx)

        use_base_wpts = use_base_wpts or self.stopline_wp_idx == -1
        if use_base_wpts:
            lane.waypoints = base_waypoints
        else:
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)

        if rospy.get_time() - self.prev_log_time >= 1:
            self.prev_log_time = rospy.get_time()
            rospy.logwarn('WaypointUpdater.generate_lane(): len(base_lane.waypoints) = %s, len(lane.waypoints) = %s, closest_idx = %s, farthest_idx = %s, stopline_wp_idx = %s, use_bw = %s, self.base_lane.waypoints[%s].twist.twist.linear.x = %s'
                          % (len(self.base_lane.waypoints), len(lane.waypoints), closest_idx, farthest_idx, self.stopline_wp_idx, use_base_wpts, closest_idx, self.base_lane.waypoints[closest_idx].twist.twist.linear.x))
        return lane

    def decelerate_waypoints(self, waypoints, closest_idx):
        stop_idx = max(self.stopline_wp_idx - closest_idx - 2, 0) # Two waypoints back from line so front of car stops at line
        temp = []
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose
            dist = self.distance(waypoints, i, stop_idx)
            vel = math.sqrt(2 * MAX_DECEL * dist)
            if vel < 1.:
                vel = 0.

            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)

        return temp

    def pose_cb(self, msg):
        self.pose = msg
        self.pose_x = self.pose.pose.position.x
        self.pose_y = self.pose.pose.position.y
        avg_pose_speed_mph = 0
        if len(self.pose_arr) < 10:
            self.pose_arr.append(np.array([self.pose_x, self.pose_y]))
            self.pose_time.append(rospy.get_time())
        else:
            self.pose_arr[self.pose_idx % 10] = np.array([self.pose_x, self.pose_y])
            self.pose_time[self.pose_idx % 10] = rospy.get_time()
            avg_pose_speed_mph = 2.23694 * np.linalg.norm(self.pose_arr[self.pose_idx % 10] - self.pose_arr[(self.pose_idx - 9) % 10]) / (
                self.pose_time[self.pose_idx % 10] - self.pose_time[(self.pose_idx - 9) % 10])
        if rospy.get_time() - self.pose_prev_log_time >= 1:
            self.pose_prev_log_time = rospy.get_time()
            rospy.logwarn('WaypointUpdater.pose_cb(): avg_pose_speed_mph = %.4f' % avg_pose_speed_mph)
        self.pose_idx += 1

    def waypoints_cb(self, waypoints):
        rospy.logwarn('WaypointUpdater.waypoints_cb(): len(waypoints.waypoints) = %s'
                      % len(waypoints.waypoints))
        self.base_lane = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y]
                                 for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.stopline_wp_idx = msg.data
        if rospy.get_time() - self.traffic_prev_log_time >= 1:
            self.traffic_prev_log_time = rospy.get_time()
            rospy.logwarn('WaypointUpdater.traffic_cb(): self.stopline_wp_idx = %s' % self.stopline_wp_idx)

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2):
            dist += dl(waypoints[i].pose.pose.position, waypoints[i+1].pose.pose.position)
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
