#!/usr/bin/env python

"""
movement.py

Based on this source:
    https://jsk-recognition.readthedocs.io/en/latest/install_astra_camera.html
I am going to assume that the field of view of the camera is 60 degrees.

"""
import math
import time

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped, Twist
from nav_msgs.msg import Odometry, OccupancyGrid
import actionlib
from actionlib_msgs.msg import GoalStatus
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

#http://www.theconstructsim.com/ros-qa-how-to-convert-quaternions-to-euler-angles/
from tf.transformations import euler_from_quaternion, quaternion_from_euler



class Movement(object):

    def __init__(self):

        # keep track of approx pose from /odom topic
        self.approx_last_pose = None
        
        rospy.init_node('movement', anonymous=False)

        self.pub = rospy.Publisher('/cmd_vel_mux/input/teleop', Twist,
                                   queue_size=10) 
        self.ac = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.ac.wait_for_server(rospy.Duration(5))

        rospy.Subscriber('processed_detectios', numpy_msg(Floats),
                         self.detection_callback)
        rospy.Subscriber('/odom', Odometry, self.set_approx_pose)

        # TODO: Also need to subscribe to a topic that is published from the
        #       memory bank on whether or not to explore the room.

    def detection_callback(self, data):
        """Moves toward target object.
        """
        pass

        # TODO: https://answers.ros.org/question/203952/move_base-goals-in-base_link-for-turtlebot/
        # first we query pose to get an idea of where the robot thinks it is ... then we add our
        # goal to it ... this may work ... but also may not if the robot quickly changes its view
        # on where it is right? 

        #http://www.theconstructsim.com/ros-qa-how-to-convert-quaternions-to-euler-angles/

        # TODO: make this more robust later ...
        if self.target_id in set(data[:, 0]):

            # gather last pose and transform into euler
            orient = self.approx_last_pose.orientation
            orient_list = [orient_q.x, orient_q.y, orient_q.z, orient_q.w]
            roll, pitch, yaw = euler_from_quaternion(orient_list)

            # now add how much we want to rotate here ... only add the z direction


            # convert back into quaternion to use with action lib
            Quaternion(quaternion_from_euler(roll, pitch, yaw))


    def auto_explore(self):
        """
        """
        pass

    def set_approx_pose(self, data):
        """
        """
        self.approx_last_pose = data.pose.pose

    

    def _rotate(self, degrees, direction):
        """Rotates turtlebot.

        Parameters
        ----------
        degrees : int
        direction : str
        """
        move_msg = Twist()
        angle = 0
        t0 = rospy.get_time()

        if direction == 'clockwise':
            move_msg.angular.z = -.4
            rad = -math.pi / (180./degrees)
            while angle > rad:
                self.pub.publish(move_msg)
                t1 = rospy.get_time()
                angle = (t1-t0)*(-.4)
        else:
            move_msg.angular.z = .4
            rad = math.pi / (180./degrees)
            while angle < rad:
                self.pub.publish(move_msg)
                t1 = rospy.get_time()
                angle = (t1-t0)*.4

        # stop turtlebot from rotating due to momentum
        move_msg.angular.z = 0
        self.pub.publish(move_msg)

    

if __name__ == '__main__':
    try:
        movement = Movement()
        movement.auto_explore()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
