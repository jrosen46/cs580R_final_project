#!/usr/bin/env python

"""
memory_bank.py

"""

import os

import numpy as np
import rospy
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg


class MemoryBank(object):
    """External memory bank node."""

    # TODO: have no idea about order of magnitude ... will depend on what layer
    #       of the network we choose ... test it
    TOLERANCE = 1e-2

    def __init__(self):

        self.mem_bank = []

        rospy.init_node('mem_bank', anonymous=False)

        rospy.Subscriber('feature_vectors', numpy_msg(Floats),
                         self._compute_l2_and_add_to_bank)

        # TODO: Need to decide on topic that will handle movement, and publish
        #       whether or not to explore the room.
        self.pub = rospy.Publisher('', String, queue_size=10)

    def _compute_l2_and_add_to_bank(self, feature_vecs):
        """Computes l2 b/n incoming feat vectors and feat vectors in mem bank.

        Parameters
        ----------
        feature_vecs : numpy.ndarray of shape (# vectors, # features)
            Will contain a number of feature vectors from the doorway of
            a room. Each row will contain a different feature vector.

        Returns
        -------

        TODO
        ----
        > will this give a memory error if too large? Possibly look at:
        https://stackoverflow.com/questions/27948363/
            numpy-broadcast-to-perform-euclidean-distance-vectorized
        """

        # This will result in a 2d array of shape: (# new feature vectors, #
        # feature vectors in `mem_bank`). The values will contain the l2
        # distance b/t each new feature vector and every feature vector in the
        # bank (i.e. value for position 2, 8 in array will be the l2 norm b/t
        # the third incoming feature vector and the ninth vector in the
        # `mem_bank`).
        l2 = np.sqrt(np.square(feature_vecs[:, np.newaxis]
                     - np.asarray(self.mem_bank)).sum(axis=2))

        if l2.min() < self.TOLERANCE:
            # TODO: DO NOT ENTER ROOM ... ALREADY BEEN THERE!
            # Publish
            pass
        else:
            # TODO: ENTER ROOM
            pass



if __name__ == '__main__':
    try:
        memory_bank = MemoryBank()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
