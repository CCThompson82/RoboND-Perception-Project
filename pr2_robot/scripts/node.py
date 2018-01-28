#!/usr/bin/env python

# Import modules
import os
import sys
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
import sensor_stick.pcl_helper as util
import pcl
import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml
from sensor_msgs.msg import PointCloud2
from sensor_stick.marker_tools import *

#TODO: Make this generalize:
P_DIR = '/home/chris/catkin_ws/src/RoboND-Perception-Project/pr2_robot'

# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    # Convert ROS msg to PCL data
    cloud = util.ros_to_pcl(pcl_msg)

    # Statistical Outlier Filtering
    K_MEAN = 5
    outlier_filter = cloud.make_statistical_outlier_filter()
    outlier_filter.set_mean_k(K_MEAN)
    STD = 0.1
    outlier_filter.set_std_dev_mul_thresh(STD)
    denoised = outlier_filter.filter()

    # Voxel Grid Downsampling
    voxelator = denoised.make_voxel_grid_filter()
    LEAF_SIZE = 0.005
    voxelator.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    vox_downsampled = voxelator.filter()

    # PassThrough Filter
    passthrough = vox_downsampled.make_passthrough_filter()
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min, axis_max = 0.60, 1.1
    passthrough.set_filter_limits(axis_min, axis_max)
    passthrough_z = passthrough.filter()
    # PassThrough Filter Y
    passthrough = passthrough_z.make_passthrough_filter()
    filter_axis = 'y'
    passthrough.set_filter_field_name(filter_axis)
    axis_min, axis_max = -0.5, 0.5
    passthrough.set_filter_limits(axis_min, axis_max)
    passthrough_filtered = passthrough.filter()

    # RANSAC Plane Segmentation for table versus objects
    table_seg = passthrough_filtered.make_segmenter()
    table_seg.set_model_type(pcl.SACMODEL_PLANE)
    table_seg.set_method_type(pcl.SAC_RANSAC)
    MAX_DISTANCE = 0.01
    table_seg.set_distance_threshold(MAX_DISTANCE)
    inliers, coefficients = table_seg.segment()

    # Extract inliers and outliers
    cloud_objects = passthrough_filtered.extract(inliers, negative=True)
    cloud_table = passthrough_filtered.extract(inliers, negative=False)

    # Euclidean Clustering
    white_cloud = util.XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.05)
    ec.set_MinClusterSize(200)
    ec.set_MaxClusterSize(5000)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()

    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_color = util.get_color_list(len(cluster_indices))
    color_cluster_point_list = []
    for j, cluster in enumerate(cluster_indices):
        for idx in cluster:
            color_cluster_point_list.append(
                [white_cloud[idx][0], white_cloud[idx][1],
                 white_cloud[idx][2], util.rgb_to_float(cluster_color[j])])
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # Convert PCL data to ROS messages
    orig_cloud_msg = util.pcl_to_ros(cloud)
    denoise_msg = util.pcl_to_ros(denoised)
    vox_down_msg = util.pcl_to_ros(vox_downsampled)
    passthrough_msg = util.pcl_to_ros(passthrough_filtered)
    obj_msg = util.pcl_to_ros(cloud_objects)
    table_msg = util.pcl_to_ros(cloud_table)
    cluster_msg = util.pcl_to_ros(cluster_cloud)

    # Publish ROS messages
    incoming_cloud_pub.publish(orig_cloud_msg)
    denoise_pub.publish(denoise_msg)
    downsampled_pub.publish(vox_down_msg)
    passthrough_pub.publish(passthrough_msg)
    pcl_obj_pub.publish(obj_msg)
    pcl_table_pub.publish(table_msg)
    pcl_cluster_pub.publish(cluster_msg)

    # Classify the clusters! (loop through each detected cluster one at a time)
    objects = []
    labels = []
    centroids = []
    for idx, pts_ls in enumerate(cluster_indices):
        # Grab the points for the cluster
        pcl_cluster = cloud_objects.extract(pts_ls)
        sample_cloud = util.pcl_to_ros(pcl_cluster)

        # Compute the associated feature vector
        chists = compute_color_histograms(sample_cloud, using_hsv=False)
        normals = get_normals(sample_cloud)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction
        prediction = clf.predict(scaler.transform(feature.reshape(1, -1)))
        label = encoder.inverse_transform(prediction)[0]
        labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_ls[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label, label_pos, idx))

        # # Add the detected object to the list of detected objects.
        # do = DetectedObject()
        # do.label = label
        # do.cloud = pcl_cluster
        # objects.append(do)
        #
        # # add the centroid to the centroids ls
        # pts_arr = np.asarray(pcl_cluster)[:, :3]
        # centroid = np.mean(pts_arr, axis=0)
        # # TODO: change centroid type
        # centroids.append(centroid)


    # Publish the list of detected objects
    rospy.loginfo('Detected {} objects: {}'.format(len(labels), labels))
    detected_objects_pub.publish(objects)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    # try:
    #     pr2_mover(detected_objects_list)
    # except rospy.ROSInterruptException:
    #     pass

# function to load parameters and request PickPlace service
def pr2_mover(objects_ls):

    # TODO: Initialize variables

    # TODO: Get/Read parameters

    # TODO: Parse parameters into individual variables

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # TODO: Loop through the pick list

        # TODO: Get the PointCloud for a given object and obtain it's centroid

        # TODO: Create 'place_pose' for the object

        # TODO: Assign the arm to be used for pick_place

        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format

        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)

            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file



if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node('perception', anonymous=True)

    # get the pick list from the param server
    object_list_param = rospy.get_param('/object_list')

    # Create Subscribers
    pcl_subscriber = rospy.Subscriber(
        '/pr2/world/points', PointCloud2, pcl_callback, queue_size=1)

    # Create Publishers
    incoming_cloud_pub = rospy.Publisher('/perception/original_cloud', PointCloud2, queue_size=1)
    denoise_pub = rospy.Publisher('/perception/denoised', PointCloud2, queue_size=1)
    downsampled_pub = rospy.Publisher('/perception/downsampled', PointCloud2, queue_size=1)
    passthrough_pub = rospy.Publisher('/perception/passthrough', PointCloud2, queue_size=1)
    pcl_obj_pub = rospy.Publisher('/perception/objects', PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher('/perception/table', PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher('/perception/clusters', PointCloud2, queue_size=1)

    object_markers_pub = rospy.Publisher('/perception/object_markers', Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher('/perception/detected_objects', DetectedObjectsArray, queue_size=1)

    # Load Model From disk
    clf_fname = os.path.join(P_DIR, 'modelling', 'svm_clf.sav')
    model = pickle.load(open(clf_fname, 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    util.get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
