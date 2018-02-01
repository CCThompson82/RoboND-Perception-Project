# Project: Perception Pick & Place


---
## Summary

The objective of this project was to execute a pick and place task for the PR2 robot.  The requisite steps for succesful
execution of the pick and place task included the training/generation of an object detection model, setting up a perception
node capable of receiving pointcloud message information via an rgbd sensor, filtering the incoming pointcloud data into
unique object clusters, and then predicting the label for each unique cluster.  Object detection outcomes were stored in the
`pr2_robot/results` directory.  

[//]: # (Image References)

[raw]: ./misc_images/raw.png
[denoised]: ./misc_images/denoised.png
[passthrough]: ./misc_images/passthrough.png
[voxel]: ./misc_images/voxel_filter.png
[conmat]: ./misc_images/conmat.png
[objects]: ./misc_images/objects.png
[clusters]: ./misc_images/clusters.png

## Methods

### Training an object detection model
The first step to this project was the training of a model capable of recognizing pointcloud
information as a specific object.  To accomplish this, the scripts from Exercise 3 that were
utilized for feature extraction and svm classifer training were copied into this repository.  
These scripts were:
  * `training.launch` into `pr2_robot/launch`
  * `capture_features.py`, `train_svm.py` into `pr2_robot/scripts`
  * `features.py`, `training_helper.py`, `pcl_helper.py` into `pr2_robot/src/sensor_stick`

In `capture_features.py`, the number of pointclouds recorded for each object was increased to 30, and in
`features.py` the number of color and normal features extracted were set to 64, and 32 respectively.
Once parameters were set, `training.launch` was launched which resulted in a training set pickle being
stored to the `pr2_robot/modelling` directory.  `train_svm` was then run, which received this
training set, fit a linear svm classifer, and provided cross-validation confusion matrix for the
models performance.  The accuracy of the model exceeded 60% for all objects in a 5-fold cross-validation.

![alt_text][conmat]
**Figure 1**- Confusion matrix for svm classifier performance trained on normality and color feature extraction.

### ROS NODE

In the script `pr2_robot/scripts/node.py` the necessary publishers and subscribers were
initiated in the `__main__` of the script.  The incoming pointcloud data was subscribed
via `pcl_subscriber`, and each step of the filtering pipeline was published for visual
inspection, debugging, and demonstration purposes.

### Filtering
All filtering steps were performed within the `pcl_callback` function in `node.py`.
The raw incoming pointcloud feed requires processing in order to be efficiently used.  An
example of the incoming feed is seen in Figure 2.  

![alt text][raw]
**Figure 2**- The raw pointcloud feed contains stochastic noise.  

#### Statistical Outlier Filter
A statistical outlier filter was used to remove points that exceeded the standard
deviation position from their neighborhood average.  The resultant feed after this denoising
can be observed in Figure 3.  

![alt_text][denoised]
**Figure 3**- Removal of statistical outliers cleans the pointcloud feed to a more
accurate representation of reality.  

#### Voxelation
Once outliers were removed, a voxelation filter could be applied for an accurate pointcloud
representation of reality.  Voxelation is the compression of a pointcloud into specific positions
based on the presence or absence of raw points in that volume of the original feed.  The result
is a downsampled feed that does not lose much information, as nearly coincident points are
merged into the same voxel position.  An example of voxelation is shown in Figure 4.

![alt_text][voxel]
**Figure 4**- Voxel downsampling of the denoised pointcloud


#### Passthrough Filter

The denoised pointcloud feed was then filtered through a z-plane passthrough filter, intended
to isolate the table and objects of interest.  The range for the pass through filter were
determined empirically.

#### Table and Object Segmentation

The next step of the pipeline was to isolate objects on top of the table from the points representing
the table itself.  To accomplish this feat, a Random Sample Consensus filter was applied to the
passthrough pointcloud.  Given a plane height of approximately 2 cm, the RANSAC filter was capable
of isolating points belonging to the objects from points belonging to the table.  An example of the
RANSAC filter outlier feed (objects) is shown in Figure 5.

![alt_text][objects]
**Figure 5**- RANSAC separation of the table (not shown) from the object clusters.

### Euclidean Clustering
The PCL implementation is a DBSCAN-like algorithm for assigning points of a pointcloud to
cluster groups.  The parameters required are a tolerance level (i.e. an approximate distance to
search for neighboring points), the minimum size of a valid cluster, and the maximum size of
a valid cluster.  The tolerance setting can dictate whether multiple groups are classified together,
or whether a single cluster is classified into multiple groups, and is usually set emperically.
Figure 6 shows the unique clustering that the Euclidean Clustering algorithm can acheive.  These
clustering steps were also performed inside `pcl_callback` within the `node.py` script.  

![alt_text][clusters]
**Figure 6**- Unique clusters identified by Euclidean Clustering.  But which cluster is which object?!

#### Object Detection
Having seperated individual clouds into unique clusters, we are now able to perform object prediction
using the trained SVM model.  The points of each cluster were subjected to the same feature extraction
function, and these features were supplied to the SVM classifier.  Object detection
was performed cluster by cluster in the `pcl_callback` function in `node.py`

#### Centroid calculation
The points for each object were analyzed for their average position (centroid) in
order to inform the robot pick pose. The results of this classification
are stored within yaml files within the `pr2_robot/results` directory.  

### Pick Message
These pieces of information were assembled into a PickPlace message, which was then sent to the
`pick_place_routine` service proxy as a request for robot movement.  The resulting actions were
generally successful in picking, but rarely successful in placing [See discussion].  This logic
was performed within the `pr2_mover` function in `node.py`.

## Discussion

To my vantage, all objects in each of the 3 worlds were correctly identified.  No KeyErrors were
noticed, originating the search for an objects dropbox location where it did not exist in the pick
set.  The model used for object detection was an SVM trained on normality and color features of
each object.  The implementation seemed to work well, though I did need to increase the number of
vantages for each object from about 5 to 30 in order to achieve good performance in cross-validation
accuracy.  

The identification of centroid positions was generally accurate, however it is important to note that
a single-vantage pointcloud will generate a biased centroid for the objects points, as it can only
detect the front of the object.  Rotation around the object, combined with some sort of memory (like
iterative closest point) would provide a more realistic representation of the object and a more
accurate calculation of its centroid.  Nevertheless, objects were routinely picked given the current
implementation.  

## Conclusion

In conclusion, the pipeline of filtering, clustering, and object detection were sufficient to
routinely identify a series of objects for a pick and place task.  
