/*
 * Surface Normal Detection and Coordinate Frame Estimation
 * =======================================================
 *
 * This program uses RGB-D camera data to detect blue markers on a surface,
 * compute their 3D positions and normal vectors, and establish a coordinate frame
 * for robot manipulation tasks. It provides a ROS service that delivers this
 * information to client applications.
 *
 * Main components:
 * 1. IMAGE PROCESSING: Detects blue markers using color segmentation
 * 2. NORMAL ESTIMATION: Computes surface normals at each marker location
 * 3. FRAME ESTIMATION: Establishes a consistent coordinate frame for manipulation
 * 4. ROS SERVICE: Provides detected data to other nodes
 *
 * Algorithm flow:
 * - Synchronize RGB and depth data from the camera
 * - Detect blue markers using HSV color thresholding
 * - Extract 3D positions for each marker centroid
 * - Compute normal vectors for each centroid using neighbor points
 * - Establish a consistent XYZ coordinate frame
 * - Store the data for service responses
 * - Visualize the detected markers and coordinate frames
 */

// axes_plane.cpp

#include <ros/ros.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/common.h>
#include <pcl/common/eigen.h>
#include <pcl_conversions/pcl_conversions.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <visualization_msgs/Marker.h>
#include <algorithm> // For std::find
#include <sstream>   // For std::ostringstream
#include <geometry_msgs/Vector3.h>

#include <opencv_services_copy/all_normal.h> // **Updated: Use all_normal service**
#include <geometry_msgs/PointStamped.h>

/******************************************************************************
 * GLOBAL VARIABLES AND DECLARATIONS
 *****************************************************************************/

// Declare synchronization policy
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> SyncPolicy;

// Global Variables
cv::Mat camera_image;
std::vector<cv::Point2f> target_centroids;
std::vector<pcl::PointXYZ> centroids_positions;

tf2_ros::Buffer tf_buffer;
tf2_ros::TransformListener *tf_listener_ptr = nullptr;

// Publisher for normal vectors as markers
ros::Publisher normal_pub;

// Service server
ros::ServiceServer service;

// Global variables to store the latest data
std::vector<geometry_msgs::Point> component_positions;
std::vector<geometry_msgs::Vector3> normals; // **Added: To store all normal vectors**
geometry_msgs::Vector3 average_normal;

// Global variables to store the average axes
geometry_msgs::Vector3 average_x_axis;
geometry_msgs::Vector3 average_y_axis;

bool data_available = false; // Flag to indicate data availability

// Specify the centroid indices you want to average
std::vector<int> centroid_indices_to_average = {2, 3, 6, 7, 4, 8, 1, 5, 9};
// std::vector<int> centroid_indices_to_average = {0,1,2,3,4,5,6,7,8};

// **Added Global Variables for Initial Values**
bool initial_normal_saved = false; // Flag to track if the initial normal is saved
geometry_msgs::Vector3 initial_normal;
geometry_msgs::Vector3 initial_x_axis;
geometry_msgs::Vector3 initial_y_axis;
std::vector<geometry_msgs::Point> initial_component_positions;
std::vector<geometry_msgs::Vector3> initial_normals; // **Added: To store initial normals**
bool initial_normals_saved = false;                  // **Added: Flag for initial normals**

/******************************************************************************
 * NORMAL ESTIMATION FUNCTIONS
 *
 * Functions that compute normal vectors from point cloud data
 *****************************************************************************/

/**
 * Function to compute average normal via plane fitting
 */
bool compute_average_normal_plane_fitting(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                          Eigen::Vector3f &average_normal_out)
{
    if (cloud->points.size() < 3)
    {
        ROS_WARN("Not enough points to fit a plane.");
        return false;
    }

    // Compute the centroid of the point cloud
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud, centroid);

    // Compute the covariance matrix
    Eigen::Matrix3f covariance;
    pcl::computeCovarianceMatrixNormalized(*cloud, centroid, covariance);

    // Perform eigenvalue decomposition
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance);
    Eigen::Vector3f eigen_values = eigen_solver.eigenvalues();
    Eigen::Matrix3f eigen_vectors = eigen_solver.eigenvectors();

    // The normal of the plane is the eigenvector corresponding to the smallest eigenvalue
    average_normal_out = eigen_vectors.col(0); // Assuming eigenvalues are sorted in ascending order
    average_normal_out.normalize();

    return true;
}

/**
 * Function to estimate normal using PCL's NormalEstimation
 */
bool estimate_normal_pcl(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                         const pcl::PointXYZ &searchPoint,
                         Eigen::Vector3f &normal_vector_out,
                         int K = 30)
{
    // Create a KdTree for neighborhood searches
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    // Perform KNN search
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);

    if (kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
    {
        // Extract the neighborhood points
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_neighborhood(new pcl::PointCloud<pcl::PointXYZ>());
        for (size_t i = 0; i < pointIdxNKNSearch.size(); ++i)
        {
            cloud_neighborhood->points.push_back(cloud->points[pointIdxNKNSearch[i]]);
        }

        // Initialize NormalEstimation
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
        ne.setInputCloud(cloud_neighborhood);

        // Create a KdTree for the neighborhood
        pcl::search::KdTree<pcl::PointXYZ>::Ptr neighborhood_kdtree(new pcl::search::KdTree<pcl::PointXYZ>());
        ne.setSearchMethod(neighborhood_kdtree);

        // Output normals
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>());

        // Set the number of nearest neighbors to use for the feature estimation
        ne.setKSearch(K);

        // Compute the normals
        ne.compute(*cloud_normals);

        if (!cloud_normals->empty())
        {
            // Assuming the first normal corresponds to the search point
            pcl::Normal normal = cloud_normals->points[0];

            // Normalize the normal vector
            Eigen::Vector3f normal_vector(normal.normal_x, normal.normal_y, normal.normal_z);
            normal_vector.normalize();

            normal_vector_out = normal_vector;
            return true;
        }
        else
        {
            ROS_WARN("Normal estimation failed: No normals computed.");
            return false;
        }
    }
    else
    {
        ROS_WARN("Normal estimation failed: KNN search did not find any neighbors.");
        return false;
    }
}

/******************************************************************************
 * VISUALIZATION FUNCTIONS
 *
 * Functions for visualizing detected markers, normal vectors, and coordinate frames
 *****************************************************************************/

/**
 * Function to publish the normal vector as a visualization marker
 */
void publish_normal_vector(const Eigen::Vector3f &normal, const pcl::PointXYZ &position, const std::string &frame_id, int id)
{
    visualization_msgs::Marker normal_marker;
    normal_marker.header.frame_id = frame_id;
    normal_marker.header.stamp = ros::Time::now();
    normal_marker.ns = "normals";
    normal_marker.id = id;
    normal_marker.type = visualization_msgs::Marker::ARROW;
    normal_marker.action = visualization_msgs::Marker::ADD;

    // Define the start and end points of the arrow
    geometry_msgs::Point start_point;
    start_point.x = position.x;
    start_point.y = position.y;
    start_point.z = position.z;

    geometry_msgs::Point end_point;
    float scale = 0.01; // Length of the arrow
    end_point.x = start_point.x + normal[0] * scale;
    end_point.y = start_point.y + normal[1] * scale;
    end_point.z = start_point.z + normal[2] * scale;

    normal_marker.points.push_back(start_point);
    normal_marker.points.push_back(end_point);

    // Set the color (e.g., red for normal vector)
    normal_marker.color.r = 1.0;
    normal_marker.color.g = 0.0;
    normal_marker.color.b = 0.0;
    normal_marker.color.a = 1.0;

    // Set the scale of the arrow
    normal_marker.scale.x = 0.01; // Shaft diameter
    normal_marker.scale.y = 0.02; // Head diameter
    normal_marker.scale.z = 0.0;  // Not used for arrows

    // Publish the marker
    normal_pub.publish(normal_marker);
}

/******************************************************************************
 * IMAGE AND POINT CLOUD PROCESSING
 *
 * Core callback function that processes synchronized RGB-D data streams,
 * detects markers, computes normals, and establishes a coordinate frame
 *****************************************************************************/

// Callback function for synchronized image and point cloud messages
void synchronized_callback(const sensor_msgs::ImageConstPtr &image_msg,
                           const sensor_msgs::PointCloud2ConstPtr &cloud_msg)
{
    // Convert ROS Image message to OpenCV image
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    camera_image = cv_ptr->image.clone();

    // Clear previous centroids and positions
    target_centroids.clear();
    centroids_positions.clear();
    normals.clear(); // **Added: Clear previous normals**

    // Detect blue centroids using color segmentation
    cv::Mat hsv_image;
    cv::cvtColor(camera_image, hsv_image, cv::COLOR_BGR2HSV);
    // Display HSV image
    // cv::imshow("HSV Image", hsv_image);

    // Adjusted blue color range in HSV to exclude purple and fine-tune detection
    cv::Scalar lower_blue(100, 100, 50);
    cv::Scalar upper_blue(130, 255, 255);

    cv::Mat mask;
    cv::inRange(hsv_image, lower_blue, upper_blue, mask);
    // Display the mask
    // cv::imshow("Blue Mask", mask);

    // Apply Gaussian Blur to reduce noise
    cv::GaussianBlur(mask, mask, cv::Size(3, 3), 0);
    // Display blurred mask
    // cv::imshow("Blurred Mask", mask);

    // Use morphological closing to fill small holes
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 1);
    // Display mask after morphological closing
    cv::imshow("Mask after Morphological Closing", mask);

    // Use connected components to detect small regions
    cv::Mat labels, stats, centroids_mat;
    int num_objects = cv::connectedComponentsWithStats(mask, labels, stats, centroids_mat, 8, CV_32S);

    // Start from label 1 to skip the background
    for (int label = 1; label < num_objects; ++label)
    {
        int area = stats.at<int>(label, cv::CC_STAT_AREA);

        // Filter out very small areas (noise)
        if (area >= 20)
        { // Adjust the threshold as needed
            double x = centroids_mat.at<double>(label, 0);
            double y = centroids_mat.at<double>(label, 1);
            target_centroids.emplace_back(cv::Point2f(static_cast<float>(x), static_cast<float>(y)));
        }
    }

    // Log the number of detected centroids
    ROS_INFO_STREAM("Number of detected centroids: " << target_centroids.size());

    // Visualize detected centroids in the first window
    cv::Mat centroids_image = camera_image.clone();
    for (size_t i = 0; i < target_centroids.size(); ++i)
    {
        cv::circle(centroids_image, target_centroids[i], 5, cv::Scalar(0, 255, 0), -1); // Green dots
        cv::putText(centroids_image, std::to_string(i), target_centroids[i] + cv::Point2f(10, -10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }

    // Log before showing the first window
    ROS_INFO("Displaying 'Detected Centroids' window.");
    cv::imshow("Detected Centroids", centroids_image);
    ROS_INFO("Displayed 'Detected Centroids' window.");

    // Convert ROS PointCloud2 message to PCL point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*cloud_msg, *pcl_cloud);

    // Create an image to draw axes for all centroids
    cv::Mat axes_image = camera_image.clone();

    // Vectors to store points for plane fitting
    pcl::PointCloud<pcl::PointXYZ>::Ptr points_for_plane_fitting(new pcl::PointCloud<pcl::PointXYZ>());

    // Vectors to store normals and points for specified centroids
    std::vector<cv::Point2f> centroids_to_average;
    std::vector<pcl::PointXYZ> points_to_average;

    // Iterate over each centroid
    for (size_t i = 0; i < target_centroids.size(); ++i)
    {
        cv::Point2f centroid_img = target_centroids[i];
        int u = static_cast<int>(centroid_img.x);
        int v = static_cast<int>(centroid_img.y);

        // Ensure pixel coordinates are within image bounds
        if (u < 0 || u >= camera_image.cols || v < 0 || v >= camera_image.rows)
        {
            ROS_WARN("Centroid %ld is out of image bounds.", i);
            continue;
        }

        // Retrieve 3D point from point cloud (assuming organized point cloud)
        int index = v * cloud_msg->width + u;
        if (index >= pcl_cloud->points.size())
        {
            ROS_WARN("Index %d out of range for point cloud with size %ld", index, pcl_cloud->points.size());
            continue;
        }

        pcl::PointXYZ searchPoint = pcl_cloud->points[index];

        // Check for invalid points (NaN or Inf)
        if (!pcl::isFinite(searchPoint))
        {
            ROS_WARN("Invalid 3D point for centroid %ld.", i);
            continue;
        }

        // Store the position of this centroid
        centroids_positions.push_back(searchPoint);

        // Estimate normal vector using nearest neighbors
        Eigen::Vector3f normal_vector;
        bool normal_estimated = estimate_normal_pcl(pcl_cloud, searchPoint, normal_vector, 10);

        if (normal_estimated)
        {
            // Transform the normal vector to the base frame
            geometry_msgs::Vector3Stamped normal_vector_camera;
            normal_vector_camera.vector.x = normal_vector[0];
            normal_vector_camera.vector.y = normal_vector[1];
            normal_vector_camera.vector.z = normal_vector[2];
            normal_vector_camera.header.frame_id = cloud_msg->header.frame_id; // Camera frame
            normal_vector_camera.header.stamp = cloud_msg->header.stamp;

            geometry_msgs::Vector3Stamped normal_vector_base;
            try
            {
                tf_buffer.transform(normal_vector_camera, normal_vector_base, "base_link", ros::Duration(1.0));
                ROS_INFO_STREAM("Normal Vector for Centroid " << i << " in Base Frame: ["
                                                              << normal_vector_base.vector.x << ", "
                                                              << normal_vector_base.vector.y << ", "
                                                              << normal_vector_base.vector.z << "]");

                // Convert geometry_msgs::Vector3 to Eigen::Vector3f
                Eigen::Vector3f normal_eigen;
                normal_eigen << normal_vector_base.vector.x, normal_vector_base.vector.y, normal_vector_base.vector.z;

                // Publish the normal vector as a marker for visualization
                publish_normal_vector(normal_eigen, searchPoint, "base_link", i);

                // **Added: Store the normal vector in the normals vector**
                normals.push_back(normal_vector_base.vector);

                // Collect points for plane fitting
                if (std::find(centroid_indices_to_average.begin(), centroid_indices_to_average.end(), i + 1) != centroid_indices_to_average.end())
                {
                    // Collect neighboring points
                    pcl::PointCloud<pcl::PointXYZ>::Ptr neighborhood(new pcl::PointCloud<pcl::PointXYZ>());
                    int K = 50; // Number of neighbors
                    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
                    kdtree.setInputCloud(pcl_cloud);

                    std::vector<int> pointIdxNKNSearch(K);
                    std::vector<float> pointNKNSquaredDistance(K);

                    if (kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
                    {
                        for (size_t j = 0; j < pointIdxNKNSearch.size(); ++j)
                        {
                            neighborhood->points.push_back(pcl_cloud->points[pointIdxNKNSearch[j]]);
                        }
                    }

                    // Add to the combined point cloud
                    *points_for_plane_fitting += *neighborhood;

                    // Collect centroids and points for visualization
                    centroids_to_average.push_back(centroid_img);
                    points_to_average.push_back(searchPoint);
                }

                // Define axes length
                float axis_length = 0.02; // Increased to 5 cm for better visibility

                // Define rotation matrix from normal vector
                // Assuming normal_vector is the Z-axis of the local coordinate system
                // We'll create a rotation matrix where:
                // X-axis is arbitrary perpendicular to Z
                // Y-axis is perpendicular to both Z and X

                Eigen::Vector3f z_axis = normal_vector.normalized();

                // Select an arbitrary vector not parallel to z_axis
                Eigen::Vector3f arbitrary;
                if (std::abs(z_axis.z()) < 0.999)
                {
                    arbitrary = Eigen::Vector3f(0, 0, 1);
                }
                else
                {
                    arbitrary = Eigen::Vector3f(1, 0, 0);
                }

                Eigen::Vector3f x_axis = (arbitrary.cross(z_axis)).normalized();
                Eigen::Vector3f y_axis = z_axis.cross(x_axis).normalized();

                // Create rotation matrix
                Eigen::Matrix3f rotation;
                rotation.col(0) = x_axis;
                rotation.col(1) = y_axis;
                rotation.col(2) = z_axis;

                // Convert Eigen::Matrix3f to cv::Mat
                cv::Mat rotation_cv(3, 3, CV_32F);
                for (int row = 0; row < 3; ++row)
                {
                    for (int col = 0; col < 3; ++col)
                    {
                        rotation_cv.at<float>(row, col) = rotation(row, col);
                    }
                }

                cv::Mat rvec;
                cv::Rodrigues(rotation_cv, rvec);

                // Define axes points in 3D
                std::vector<cv::Point3f> axes;
                axes.emplace_back(cv::Point3f(0, 0, 0));
                axes.emplace_back(cv::Point3f(axis_length, 0, 0)); // X-axis
                axes.emplace_back(cv::Point3f(0, axis_length, 0)); // Y-axis
                axes.emplace_back(cv::Point3f(0, 0, axis_length)); // Z-axis

                // Define camera intrinsic matrix
                double fx = 1138.8; // Computed focal length in x
                double fy = 1140.0; // Computed focal length in y
                cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << fx, 0, camera_image.cols / 2.0,
                                         0, fy, camera_image.rows / 2.0,
                                         0, 0, 1);

                // Define distortion coefficients (assuming no distortion)
                cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);

                // Define rotation and translation vectors
                cv::Mat tvec = (cv::Mat_<double>(3, 1) << searchPoint.x, searchPoint.y, searchPoint.z);

                std::vector<cv::Point2f> image_points;
                cv::projectPoints(axes, rvec, tvec, camera_matrix, dist_coeffs, image_points);

                // Draw the axes on the axes_image
                // Draw X-axis in red
                cv::arrowedLine(axes_image, centroid_img, image_points[1],
                                cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                // Draw Y-axis in green
                cv::arrowedLine(axes_image, centroid_img, image_points[2],
                                cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
                // Draw Z-axis in blue
                cv::arrowedLine(axes_image, centroid_img, image_points[3],
                                cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
            }
            catch (tf2::TransformException &ex)
            {
                ROS_WARN("Failed to transform normal vector: %s", ex.what());
            }
        }
    }

    // Compute average normal using plane fitting
    Eigen::Vector3f average_normal_eigen;
    geometry_msgs::Vector3Stamped normal_vector_base;
    if (points_for_plane_fitting->points.size() >= 3)
    {
        if (compute_average_normal_plane_fitting(points_for_plane_fitting, average_normal_eigen))
        {
            // Normalize the normal vector
            average_normal_eigen.normalize();

            // Transform the average normal vector to the base frame
            geometry_msgs::Vector3Stamped average_normal_camera;
            average_normal_camera.vector.x = average_normal_eigen[0];
            average_normal_camera.vector.y = average_normal_eigen[1];
            average_normal_camera.vector.z = average_normal_eigen[2];
            average_normal_camera.header.frame_id = cloud_msg->header.frame_id; // Camera frame
            average_normal_camera.header.stamp = cloud_msg->header.stamp;

            try
            {
                tf_buffer.transform(average_normal_camera, normal_vector_base, "base_link", ros::Duration(1.0));
                ROS_INFO_STREAM("Average Normal Vector in Base Frame: ["
                                << normal_vector_base.vector.x << ", "
                                << normal_vector_base.vector.y << ", "
                                << normal_vector_base.vector.z << "]");

                // Use the average position of the points as the origin
                pcl::PointXYZ average_point;
                average_point.x = 0;
                average_point.y = 0;
                average_point.z = 0;
                for (const auto &pt : points_to_average)
                {
                    average_point.x += pt.x;
                    average_point.y += pt.y;
                    average_point.z += pt.z;
                }
                average_point.x /= points_to_average.size();
                average_point.y /= points_to_average.size();
                average_point.z /= points_to_average.size();

                // Publish the average normal vector as a marker
                Eigen::Vector3f average_normal_eigen_vec;
                average_normal_eigen_vec << normal_vector_base.vector.x, normal_vector_base.vector.y, normal_vector_base.vector.z;
                publish_normal_vector(average_normal_eigen_vec, average_point, "base_link", 999);

                // Define axes length
                float axis_length = 0.02; // 5 cm for better visibility

                if (points_to_average.size() >= 2)
                {
                    // Define the index for the second target point
                    // Step 1: Dynamically select the closest point to average_point as second_target_point
                    size_t closest_idx = 0;
                    float min_distance_sq = std::numeric_limits<float>::max();

                    for (size_t i = 0; i < points_to_average.size(); ++i)
                    {
                        float dx = points_to_average[i].x - average_point.x;
                        float dy = points_to_average[i].y - average_point.y;
                        float dz = points_to_average[i].z - average_point.z;
                        float distance_sq = dx * dx + dy * dy + dz * dz;

                        if (distance_sq < min_distance_sq)
                        {
                            min_distance_sq = distance_sq;
                            closest_idx = i;
                        }
                    }

                    // Check if the index is within the range
                    if (closest_idx < points_to_average.size())
                    {
                        // Step 2: Define X-axis as the vector from average_point to second_target_point
                        pcl::PointXYZ second_target_point = points_to_average[closest_idx];

                        Eigen::Vector3f x_axis_eigen(
                            second_target_point.x - average_point.x,
                            second_target_point.y - average_point.y,
                            second_target_point.z - average_point.z);
                        x_axis_eigen.normalize();

                        // Step 2: Project X-axis onto plane orthogonal to Z-axis
                        Eigen::Vector3f z_axis = average_normal_eigen.normalized();
                        Eigen::Vector3f x_axis_projected = x_axis_eigen - (x_axis_eigen.dot(z_axis)) * z_axis;

                        // Handle degenerate case where projected X-axis is near zero
                        if (x_axis_projected.norm() < 1e-6)
                        {
                            ROS_WARN("Projected X-axis is near zero. Selecting an arbitrary perpendicular vector.");
                            // Select an arbitrary vector not parallel to Z-axis
                            Eigen::Vector3f arbitrary;
                            if (std::abs(z_axis.z()) < 0.99)
                                arbitrary = Eigen::Vector3f(0, 0, 1);
                            else
                                arbitrary = Eigen::Vector3f(1, 0, 0);

                            x_axis_projected = arbitrary.cross(z_axis).normalized();
                        }
                        else
                        {
                            x_axis_projected.normalize();
                        }

                        // Step 3: Compute Y-axis as cross product of Z and projected X axes
                        Eigen::Vector3f y_axis = z_axis.cross(x_axis_projected).normalized();

                        // Step 4: Recompute X-axis to ensure orthonormality
                        x_axis_projected = y_axis.cross(z_axis).normalized();

                        // Transform the X axis to the base frame
                        geometry_msgs::Vector3Stamped x_axis_camera;
                        x_axis_camera.vector.x = x_axis_projected[0];
                        x_axis_camera.vector.y = x_axis_projected[1];
                        x_axis_camera.vector.z = x_axis_projected[2];
                        x_axis_camera.header.frame_id = cloud_msg->header.frame_id;
                        x_axis_camera.header.stamp = cloud_msg->header.stamp;

                        geometry_msgs::Vector3Stamped x_axis_base;
                        try
                        {
                            tf_buffer.transform(x_axis_camera, x_axis_base, "base_link", ros::Duration(1.0));
                            average_x_axis = x_axis_base.vector;
                            ROS_INFO_STREAM("Average X Axis in Base Frame: ["
                                            << average_x_axis.x << ", "
                                            << average_x_axis.y << ", "
                                            << average_x_axis.z << "]");
                        }
                        catch (tf2::TransformException &ex)
                        {
                            ROS_WARN("Failed to transform X axis: %s", ex.what());
                        }

                        // Transform the Y axis to the base frame
                        geometry_msgs::Vector3Stamped y_axis_camera;
                        y_axis_camera.vector.x = y_axis[0];
                        y_axis_camera.vector.y = y_axis[1];
                        y_axis_camera.vector.z = y_axis[2];
                        y_axis_camera.header.frame_id = cloud_msg->header.frame_id;
                        y_axis_camera.header.stamp = cloud_msg->header.stamp;

                        geometry_msgs::Vector3Stamped y_axis_base;
                        try
                        {
                            tf_buffer.transform(y_axis_camera, y_axis_base, "base_link", ros::Duration(1.0));
                            average_y_axis = y_axis_base.vector;
                            ROS_INFO_STREAM("Average Y Axis in Base Frame: ["
                                            << average_y_axis.x << ", "
                                            << average_y_axis.y << ", "
                                            << average_y_axis.z << "]");
                        }
                        catch (tf2::TransformException &ex)
                        {
                            ROS_WARN("Failed to transform Y axis: %s", ex.what());
                        }

                        // Create rotation matrix using orthonormal axes
                        Eigen::Matrix3f rotation;
                        rotation.col(0) = x_axis_projected;
                        rotation.col(1) = y_axis;
                        rotation.col(2) = z_axis;

                        // Convert Eigen::Matrix3f to cv::Mat for OpenCV processing
                        cv::Mat rotation_cv(3, 3, CV_32F);
                        for (int row = 0; row < 3; ++row)
                        {
                            for (int col = 0; col < 3; ++col)
                            {
                                rotation_cv.at<float>(row, col) = rotation(row, col);
                            }
                        }

                        // Convert rotation matrix to rotation vector using Rodrigues
                        cv::Mat rvec;
                        cv::Rodrigues(rotation_cv, rvec);

                        // Define axes points in 3D
                        std::vector<cv::Point3f> axes;
                        axes.emplace_back(cv::Point3f(0, 0, 0));
                        axes.emplace_back(cv::Point3f(axis_length, 0, 0)); // X-axis
                        axes.emplace_back(cv::Point3f(0, axis_length, 0)); // Y-axis
                        axes.emplace_back(cv::Point3f(0, 0, axis_length)); // Z-axis

                        // Define camera intrinsic matrix
                        double fx = 1138.8; // Computed focal length in x
                        double fy = 1140.0; // Computed focal length in y
                        cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << fx, 0, camera_image.cols / 2.0,
                                                 0, fy, camera_image.rows / 2.0,
                                                 0, 0, 1);

                        // Define distortion coefficients (assuming no distortion)
                        cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);

                        // Define rotation and translation vectors
                        cv::Mat tvec = (cv::Mat_<double>(3, 1) << average_point.x, average_point.y, average_point.z);

                        // Project the 3D axes points onto the image plane
                        std::vector<cv::Point2f> image_points;
                        cv::projectPoints(axes, rvec, tvec, camera_matrix, dist_coeffs, image_points);

                        // Compute the average centroid position in the image
                        cv::Point2f average_centroid_img(0, 0);
                        for (const auto &c : centroids_to_average)
                        {
                            average_centroid_img += c;
                        }
                        average_centroid_img /= static_cast<float>(centroids_to_average.size());

                        // // Draw the axes on the image
                        // // Draw X-axis in red
                        // cv::arrowedLine(axes_image, average_centroid_img, image_points[1],
                        //                 cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
                        // // Draw Y-axis in green
                        // cv::arrowedLine(axes_image, average_centroid_img, image_points[2],
                        //                 cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
                        // // Draw Z-axis in blue
                        // cv::arrowedLine(axes_image, average_centroid_img, image_points[3],
                        //                 cv::Scalar(255, 0, 0), 3, cv::LINE_AA);
                    }
                    else
                    {
                        ROS_WARN("Selected indices are out of range for points_to_average.");
                    }
                }
                else
                {
                    ROS_WARN("Not enough points to define X axis based on target points.");
                }
            }
            catch (tf2::TransformException &ex)
            {
                ROS_WARN("Failed to transform average normal vector: %s", ex.what());
            }
        }
        else
        {
            ROS_WARN("Plane fitting failed.");
        }
    }
    else
    {
        ROS_WARN("Not enough points to compute average normal vector.");
    }

    // Combine with the original image for better visibility
    cv::Mat combined_axes_image;
    cv::addWeighted(camera_image, 0.7, axes_image, 0.3, 0, combined_axes_image);

    // Log before showing the second window
    ROS_INFO("Displaying 'Centroids Axes' window.");
    cv::imshow("Centroids Axes", combined_axes_image);
    ROS_INFO("Displayed 'Centroids Axes' window.");

    // Prepare data for the service response
    if (centroids_positions.empty())
    {
        ROS_ERROR("No valid centroid positions were collected. Data will not be updated.");
        data_available = false;
        return;
    }

    // Clear previous data
    component_positions.clear();

    // Collect and transform positions of each centroid
    for (const auto &centroid_pt : centroids_positions)
    {
        geometry_msgs::PointStamped point_camera;
        point_camera.point.x = centroid_pt.x;
        point_camera.point.y = centroid_pt.y;
        point_camera.point.z = centroid_pt.z;
        point_camera.header.frame_id = cloud_msg->header.frame_id;
        point_camera.header.stamp = cloud_msg->header.stamp;

        geometry_msgs::PointStamped point_base;
        try
        {
            tf_buffer.transform(point_camera, point_base, "base_link", ros::Duration(1.0));
            component_positions.push_back(point_base.point);
        }
        catch (tf2::TransformException &ex)
        {
            ROS_WARN("Failed to transform centroid point: %s", ex.what());
            continue;
        }
    }

    // Store the average normal vector in 'base_link' frame
    average_normal = normal_vector_base.vector;

    // Set data_available flag
    data_available = true;

    // **Save the Initial Values if Not Already Saved**
    if (!initial_normals_saved && data_available)
    {
        initial_normal = average_normal;
        initial_x_axis = average_x_axis;
        initial_y_axis = average_y_axis;
        initial_component_positions = component_positions;
        initial_normals = normals; // **Added: Save initial normals**
        initial_normal_saved = true;
        initial_normals_saved = true; // **Added: Update the flag**

        ROS_INFO_STREAM("Initial normal and axes saved:");
        ROS_INFO_STREAM("Initial Normal: [" << initial_normal.x << ", " << initial_normal.y << ", " << initial_normal.z << "]");
        ROS_INFO_STREAM("Initial X Axis: [" << initial_x_axis.x << ", " << initial_x_axis.y << ", " << initial_x_axis.z << "]");
        ROS_INFO_STREAM("Initial Y Axis: [" << initial_y_axis.x << ", " << initial_y_axis.y << ", " << initial_y_axis.z << "]");
        ROS_INFO_STREAM("Number of Initial Normals: " << initial_normals.size());
    }

    // Log the data
    ROS_INFO_STREAM("Data updated: " << component_positions.size() << " component positions and average normal vector.");
}

/******************************************************************************
 * ROS SERVICE HANDLER
 *
 * Provides surface information (positions, normals, coordinate frame) to clients
 * via a ROS service interface
 *****************************************************************************/

// Service callback function
bool handle_average_normal(opencv_services_copy::all_normal::Request &req,
                           opencv_services_copy::all_normal::Response &res)
{
    if (data_available || initial_normals_saved)
    {
        // **Use the saved initial values if available**
        if (initial_normals_saved)
        {
            res.component_positions = initial_component_positions;
            res.normals = initial_normals; // **Set initial normals**
            res.average_normal = initial_normal;
            res.average_x_axis = initial_x_axis;
            res.average_y_axis = initial_y_axis;

            ROS_INFO_STREAM("Sending saved initial normal and axes:");
            ROS_INFO_STREAM("Initial Normal: [" << res.average_normal.x << ", " << res.average_normal.y << ", " << res.average_normal.z << "]");
            ROS_INFO_STREAM("Initial X Axis: [" << res.average_x_axis.x << ", " << res.average_x_axis.y << ", " << res.average_x_axis.z << "]");
            ROS_INFO_STREAM("Initial Y Axis: [" << res.average_y_axis.x << ", " << res.average_y_axis.y << ", " << res.average_y_axis.z << "]");
            ROS_INFO_STREAM("Number of Initial Normals: " << res.normals.size());
        }
        else
        {
            res.component_positions = component_positions;
            res.normals = normals; // **Set current normals**
            res.average_normal = average_normal;
            res.average_x_axis = average_x_axis;
            res.average_y_axis = average_y_axis;

            ROS_INFO_STREAM("Sending current normal and axes.");
            ROS_INFO_STREAM("Number of Current Normals: " << res.normals.size());
        }

        res.success = true;
        return true;
    }
    else
    {
        ROS_WARN("No data available to send in response.");
        res.success = false;
        return true;
    }
}

/******************************************************************************
 * MAIN FUNCTION
 *
 * Initializes the ROS node, sets up subscribers, publishers, and services,
 * and runs the main processing loop
 *****************************************************************************/

int main(int argc, char **argv)
{
    // Initialize the ROS node
    ros::init(argc, argv, "normal_estimation_node");
    ros::NodeHandle nh;

    // Initialize TF2 listener
    tf_listener_ptr = new tf2_ros::TransformListener(tf_buffer);

    // Initialize publisher for normal vectors as markers
    normal_pub = nh.advertise<visualization_msgs::Marker>("normal_vector_marker", 1);

    // Initialize the service server
    service = nh.advertiseService("all_normal_service", handle_average_normal); // **Updated service name**

    // Subscribe to image and point cloud topics using message_filters for synchronization
    message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, "/rgbd_camera/ir/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub(nh, "/rgbd_camera/depth/points", 1);

    // Synchronize the incoming messages
    message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(10), image_sub, cloud_sub);
    sync.registerCallback(boost::bind(&synchronized_callback, _1, _2));

    // OpenCV windows for visualization
    ROS_INFO("Creating OpenCV windows.");
    cv::namedWindow("Detected Centroids", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Centroids Axes", cv::WINDOW_AUTOSIZE);
    ROS_INFO("OpenCV windows created.");

    // Set a loop rate (e.g., 30 Hz)
    ros::Rate loop_rate(30);

    while (ros::ok())
    {
        ros::spinOnce();   // Process callbacks
        cv::waitKey(1);    // Handle OpenCV window events
        loop_rate.sleep(); // Sleep to maintain loop rate
    }

    // Clean up
    ROS_INFO("Destroying OpenCV windows.");
    cv::destroyAllWindows();
    if (tf_listener_ptr)
    {
        delete tf_listener_ptr;
    }

    ROS_INFO("Shutting down normal_estimation_node.");
    return 0;
}
