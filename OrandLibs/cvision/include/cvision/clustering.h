#ifndef CLUSTERING_H
#define CLUSTERING_H
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class Clustering
{
public:
    //data: each row represent a feature vector
    static double clus_kmeans_opencv(cv::Mat& data, cv::Mat& clusters, int K);
    static double clus_kmeans_opencv(cv::Mat &data, cv::Mat &clusters, int K, cv::Mat& bestLabels);
#ifndef NO_VLFEAT
    static double clus_kmeans_vlfeat(cv::Mat& data, cv::Mat& clusters, int K);
#endif
    static void saveClusters(std::string filename, cv::Mat& clusters);
    static void saveLabels(std::string filename, cv::Mat& labels);

    static cv::Mat readClusters(std::string filename);
    static float* readClusters(std::string filename, int *n_clusters, int *size_des);
    static void readLabels(std::string filename, cv::Mat& labels);
};

#endif // CLUSTERING_H
