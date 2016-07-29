#include "clustering.h"
#include <fstream>
#include <iostream>
#include "jmsr/JUtil.h"
#ifndef NO_VLFEAT
	extern "C" {
		#include <vl/generic.h>
		#include <vl/kmeans.h>
	}
#endif

using namespace std;
/*----------------------------------------------------------------------------*/
double Clustering::clus_kmeans_opencv(cv::Mat &data, cv::Mat &clusters, int K)
{
    cv::Mat bestLabels;
    double comp=cv::kmeans(data, K, bestLabels,
                    cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20000, 0.00001),
                    10,
                    cv::KMEANS_RANDOM_CENTERS, clusters);
    clusters.convertTo(clusters,CV_32F);
    return comp;
}
/*----------------------------------------------------------------------------*/
double Clustering::clus_kmeans_opencv(cv::Mat &data, cv::Mat &clusters, int K, cv::Mat& bestLabels)
{
    double comp=cv::kmeans(data, K, bestLabels,
                    cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001),
                    5,
                    cv::KMEANS_RANDOM_CENTERS, clusters);
    clusters.convertTo(clusters,CV_32F);
    return comp;
}
/*----------------------------------------------------------------------------*/
#ifndef NO_VLFEAT
	double Clustering::clus_kmeans_vlfeat(cv::Mat &data, cv::Mat &clusters, int K)
	{
		VlKMeans *kmeans;
		kmeans = vl_kmeans_new(VL_TYPE_FLOAT, VlDistanceL2);
		vl_kmeans_set_initialization(kmeans, VlKMeansPlusPlus);
		vl_kmeans_set_algorithm(kmeans, VlKMeansElkan);
		vl_kmeans_set_max_num_iterations(kmeans, 10000);
		vl_kmeans_set_num_repetitions(kmeans, 5);
		double comp=vl_kmeans_cluster(kmeans, data.data, data.cols, data.rows, K);
		clusters=cv::Mat(K, data.cols, CV_32F, kmeans->centers);
		return comp;
	}
#endif
/*----------------------------------------------------------------------------*/
//filename is a binary file, where the cluster data is stored
void Clustering::saveClusters(std::string filename, cv::Mat &clusters)
{
    int n_clusters=clusters.rows;
    int size_des=clusters.cols;
    //-------------------------------------Writing to a binary file
    ofstream f_out(filename.c_str(), std::ios::out|std::ios::binary);
    f_out.write(reinterpret_cast<char*>(&n_clusters), sizeof(int));
    f_out.write(reinterpret_cast<char*>(&size_des), sizeof(int));
    f_out.write(reinterpret_cast<char*>(clusters.data), size_des*n_clusters*sizeof(float));
    f_out.close();
    //-------------------------------------Writing to a textfile file
    //if(txt_file)//if a txt_file is required
    {
        ofstream f_out_txt((filename+".txt").c_str(), std::ios::out);
        f_out_txt<<n_clusters<<" "<<size_des<<std::endl;
        for(int i=0; i<n_clusters; i++)
        {
            for(int j=0; j<size_des; j++)
            {
                f_out_txt<<clusters.at<float>(i,j)<<" ";
            }
            f_out_txt<<std::endl;
        }
        f_out_txt.close();
    }
}
/*----------------------------------------------------------------------------*/
void Clustering::saveLabels(std::string filename, cv::Mat& labels){
	int n_rows=labels.rows;
	int n_cols=labels.cols;
	int n_bytes=0;
	int depth=labels.depth();
	if(depth==CV_8U) n_bytes=1;
	else if (depth==CV_8S) n_bytes=1;
	else if (depth==CV_16U) n_bytes=2;
	else if (depth==CV_16S) n_bytes=2;
	else if (depth==CV_32S) n_bytes=4;
	else if (depth==CV_32F) n_bytes=4;
	else if (depth==CV_64F) n_bytes=8;
	else{
		std::cerr<<"unrecognized type for labels"<<std::endl;
		exit(EXIT_FAILURE);
	}
	ofstream f_out(filename.c_str(), ios::out | ios::binary);
	assert(f_out.is_open());
	f_out.write(reinterpret_cast<char*>(&n_rows), sizeof(int));
	f_out.write(reinterpret_cast<char*>(&n_cols), sizeof(int));
	f_out.write(reinterpret_cast<char*>(&depth), sizeof(int));
	f_out.write(reinterpret_cast<char*>(labels.data), n_rows*n_cols*n_bytes);
	f_out.close();
}
/*----------------------------------------------------------------------------*/
//filename is a binary filename, produced by saveClusters
cv::Mat Clustering::readClusters(std::string filename)
{
    int n_clusters=0, size_des=0;
    ifstream f_in(filename.c_str(), std::ios::in | std::ios::binary);
    assert(f_in.is_open());
    f_in.read(reinterpret_cast<char*>(&n_clusters), sizeof(int));
    f_in.read(reinterpret_cast<char*>(&size_des), sizeof(int));
    cv::Mat clusters(n_clusters, size_des, CV_32F);
    f_in.read(reinterpret_cast<char*>(clusters.data), size_des*n_clusters*sizeof(float));
    f_in.close();
    return clusters;
}
/*----------------------------------------------------------------------------*/
//filename is a binary filename, produced by saveClusters
float* Clustering::readClusters(std::string filename, int* n_clusters, int* size_des)
{

    ifstream f_in(filename, std::ios::in| std::ios::binary);
    JUtil::jmsr_assert(f_in.is_open(),filename + " can't be opened");
    f_in.read(reinterpret_cast<char*>(n_clusters), sizeof(int));
    f_in.read(reinterpret_cast<char*>(size_des), sizeof(int));

    int data_length=(*n_clusters)*(*size_des);
    float* data_clusters=new float[data_length];
    for(int i=0; i<data_length;i++) data_clusters[i]=0;

    f_in.read(reinterpret_cast<char*>(data_clusters), data_length*sizeof(float));

    f_in.close();
    return data_clusters;
}
/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Clustering::readLabels(std::string filename, cv::Mat& labels){

	ifstream f_in(filename.c_str(), ios::in | ios::binary);
	assert(f_in.is_open());

	int n_rows=0;
	int n_cols=0;
	int n_bytes=0;
	int depth=0;
	f_in.read(reinterpret_cast<char*>(&n_rows), sizeof(int));
	f_in.read(reinterpret_cast<char*>(&n_cols), sizeof(int));
	f_in.read(reinterpret_cast<char*>(&depth), sizeof(int));

	if(depth==CV_8U) n_bytes=1;
	else if (depth==CV_8S) n_bytes=1;
	else if (depth==CV_16U) n_bytes=2;
	else if (depth==CV_16S) n_bytes=2;
	else if (depth==CV_32S) n_bytes=4;
	else if (depth==CV_32F) n_bytes=4;
	else if (depth==CV_64F) n_bytes=8;
	else{
		std::cerr<<"unrecognized type for labels"<<std::endl;
		exit(EXIT_FAILURE);
	}
	labels.create(n_rows, n_cols, depth);
	f_in.read(reinterpret_cast<char*>(labels.data), n_rows*n_cols*n_bytes);
	f_in.close();
}
