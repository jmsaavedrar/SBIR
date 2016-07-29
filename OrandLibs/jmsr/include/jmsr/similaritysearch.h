/* Jose M. Saavedra Rondo Julio 2013
   This is the definition of SimilaritySearch class
   that uses  the flann lib for efficient search
  */
#ifndef SIMILARITYSEARCH_H
#define SIMILARITYSEARCH_H

#include <opencv2/highgui/highgui.hpp>
#include <flann/flann.hpp>
#include <string>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <map>
#include <mutex>
#include "JDistance.h"

#define J_LINEAR 1
#define J_KDTREE 2

#define J_INDEX_LINEAR 10
#define J_INDEX_FLANN 20

/*--------------------------------------------------------------------------*/
// PairResult is used to represent a pair (idx, dist). This must be used in the future
// Using this pair, sorting becomes very easy
class PairResult
{
public:
    int idx;
    float dist;
    float score;
    PairResult()
    {
        idx=0; dist=0; score=0;
    }
    //---------------------------------------------------------------
    PairResult(int _idx, float _dist ,float _score=0)
    {
        idx=_idx;
        dist=_dist;
        score=_score;
    }
    //---------------------------------------------------------------
    static bool comparePairResult(const PairResult& pair_A, const PairResult& pair_B)
    {
        return (pair_A.dist<pair_B.dist);
    }
    //---------------------------------------------------------------
    static bool comparePairResultByScore(const PairResult& pair_A, const PairResult& pair_B)
    {
    	return (pair_A.score>pair_B.score);
    }
};
/*--------------------------------------------------------------------------*/
class ResultQuery
{
public:
    std::vector<int> idx;
    std::vector<float> distances;
    std::vector<std::string> names; //optional
    ResultQuery();
    ResultQuery(int K);
    int getSize();
    static ResultQuery pooling(std::vector<ResultQuery*> list, std::vector<float> weights, int K);
    static ResultQuery poolingWithSubset(std::vector<ResultQuery*> list, std::vector<float> weights, int K);
    static ResultQuery merge(ResultQuery& result_A, ResultQuery& result_B, int K);
    static ResultQuery merge_optimized(ResultQuery& result_A, ResultQuery& result_B, int K, int size_db);
    static ResultQuery mergeWithSubset(ResultQuery& result_A, ResultQuery& result_B,  int K);
    static ResultQuery mergeSubsetCompleteByIntercalation(ResultQuery& result_st, ResultQuery& result_c, int n_step);
    static ResultQuery mergeSubsetCompleteByRanking(ResultQuery& result_st, ResultQuery& result_c,  int n_max_subset);
};

/*--------------------------------------------------------------------------*/
template <typename Distance> // Distances of type  <flann::L1<float> >
class SimilaritySearch
{
private:
    std::vector<std::string> image_names; //store the names of each objetct in the index
    std::vector<std::string> image_clases;//store the class names of each object in the index
    std::string index_name; //this is not used now, but represet a name of the index saving in the filesystem
    float *db_vectors;
    flann::Index<Distance> *index; //this is the index
    int db_size;//number of objects
    int db_dim;//dimension of each descriptor
    std::mutex sim_mutex;
    void knnsearch(float *query, int query_size,
    		std::vector<std::vector<int> >& indices, std::vector<std::vector<float> >& dists, int K);
public:
    SimilaritySearch();
    ~SimilaritySearch();    
    SimilaritySearch(std::vector<std::string> _image_names, std::vector<std::string> _image_clases, float *db, int db_rows, int db_cols, int type=J_KDTREE);
    SimilaritySearch(std::vector<std::string> _image_names, std::vector<std::string> _image_clases, cv::Mat db, int type=J_KDTREE);
    SimilaritySearch(float *db, int db_rows, int db_cols,int type=J_KDTREE);
    void init(float *db, int db_rows, int db_cols, int type);
    std::string getName(int idx); //return the image name given an index, idx position on the dataset (not a rank)
    std::string getClass(int idx); // return the image class given the index
    int getDBSize(); //Get the number of objects
    int getDim();
    ResultQuery searchBySimilarity(float *query, int query_size, int K=-1);//-1  = all
    ResultQuery searchBySimilarity(cv::Mat mat_query, int K=-1);//-1  = all
    void releaseIndex();
/*--------------------------------------------------------------------------*/
};
/*--------------------------------------------------------------------------*/
template <class Distance>
class JLinearSearch
{
private:
    std::vector<std::string> image_names;
    std::vector<std::string> image_clases;
    float *db_vectors; //it'll reside in memory
    std::map<std::string, int> cod_idx;
    int db_size;//number of objects
    int db_dim;//dimension of each descriptor
public:
    JLinearSearch();
    ~JLinearSearch();
    JLinearSearch(std::vector<std::string> _image_names, std::vector<std::string> _image_clases, float *db, int db_rows, int db_cols);
    std::string getName(int idx);
    std::string getClass(int idx);
    int getDBSize();
    int getDim();
    ResultQuery searchBySimilarity(float *query, int query_size, int K=-1);//-1  = all
    ResultQuery searchBySimilarity(float *query, int query_size, std::vector<std::string> subset, int K=-1);//-1  = all
    ResultQuery searchBySimilarity(cv::Mat mat_query, int K=-1);//-1  = all
    ResultQuery searchBySimilarity(cv::Mat mat_query, std::vector<std::string> subset, int K=-1);//-1  = all
    void releaseIndex();
};
/*--------------------------------------------------------------------------*/
template <class Index>
class Ranking{
	private:
		std::vector<std::string> ids_vector;
		std::vector<std::string> classes_vector;
		std::vector<float> dists_vector;
		std::string id_query;
		std::string class_query;
		std::string target_query;
		std::vector<float> precision_vec;
		float mAP;
	public:
	Ranking();
		void create(ResultQuery& rstQ, Index* index,
				std::string _id_query, std::string _class_query, std::string _target_query);
		std::string getId(int idx);
		std::string getClass(int idx);
		float getDist(int idx);
		void save(std::string str_result_dir, std::string sTipo);
		float getMAP();
		void getNormalizedPrecision(float PR[11]);
};
/*--------------------------------------------------------------------------*/

#endif // SIMILARITYSEARCH_H
