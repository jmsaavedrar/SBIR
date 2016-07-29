/* Jose M. Saavedra Rondo Julio 2013
 This is the implementation of SimilaritySearch class
 that uses  the flann lib for efficient search
 */
#include "similaritysearch.h"
#include "JUtil.h"
#include <iostream>
#include <fstream>
#define J_MAX_DIST 1000000;


ResultQuery::ResultQuery(): idx(0),distances(0){

}
/*-------------------------------------------------------------------------------*/
ResultQuery::ResultQuery(int K)
{
	idx.resize(K);
	distances.resize(K);
	names.resize(K);
}
/*-------------------------------------------------------------------------------*/
int ResultQuery::getSize() //number of items
{
	return idx.size();
}
/*-------------------------------------------------------------------------------*/
ResultQuery ResultQuery::pooling(std::vector<ResultQuery*> list, std::vector<float> weights, int K)
{
	ResultQuery result(K);
	if(list.size()==0) return result;
	if(list.size()==1) {
		for(int i=0; i<K; i++)
		{
			result.idx[i]=list[0]->idx[i];
			result.distances[i]=list[0]->distances[i];
		}
		return result;
	}
	//All the resultQuery are sorted
	int n_results=list[0]->getSize(); //get the size of the result query
	std::vector<PairResult> pooled_result(n_results);
	for (int ir=0; ir<n_results; ir++)
	{
		pooled_result[ir].idx=ir;
		pooled_result[ir].dist=0;
	}
	//std::cout<<" --> Pooling  result queries"<<std::endl;
	for (unsigned int i=0; i<list.size(); i++)
	{
		if(list[i]->getSize()!=n_results)
		{
			std::cerr<<"Pooling Error: The size of result queries are incompatibles!"<<std::endl;
			exit(EXIT_FAILURE);
		}
		//std::cout<<" --> Processing result query "<<i<<" with weight="<<weights[i]<<std::endl;
		for (int ir=0; ir<n_results; ir++)
		{
			pooled_result[list[i]->idx[ir]].dist+=ir*weights[i];
		}
		//std::cout<<" <-- Processing result query "<<i<<std::endl;
	}
   // std::cout<<" --> Sorting pooled result query"<<std::endl;
	std::sort(pooled_result.begin(), pooled_result.end(),PairResult::comparePairResult);
	//std::cout<<" <-- Sorting pooled result query"<<std::endl;
	//std::cout<<" --> Bulding pooled ResultQuery  object"<<std::endl;
	for(int i=0; i<K; i++)
	{
		result.idx[i]=pooled_result[i].idx;
		result.distances[i]=pooled_result[i].dist;
	}
	std::cout<<" <-- Building pooled ResultQuery object"<<std::endl;
	return result;
}
/*-------------------------------------------------------------------------------*/
ResultQuery ResultQuery::poolingWithSubset(std::vector<ResultQuery*> list, std::vector<float> weights, int K)
{
	ResultQuery result;
	if(list.size()==0) return result;
	//All the resultQuery are sorted
	int n_results=list[0]->getSize(); //get the size of the result query
	std::vector<PairResult> pooled_result(n_results);
	std::map<int, int> idx_idx1;
	int idx=0, idx1=0;
	for (int ir=0; ir<n_results; ir++)
	{
		idx=list[0]->idx[ir];
		idx_idx1[idx]=ir;

		pooled_result[ir].idx=idx;
		pooled_result[ir].dist=ir*weights[0];
	}
	for (unsigned int i=1; i<list.size(); i++)
	{
		if(list[i]->getSize()!=n_results)
		{
			std::cerr<<"Pooling Error: The size of result queries are incompatibles!"<<std::endl;
			exit(EXIT_FAILURE);
		}
		for (int ir=0; ir<n_results; ir++)
		{
			idx1=idx_idx1[list[i]->idx[ir]];
			pooled_result[idx1].dist+=ir*weights[i];
		}
	}

	std::sort(pooled_result.begin(), pooled_result.end(),PairResult::comparePairResult);

	K=std::min(K, n_results);
	result=ResultQuery(K);
	for(int i=0; i<K; i++)
	{
		result.idx[i]=pooled_result[i].idx;
		result.distances[i]=pooled_result[i].dist;
	}
	std::cout<<" <-- Building pooled ResultQuery object"<<std::endl;
	return result;
}
/*-------------------------------------------------------------------------------*/
ResultQuery ResultQuery::mergeWithSubset(ResultQuery& result_A, ResultQuery& result_B,  int K)
{
	int n_total=result_A.getSize();
	std::vector<PairResult> merged_result(n_total);
	std::map<int, int> idx_idx1;
	int count=0;
	int idx=0;

	for(int i=0; i<result_A.getSize();i++)
	{
		merged_result[count].idx=result_A.idx[i];
		merged_result[count].dist=result_A.distances[i];
		idx_idx1[result_A.idx[i]]=count;
		count++;
	}

	for(int i=0; i<result_B.getSize();i++)
	{
		if(idx_idx1.find(result_B.idx[i])==idx_idx1.end())
		{
			std::cerr<<"ERROR: There the two subsets are different!!!"<<std::endl;
			exit(EXIT_FAILURE);
		}
		idx=idx_idx1[result_B.idx[i]];
		merged_result[idx].dist=std::min(merged_result[idx].dist, result_B.distances[i]);

	}
	std::cout<<" <-- Sorting"<<std::endl;
	std::sort(merged_result.begin(), merged_result.end(),PairResult::comparePairResult);
	K=std::min(K,n_total);
	std::cout<<" <-- copying"<<K<<std::endl;
	ResultQuery result(K);
	for(int i=0; i<K; i++)
	{
		result.idx[i]=merged_result[i].idx;
		result.distances[i]=merged_result[i].dist;
	}
	return result;
}
/*-------------------------------------------------------------------------------*/
//result_A, and result_B must be complete rankings
//The idxs on each result must contain all idxs of the underlying dataset
//In this implementation MIN merge is used
ResultQuery ResultQuery::merge(ResultQuery& result_A, ResultQuery& result_B, int K)
{
	int n_results=result_A.getSize();
	if(result_B.getSize()!=n_results)
	{
		std::cerr<<"Pooling Error: The size of result queries are incompatibles!"<<std::endl;
		exit(EXIT_FAILURE);
	}
	std::vector<PairResult> merged_result(n_results);
	//--------------------------------- Inializing merged_result
	std::cout<<" <-- Inializing merged_result"<<std::endl;
	for (int ir=0; ir<n_results; ir++)
	{
		merged_result[ir].idx=ir;
		merged_result[ir].dist=J_MAX_DIST;
	}
	//--------------------------------- Copying the distances of the result_A
	std::cout<<" <-- Copying the distances of the result_A"<<std::endl;
	for (int ir=0; ir<n_results; ir++)
	{
		merged_result[result_A.idx[ir]].dist=result_A.distances[ir];
		//std::cout<<result_A.idx[ir]<<" d="<<result_A.distances[ir];
	}
	float d=0;
	std::cout<<" <-- Merge"<<std::endl;
	//--------------------------------- Merge with the result_B
	for (int ir=0; ir<n_results; ir++)
	{
		d=merged_result[result_B.idx[ir]].dist;
		merged_result[result_B.idx[ir]].dist=std::min(d,result_B.distances[ir]);
		//std::cout<<result_B.idx[ir]<<" d="<<std::min(d,result_B.distances[ir]);
	}
	std::cout<<" <-- Sorting"<<std::endl;
	//--------------------------------- sorting merged_result
	std::sort(merged_result.begin(), merged_result.end(),PairResult::comparePairResult);
	//--------------------------------- copying to a ResultQuery object
	K=std::min(K,n_results);
	std::cout<<" <-- copying"<<std::endl;
	ResultQuery result(K);
	std::cout<<K<<std::endl;
	for(int i=0; i<K; i++)
	{
		result.idx[i]=merged_result[i].idx;
		result.distances[i]=merged_result[i].dist;
		//std::cout<<result.idx[i]<<" "<<result.distances[i]<<std::endl;
	}
	return result;
}
/*-------------------------------------------------------------------------------*/
//result_A, and result_B must be complete rankings
//The idxs on each result must contain all idxs of the underlying dataset
//In this implementation MIN merge is used
ResultQuery ResultQuery::merge_optimized(ResultQuery& result_A, ResultQuery& result_B, int K, int size_db)
{
    int n_results=size_db;
    if(result_B.getSize()!=result_B.getSize())
    {
        std::cerr<<"Pooling Error: The size of result queries are incompatibles!"<<std::endl;
        exit(EXIT_FAILURE);
    }
    std::vector<PairResult> merged_result(n_results);
    //--------------------------------- Inializing merged_result
    //std::cout<<" <-- Inializing merged_result"<<std::endl;
    for (int ir=0; ir<n_results; ir++)
    {
        merged_result[ir].idx=ir;
        merged_result[ir].dist=result_B.getSize();
    }
    //--------------------------------- Copying the distances of the result_A
    //std::cout<<" <-- Copying the distances of the result_A"<<std::endl;
    for (int ir=0; ir<result_A.getSize(); ir++)
    {
        merged_result[result_A.idx[ir]].dist=result_A.distances[ir];
    }
    float d=0;
    //std::cout<<" <-- Merge"<<std::endl;
    //--------------------------------- Merge wiht the result_B
    for (int ir=0; ir<result_B.getSize(); ir++)
    {

        d=merged_result[result_B.idx[ir]].dist;
        merged_result[result_B.idx[ir]].dist=std::min(d,result_B.distances[ir]);
    }
    //std::cout<<" <-- Sorting"<<std::endl;
    //--------------------------------- sorting merged_result
    std::sort(merged_result.begin(), merged_result.end(),PairResult::comparePairResult);
    //--------------------------------- copying to a ResultQuery object
    //std::cout<<" <-- copying"<<std::endl;
   // std::cout<<K<<std::endl;
    ResultQuery result(K);
    //std::cout<<K<<std::endl;
    for(int i=0; i<K; i++)
    {
        result.idx[i]=merged_result[i].idx;
        result.distances[i]=merged_result[i].dist;
        //std::cout<<result.idx[i]<<" "<<result.distances[i]<<std::endl;
    }
    return result;
}
/*-------------------------------------------------------------------------------*/
/**
 *
 * @param result_st
 * @param result_c
 * @param n_step: step for intercalate results
 * @return
 */
ResultQuery ResultQuery::mergeSubsetCompleteByIntercalation(ResultQuery& result_st, ResultQuery& result_c,  int n_step)
{
	int K=std::max(result_st.getSize(), result_c.getSize());
	std::map <int, int> id_product2idx;
	ResultQuery result(K);
	int i_st=0, i_c=0, i_result=0, i=0;
	i=0;
	i_st=0;
	i_c=0;
	while(i_result<K && (i_st<result_st.getSize() || i_c<result_c.getSize())){
		i=0;
		while(i<n_step && i_st<result_st.getSize() && i_result<K){
			if(id_product2idx.find(result_st.idx[i_st])==id_product2idx.end()){ //product does not exist
				id_product2idx[result_st.idx[i_st]]=i_result;
				result.idx[i_result]=result_st.idx[i_st];
				result.distances[i_result]=result_st.distances[i_st];
				i_result++;
				i++;
			}
			i_st++;
		}
		i=0;
		while(i<n_step && i_c<result_c.getSize() && i_result<K){
			if(id_product2idx.find(result_c.idx[i_c])==id_product2idx.end()){ //product does not exist
				id_product2idx[result_c.idx[i_c]]=i_result;
				result.idx[i_result]=result_c.idx[i_c];
				result.distances[i_result]=result_c.distances[i_c];
				i_result++;
				i++;
			}
			i_c++;
		}
	}
	//resize K
	return result;
}
/*-------------------------------------------------------------------------------*/
/**
 *
 * @param result_st
 * @param result_c
 * @param n_max_subset: limit number from result_st
 * @return
 */
ResultQuery ResultQuery::mergeSubsetCompleteByRanking(ResultQuery& result_st, ResultQuery& result_c,  int n_max_subset)
{
	int K=std::max(result_st.getSize(), result_c.getSize());
	std::map <int, int> id_product2idx;
	std::vector<PairResult> merged_result(K);
	int i_st=0, i_c=0, count_results=0, i=0, idx_result=0;
	i=0;
	i_st=0;
	n_max_subset=std::min(n_max_subset, result_st.getSize());
	while(i<n_max_subset && count_results<K)
	{
		if(id_product2idx.find(result_st.idx[i_st])==id_product2idx.end()) //product does not exist
		{
			idx_result=count_results;
			id_product2idx[result_st.idx[i_st]]=idx_result;
			merged_result[idx_result].idx=result_st.idx[i_st];
			merged_result[idx_result].dist=result_st.distances[i_st];
			count_results++;
			i++;
		}
		i_st++;
	}
	i_c=0;
	while(i_c<result_c.getSize() && count_results<K)
	{
		if(id_product2idx.find(result_c.idx[i_c])==id_product2idx.end()) //product does not exist
		{
			idx_result=count_results;
			id_product2idx[result_c.idx[i_c]]=idx_result;
			merged_result[idx_result].idx=result_c.idx[i_c];
			merged_result[idx_result].dist=result_c.distances[i_c];
			count_results++;
		}
		else
		{
			idx_result=id_product2idx[result_c.idx[i_c]];
			merged_result[idx_result].dist=std::min(merged_result[idx_result].dist, result_c.distances[i_c]);
		}
		i_c++;
	}
	merged_result.resize(count_results);
	std::sort(merged_result.begin(), merged_result.end(),PairResult::comparePairResult);
	//--------------------------------- copying to a ResultQuery object
	ResultQuery result(count_results);
	for(int i=0; i<count_results; i++)
	{
		result.idx[i]=merged_result[i].idx;
		result.distances[i]=merged_result[i].dist;
	}
	return result;
}
/*-------------------------------------------------------------------------------*/
template<typename Distance>
SimilaritySearch<Distance>::SimilaritySearch()
{
	image_names.clear();
	image_clases.clear();
	index = NULL;
	db_size = -1;
	db_dim = 0;
}
/*-------------------------------------------------------------------------------*/
//In this case, the input is a float*
template<typename Distance>
SimilaritySearch<Distance>::SimilaritySearch(
		std::vector<std::string> _image_names,
		std::vector<std::string> _image_clases, float *db, int db_rows,
		int db_cols, int type)
{
	db_vectors=db;
	image_names = _image_names;
	image_clases = _image_clases;
	init(db, db_rows, db_cols, type);
}
/*-------------------------------------------------------------------------------*/
//In this case, the input is a Mat
template<typename Distance>
SimilaritySearch<Distance>::SimilaritySearch(
		std::vector<std::string> _image_names,
		std::vector<std::string> _image_clases, cv::Mat db, int type)
{
	//Be careful, this uses the same buffer of db.data, if db is releases, the index'll fail
	if (db.type() != CV_32F)
	{
		std::cerr << "Error: The input is not a float matrix" << std::endl;
		exit(EXIT_FAILURE);
	}
	image_names = _image_names;
	image_clases = _image_clases;
	float *data=reinterpret_cast<float*>(db.data);
	//db_vectors=data;
	init(data, db.rows, db.cols, type);
}
/*-------------------------------------------------------------------------------*/
template<typename Distance>
SimilaritySearch<Distance>::SimilaritySearch(float *db, int db_rows,
		int db_cols, int type)
{
	db_vectors=db;
	image_names.clear();
	image_clases.clear();
	init(db, db_rows, db_cols, type);
}
/*-------------------------------------------------------------------------------*/
template<typename Distance>
void  SimilaritySearch<Distance>::init(float *db, int db_rows,
		int db_cols, int type){
	db_size = db_rows;
	db_dim = db_cols;
	flann::Matrix<float> datos(db, db_rows, db_cols);
	if (type == J_LINEAR)
		index = new flann::Index<Distance>(datos, flann::LinearIndexParams()); //implements a Linear idnex
	else if (type == J_KDTREE)
		index = new flann::Index<Distance>(datos, flann::KDTreeIndexParams()); //implements a KDTree index
	index->buildIndex();
}

/*-------------------------------------------------------------------------------*/
template<typename Distance>
std::string SimilaritySearch<Distance>::getClass(int idx)
{
	if (idx < (int) image_clases.size())
		return image_clases[idx];
	else
		return "";
}
/*-------------------------------------------------------------------------------*/
template<typename Distance>
std::string SimilaritySearch<Distance>::getName(int idx)
{
	if (idx < (int) image_names.size())
		return image_names[idx];
	else
		return "";
}
/*-------------------------------------------------------------------------------*/
template<typename Distance>
int SimilaritySearch<Distance>::getDBSize()
{
	return db_size;
}
/*-------------------------------------------------------------------------------*/
template<typename Distance>
int SimilaritySearch<Distance>::getDim()
{
	return db_dim;
}
/*-------------------------------------------------------------------------------*/
template<typename Distance>
void SimilaritySearch<Distance>::knnsearch(float *query, int query_size, std::vector<std::vector<int> >& indices,
		std::vector<std::vector<float> >& dists, int K){
	//std::lock_guard<std::mutex> lck_flann(sim_mutex);
	flann::Matrix<float> f_query(query, 1, query_size);
	index->knnSearch(f_query, indices, dists, K, flann::SearchParams(128));
}
/*-------------------------------------------------------------------------------*/
template<typename Distance>
ResultQuery SimilaritySearch<Distance>::searchBySimilarity(float *query,
		int query_size, int K)
{
	ResultQuery result;
	if (K == -1)
	{
		K = db_size;
	}
	//std::cout<<"querying in index"<<std::endl;
	std::vector<std::vector<int> > indices;
	std::vector<std::vector<float> > dists;
	if (db_dim != query_size)
	{
		std::cout << "db_dim=" << db_dim << " query_dim=" << query_size
				<< std::endl;
		std::cout << "ERROR: incompatible sizes between query an db!"
				<< std::endl;
		exit(EXIT_FAILURE);
	}
	//std::cout<<index->veclen()<<" "<<query_size<<std::endl;
	//--flann::Matrix<float> f_query(query, 1, query_size);
	//--index->knnSearch(f_query, indices, dists, K, flann::SearchParams(128));
	knnsearch(query, query_size, indices, dists, K);
	//std::cout<<"end querying in index"<<std::endl;
	result.idx = indices[0];
	result.distances = dists[0];

	return result;
}
/*-------------------------------------------------------------------------------*/
template<typename Distance>
ResultQuery SimilaritySearch<Distance>::searchBySimilarity(cv::Mat mat_query,
		int K)
{
	//Solo funciona si mat_query es CV_32F
	ResultQuery result;
	if (K == -1)
	{
		K = db_size;
	}
	if (mat_query.rows != 1)
	{
		std::cout << "ERROR: The query for the index has multi-rows!!!"
				<< std::endl;
		exit(EXIT_FAILURE);
	}
	int query_size = mat_query.cols;
	//std::cout<<"querying in index"<<std::endl;
	std::vector<std::vector<int> > indices;
	std::vector<std::vector<float> > dists;
	if (db_dim != query_size)
	{
		std::cout << "db_dim=" << db_dim << " query_dim=" << query_size
				<< std::endl;
		std::cout << "ERROR: incompatible sizes between query an db!"
				<< std::endl;
		exit(EXIT_FAILURE);
	}
	//std::cout<<index->veclen()<<" "<<query_size<<std::endl;
	//---flann::Matrix<float> f_query(reinterpret_cast<float*>(mat_query.data), 1,query_size);
	//--index->knnSearch(f_query, indices, dists, K, flann::SearchParams(128));
	knnsearch(reinterpret_cast<float*>(mat_query.data), query_size, indices, dists, K);
	//---------
	//std::cout<<"end querying in index"<<std::endl;
	result.idx = indices[0];
	result.distances = dists[0];
	return result;
}
/*-------------------------------------------------------------------------------*/
template<typename Distance>
SimilaritySearch<Distance>::~SimilaritySearch()
{
	//  delete index;
	// Please use release index to deallocate the index
}
/*-------------------------------------------------------------------------------*/
template<typename Distance>
void SimilaritySearch<Distance>::releaseIndex()
{    //analizar el borrado de las matrices de flann con delete matrix.ptr
	delete index;
	delete[] db_vectors;
}

/*-------------------------------------------------------------------------------*/

template<class Distance>
JLinearSearch<Distance>::JLinearSearch()
{
	image_names.clear();
	image_clases.clear();
	db_vectors = NULL;
	db_size = -1;
	db_dim = 0;
}

template<class Distance>
JLinearSearch<Distance>::JLinearSearch(std::vector<std::string> _image_names,
		std::vector<std::string> _image_clases, float *db, int db_rows,
		int db_cols)
{
	image_names = _image_names;
	image_clases = _image_clases;
	db_size = db_rows;
	db_dim = db_cols;
	int total_size = db_size * db_dim;
	db_vectors = new float[total_size];
	std::copy(db, db + total_size, db_vectors);
	for (int i = 0; i < image_names.size(); i++)
	{
		cod_idx[image_names[i]] = i;
	}
}
/*-------------------------------------------------------------------------------*/
template<class Distance>
std::string JLinearSearch<Distance>::getClass(int idx)
{
	if (idx < (int) image_clases.size())
		return image_clases[idx];
	else
		return "";
}
/*-------------------------------------------------------------------------------*/
template<class Distance>
std::string JLinearSearch<Distance>::getName(int idx)
{
	if (idx < (int) image_names.size())
		return image_names[idx];
	else
		return "";
}
/*-------------------------------------------------------------------------------*/
template<class Distance>
int JLinearSearch<Distance>::getDBSize()
{
	return db_size;
}
/*-------------------------------------------------------------------------------*/
template<class Distance>
int JLinearSearch<Distance>::getDim()
{
	return db_dim;
}
/*-------------------------------------------------------------------------------*/
template<class Distance>
ResultQuery JLinearSearch<Distance>::searchBySimilarity(float *query,
		int query_size, int K)
{
	if ((K == -1)||(K>db_size))
		K = db_size;
	std::vector<PairResult> ranking(db_size);
	float *vector_test=NULL;
	if (query_size != db_dim)
	{
		std::cerr << "JLinearSearch: query_size!=db_dim" << std::endl;
	}
	Distance obj_dist;
	float dist = 0;

	for (int i = 0; i < db_size; i++)
	{
		vector_test=db_vectors+i*db_dim;
		dist = obj_dist.getDistance(query, vector_test, db_dim);
		ranking[i].idx=i;
		ranking[i].dist=dist;
	}
	std::sort(ranking.begin(), ranking.end(), PairResult::comparePairResult);
	ResultQuery result(K);
	for (int i = 0; i < K; i++)
	{
		result.idx[i] = ranking[i].idx;
		result.distances[i] = ranking[i].dist;
	}
	return result;
}
/*-------------------------------------------------------------------------------*/
template<class Distance>
ResultQuery JLinearSearch<Distance>::searchBySimilarity(float *query,
		int query_size, std::vector<std::string> subset, int K)
{
	if ((K == -1)||(K>subset.size()))
		K = subset.size();
	std::vector<PairResult> ranking(subset.size());
	float *vector_test=NULL;
	int idx = 0;
	if (query_size != db_dim)
	{
		std::cerr << "JLinearSearch: query_size!=db_dim" << std::endl;
	}
	Distance obj_dist;
	float dist = 0;

	for (int i = 0; i < subset.size(); i++)
	{
		idx = cod_idx[subset[i]];
		//std::cout<<subset[i]<<" "<<image_names[idx]<<std::endl;
		vector_test=db_vectors+idx*db_dim;
		dist = obj_dist.getDistance(query, vector_test, db_dim);
		ranking[i].idx=idx;
		ranking[i].dist=dist;
	}
	std::sort(ranking.begin(), ranking.end(), PairResult::comparePairResult);
	ResultQuery result(K);
	for (int i = 0; i < K; i++)
	{
		result.idx[i] = ranking[i].idx;
		result.distances[i] = ranking[i].dist;
	}
	return result;
}
/*-------------------------------------------------------------------------------*/
template<class Distance>
ResultQuery JLinearSearch<Distance>::searchBySimilarity(cv::Mat mat_query, int K)
{
	int query_size=mat_query.cols;
	return searchBySimilarity(reinterpret_cast<float*>(mat_query.data), query_size, K);
}
/*-------------------------------------------------------------------------------*/
template<class Distance>
ResultQuery JLinearSearch<Distance>::searchBySimilarity(cv::Mat mat_query, std::vector<std::string> subset, int K)
{
	int query_size=mat_query.cols;
	return searchBySimilarity(reinterpret_cast<float*>(mat_query.data), query_size, subset, K);
}
/*-------------------------------------------------------------------------------*/
template<class Distance>
void JLinearSearch<Distance>::releaseIndex()
{
	delete[] db_vectors;
}
/*-------------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------*/
template<class Distance>
JLinearSearch<Distance>::~JLinearSearch()
{
}
/*-------------------------------------------------------------------------------*/
template<class Index>
Ranking<Index>::Ranking():ids_vector(0), classes_vector(0), dists_vector(0),
id_query("NULL"), class_query("NULL"), target_query("NULL"), precision_vec(0), mAP(-1){
}
/*-------------------------------------------------------------------------------*/
template<class Index>
void Ranking<Index>::create(ResultQuery& rstQ, Index* index,
		std::string _id_query, std::string _class_query, std::string _target_query){
	int vec_size=rstQ.getSize();
	ids_vector.resize(vec_size);
	classes_vector.resize(vec_size);
	dists_vector.resize(vec_size);
	id_query=_id_query;
	class_query=_class_query;
	target_query=_target_query;
	mAP=0;
	float pr=0;
	int n_relevants=0;
	precision_vec.clear();
	for(int i=0; i<vec_size;i++){
		ids_vector[i]=index->getName(rstQ.idx[i]);
		classes_vector[i]=index->getClass(rstQ.idx[i]);
		dists_vector[i]=rstQ.distances[i];
		if(class_query.compare(classes_vector[i])==0){
			n_relevants++;
			pr=((float)n_relevants/(i+1));
			mAP+=pr;
			precision_vec.push_back(pr);
		}
	}
	mAP=mAP/n_relevants;
}
/*-------------------------------------------------------------------------------*/
template<class Index>
std::string Ranking<Index>::getId(int idx){
	return ids_vector[idx];
}
/*-------------------------------------------------------------------------------*/
template<class Index>
std::string Ranking<Index>::getClass(int idx){
	return classes_vector[idx];
}
/*-------------------------------------------------------------------------------*/
template<class Index>
float Ranking<Index>::getDist(int idx){
	return dists_vector[idx];
}
/*-------------------------------------------------------------------------------*/
template<class Index>
void Ranking<Index>::save(std::string str_result_dir, std::string sTipo){
	std::string str_file=str_result_dir+"/"+id_query+"_"+sTipo+".ranking";
	std::ofstream f_out(str_file);
	assert(f_out.is_open());
	std::cout<<"copy to: "<<str_file<<std::endl;
	f_out<<"Query: " <<id_query<<" Class: "<<class_query<<" Target "<<target_query<<std::endl;
	std::cout<<"saving "<<ids_vector.size()<<" images"<<std::endl;
	float mqr=0;
	for(size_t i=0; i<ids_vector.size(); i++){
		f_out<<i+1<<" ";
		f_out<<ids_vector[i]<<" ";
		f_out<<classes_vector[i]<<" ";
		f_out<<dists_vector[i]<<std::endl;
		if ((class_query.compare(classes_vector[i]))==0 && (target_query.compare(ids_vector[i]))==0){
			mqr=i+1;
		}
	}
	float PR[11];
	getNormalizedPrecision(PR);
	for(int i=0; i<10; i++){
		f_out<<PR[i]<<" ";
	}
	f_out<<PR[10]<<std::endl;;
	f_out<<"MQR: "<<mqr<<std::endl;
	f_out.close();
}
/*-------------------------------------------------------------------------------*/
template<class Index>
float Ranking<Index>::getMAP(){
	return mAP;
}
/*-------------------------------------------------------------------------------*/
template<class Index>
void Ranking<Index>::getNormalizedPrecision(float PR[11]){
	if((int)precision_vec.size()>0){
		JUtil::computeNormalizedPrecision(precision_vec,PR);
	}
}

//------------------------------------------------ When you use templates
// you should specify the all the possibles instantiations
template class SimilaritySearch<flann::L2<float> > ;
template class SimilaritySearch<flann::L1<float> > ;
template class SimilaritySearch<flann::HellingerDistance<float> > ;
template class SimilaritySearch<flann::HistIntersectionDistance<float> > ;
template class SimilaritySearch<flann::ChiSquareDistance<float> > ;
template class SimilaritySearch<flann::KL_Divergence<float> > ;

//------------------------------------------------
template class JLinearSearch<JL1>;
template class JLinearSearch<JL2>;
template class JLinearSearch<JHellinger>;
template class JLinearSearch<JChiSquare>;


template class Ranking<SimilaritySearch<flann::L1<float> > >;
template class Ranking<SimilaritySearch<flann::L2<float> > >;
template class Ranking<SimilaritySearch<flann::HellingerDistance<float> > >;
template class Ranking<SimilaritySearch<flann::ChiSquareDistance<float> > >;

template class Ranking<JLinearSearch<JChiSquare> >;
template class Ranking<JLinearSearch<JHellinger> >;
template class Ranking<JLinearSearch<JL1> >;
template class Ranking<JLinearSearch<JL2> >;

