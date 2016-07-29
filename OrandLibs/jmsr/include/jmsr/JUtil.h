/*
 * JUtil.h
 *
 *  Created on: Feb 2, 2015
 *  Author: José M. Saavedra
 *  Copyright Orand S.A.
 */


#ifndef JMSR_JUTIL_H_
#define JMSR_JUTIL_H_

#include "opencv2/highgui/highgui.hpp"
#include "similaritysearch.h"
#include <vector>
#include <map>
#include <string>


#define miINT8 1
#define miUINT8 2
#define miINT16 3
#define miUINT16 4
#define miINT32 5
#define miUINT32 6
#define miSINGLE 7
#define miDOUBLE 9
#define miMATRIX 14

#define FILE_FORMAT_BINARY 100
#define FILE_FORMAT_TEXT 200

#define INPUT_FILE_FORMAT_1 100
#define INPUT_FILE_FORMAT_2 200
#define INPUT_FILE_FORMAT_3 300


/*-------------------------------------LineCommandInput-------------------------------------------------------*/
class CommandLine{
private:
	std::map<std::string, std::string> map_params;
	void load(int args, char* vargs[]);
public:
	CommandLine(int args, char* vargs[]);
	std::string getValue(const std::string& param);
	~CommandLine();
};
/*-------------------------------------CInputData------------------------------------------------------------*/
class  CInputData{
private:
	int format_input;
	int num_objects;
	std::string str_path;
	std::vector<std::string> ids_vector; //id for each image
	std::vector<std::string> names_vector; //filename of each image, whe format=F3 it is the complete path
	std::vector<std::string> classes_vector; // a string containing the categories, many categories are separated by commas
public:
	CInputData(int _format_input=INPUT_FILE_FORMAT_1);
	void resize(int _n);
	void setPath(std::string _str_path);
	void setData(std::string _id, std::string _name, std::string _class, int pos);
	void addData(std::string _id, std::string _name, std::string _class);
	std::string getId(int i) const;
	std::string getName(int i) const;
	std::string getAbsoluteName(int i) const;
	std::string getClass(int i) const;
	std::string getPath() const;
	int getNumObjects() const;
	int getFormat()const;
	void addSuffixToDBPath(std::string suffix);

};
/*-------------------------------------QueryData------------------------------------------------------------*/
class  QueryData{
private:
	int num_objects;
	std::string str_path;
	std::vector<std::string> ids_vector;
	std::vector<std::string> names_vector;
	std::vector<std::string> classes_vector;
	std::vector<std::string> targets_vector;
	std::map<std::string, int> class_2_id;
	std::vector<std::string> class_names;
public:
	QueryData();
	void resize(int _n);
	void setPath(std::string _str_path);
	void setData(std::string _name, std::string _class, std::string _target, int pos);
	std::string getId(int i);
	std::string getName(int i);
	std::string getTarget(int i);
	std::string getAbsoluteName(int i);
	std::string getClass(int i);
	int getClassId(int i);
	std::string getClassName(int classID);
	std::string getPath();
	int getNumOfClasses();
	int getNumObjects();
};
/*-------------------------------------GT_Query------------------------------------------------------------*/
class GT_Query{
public:
	static const int Q_NEAR;
	static const int Q_EXACT;
private:
	std::string query_name;
	std::map<std::string, int> q_relevants;
	std::map<std::string, bool> q_classes;
	int n_nears;
	int n_exacts;
	int n_classes;

public:
	GT_Query(std::string _query_name, std::string gt_query_file);
	int getNumberOfNear() const;
	int getNumberOfExact() const;
	int getNumberOfClasses() const;
	bool isNearTo(std::string str_code) const;
	bool isExactTo(std::string str_code) const;
	bool belongsTo(std::string str_class) const;
	std::string getName() const;
	~GT_Query();

	static void loadGT(std::string str_gt_dir, std::vector<GT_Query>& v_gt_queries);
};
/*-------------------------------------ObjClassDist------------------------------------------------------------*/
class ObjClassDist{
private:
	std::string str_obj;
	std::string str_class;
	float f_dist;
public:
	ObjClassDist();
	ObjClassDist(std::string _str_obj,  std::string _str_class, float _f_dist);
	void setValues(std::string _str_obj,  std::string _str_class, float _f_dist);
	std::string getObj() const;
	std::string getClass() const;
	float getDist()  const;
};
/*-------------------------------------GT_Ranking------------------------------------------------------------*/
class GT_Ranking{
private:
	float f_near_AP;
	float f_exact_AP;
	float f_class_AP;
	float f_rank;
public:
	GT_Ranking(const std::vector<ObjClassDist>& vec_ranking, const GT_Query& gt_query);
	GT_Ranking(std::string str_ranking_file, const GT_Query& gt_query);
	float getNearAP() const;
	float getExactAP() const;
	float getClassAP() const;
	float getRank() const;
	static void save(const std::vector<ObjClassDist>& vec_ranking, std::string str_file);

};
/*-------------------------------------ConfigFile------------------------------------------------------------*/
class ConfigFile{
private:
	std::map<std::string, std::string> map_par_value;
public:
	ConfigFile(std::string file, char separator='\t');
	bool isDefined(std::string param);
	std::string getValue(std::string param);
};
/*-------------------------------------JUtil------------------------------------------------------------*/
class JUtil{
private:
	static int getMatlabSizeBytes(int type);
public:
	static const cv::Scalar COLOR_RED;
	static const cv::Scalar COLOR_BLUE;
	static const cv::Scalar COLOR_GREEN;
	static const cv::Scalar COLOR_YELLOW;

	JUtil();
	/**
	 * @param db_filename: text file containing all descriptors (JMSR_format)
	 * @param db: buffer of descriptors
	 * @param num_descriptors
	 * @param size_des
	 * @param v_obj_name:
	 * @param v_obj_class:
	 * @return
	 */
	static void loadDataset(std::string db_filename, float *&db, int *num_descriptors, int *size_des,
			std::vector<std::string> &v_obj_name ,std::vector<std::string> &v_obj_class, int file_format=FILE_FORMAT_TEXT);

	static void loadDataset_text(std::string db_filename, float *&db, int *num_descriptors, int *size_des,
				std::vector<std::string> &v_obj_name ,std::vector<std::string> &v_obj_class);

	static void loadDataset_binary(std::string db_filename, float *&db, int *num_descriptors, int *size_des,
					std::vector<std::string> &v_obj_name ,std::vector<std::string> &v_obj_class);
	/**
	 *
	 * @param number
	 * @return
	 */
	static std::string intToString(int number);
	/**
	 *
	 * @param input
	 * @param delimiter
	 * @return
	 */
	static std::vector<std::string> splitString(const std::string& input,char delimiter);
	static float rad2deg(float rad);
	static float deg2rad(float deg);
	static std::string eraseDuplicatedWords(std::string str_input);
	static void matlab2Mat(std::string mat_file, cv::Mat& mat);
	static std::vector<std::vector<int> > getSubsets(int N);
	static void readCInputData(std::string input_file, CInputData& c_input_data);
	static void readQueryData(std::string q_file, QueryData& input_data);
	static bool isRGB(cv::Mat image);
	static bool isRGBA(cv::Mat image);
	static void saveDescriptors(CInputData& db_data, float *des, int size_des, std::string file, int file_format=FILE_FORMAT_BINARY);
	static float* readDescriptors(std::string str_file, int *_n_objects, int* _size_des,
			std::vector<std::string>& obj_ids, std::vector<std::string>& obj_classes,
			int file_format=FILE_FORMAT_BINARY);
	static float* readDescriptors(std::string str_file, int *_n_objects, int* _size_des,
				std::vector<std::string>& obj_ids, std::vector<std::string>& obj_classes,
				std::map<std::string, int>& class2idx, std::vector<std::string>& namesByClass,
				int file_format=FILE_FORMAT_BINARY);
	static void saveMat(std::string str_file, cv::Mat& mat);
	static void readMat(std::string str_file, cv::Mat& mat);
	static float* readMat(std::string str_file, int target_depth, int* _n_rows, int* _n_cols);
	static bool isValid(int x, int y, int min_x,int max_x, int min_y, int max_y);
	static float getKernelProb(float* des, float* center, float* std, int dim);
	static float getKernelProbDist(float dist, float mu, float sigma);
	static void computeNormalizedPrecision(std::vector<float>& prec_vec, float PR[11]);
	static void jmsr_assert(bool assertion, std::string mss);
	template <class T> static void setArray(T *vector, int n_size, T val);
	static std::string lineArgumentsToString(int nargs, char* vargs[]);
	static std::string getInputValue(const std::string& input, std::string param);
	static std::string getBasename(std::string input);
	static std::string getDirname(std::string input);
	static std::string deleteExtension(std::string input);
	static std::string str_replace(std::string &s, const std::string &toReplace, const std::string &replaceWith);

	static float getEntropy(float* probs, int dim, bool normalized=false);
	static bool file_is_readeable(std::string str_file);
	template <class T> static void buffer_resize(T *&array, int* _cur_size, bool shrink=false);
	// some 1d functions
	static float getGaussianWeight(float d, float sigma);
	template <class T> static void findMaxMin(T* v, unsigned int size,
													T* max_v, unsigned int* idx_max,
													T* min_v, unsigned int* idx_min,
													int start_pos=-1, int end_pos=-1);
	static void colage(const std::vector<std::string>& vec_im, cv::Mat& mat_out, cv::Size size,
						int n_rows, int n_cols,
						int h_space=0, int v_space=0, bool first_is_query=false);
	static bool fileExists(const std::string& str_file);
	virtual ~JUtil();
};
#endif /* JMSR_JUTIL_H_ */
