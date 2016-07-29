/*
 * JUtil.cpp
 *
 *  Created on: Feb 2, 2015
 *  Author: José M. Saavedra
 *  Copyright Orand S.A.
 */

#include "JUtil.h"
#include "opencv2/imgproc/imgproc.hpp"
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <cassert>
#include <iostream>
#include <cmath>
#include <stdexcept>
/*-------------------------------------------------------------------------------------------*/
const cv::Scalar JUtil::COLOR_RED(0,0,255);
const cv::Scalar JUtil::COLOR_BLUE(255,0,0);
const cv::Scalar JUtil::COLOR_GREEN(0,255,0);
const cv::Scalar JUtil::COLOR_YELLOW(0,255,255);

const int GT_Query::Q_NEAR=100;
const int GT_Query::Q_EXACT=200;
/*-------------------------------------------------------------------------------------------*/
CommandLine::CommandLine(int nargs, char* vargs[]):map_params(){
	std::string str_input;
	std::string sep("=");
	for(int i=1; i<nargs; i++){
		str_input=std::string(vargs[i]);
		size_t pos=str_input.find(sep);
		if(pos==std::string::npos){
			throw std::runtime_error( sep + " is missed, incorrect input!");
		}
		map_params[str_input.substr(0,pos)]=str_input.substr(pos+1);
	}
}

std::string CommandLine::getValue(const std::string& param){
	std::string str_val("");
	if (map_params.find(param)!=map_params.end()){
		str_val=map_params[param];
	}
	else{
		std::cout<<"Warning: param "<<param<<" not found"<<std::endl;
	}
	return str_val;
}

CommandLine::~CommandLine(){

}
/*-------------------------------------------------------------------------------------------*/
QueryData::QueryData():num_objects(0),str_path(""),
ids_vector(0), names_vector(0), classes_vector(0), targets_vector(0),  class_names(0){
}
void QueryData::resize(int _n){
	num_objects=_n;
	ids_vector.resize(num_objects);
	names_vector.resize(num_objects);
	classes_vector.resize(num_objects);
	targets_vector.resize(num_objects);
}
void QueryData::setPath(std::string _str_path){
	str_path=_str_path;
}
void QueryData::setData(std::string _name, std::string _class, std::string _target, int pos){
	assert(pos<num_objects && pos>=0);
	names_vector[pos]=_name;
	classes_vector[pos]=_class;
	targets_vector[pos]=_target;
	ids_vector[pos]=_name;
	int pos_dot=ids_vector[pos].find_last_of('.');
	ids_vector[pos]=ids_vector[pos].substr(0, pos_dot);
	if(class_2_id.find(_class)==class_2_id.end()){
		class_2_id[_class]=class_names.size();
		class_names.push_back(_class);
	}
}
std::string QueryData::getName(int i){
	return names_vector[i];
}
std::string QueryData::getTarget(int i){
	return targets_vector[i];
}
std::string QueryData::getId(int i){
	return ids_vector[i]; //convert to quietar extension
}
std::string QueryData::getAbsoluteName(int i){
	return str_path+"/"+names_vector[i];
}
std::string QueryData::getClass(int i){
	return classes_vector[i];
}
int QueryData::getClassId(int i){
	return class_2_id[classes_vector[i]];
}
std::string QueryData::getClassName(int classID){
	return class_names[classID];
}
std::string QueryData::getPath(){
	return str_path;
}
int QueryData::getNumObjects(){
	return num_objects;
}
int QueryData::getNumOfClasses(){
	return (int)class_names.size();
}

/*-------------------------------------------------------------------------------------------*/
CInputData::CInputData(int _format_input):format_input(_format_input),num_objects(0),str_path(""),
ids_vector(0), names_vector(0), classes_vector(0){
}
void CInputData::resize(int _n){
	num_objects=_n;
	ids_vector.resize(num_objects);
	names_vector.resize(num_objects);
	classes_vector.resize(num_objects);
}
void CInputData::setPath(std::string _str_path){
	str_path=_str_path;
}
void CInputData::setData(std::string _id, std::string _name, std::string _class, int pos){
	assert(pos<num_objects && pos>=0);
	ids_vector[pos]=_id;
	names_vector[pos]=_name;
	classes_vector[pos]=_class;
}
void CInputData::addData(std::string _id, std::string _name, std::string _class){
	ids_vector.push_back(_id);
	names_vector.push_back(_name);
	classes_vector.push_back(_class);
	num_objects++;
}
std::string CInputData::getId(int i) const{
	return ids_vector[i];
}
std::string CInputData::getName(int i) const{
	return names_vector[i];
}
std::string CInputData::getAbsoluteName(int i) const{
	if (format_input==INPUT_FILE_FORMAT_1){
		return str_path+"/"+classes_vector[i]+"/"+names_vector[i];
	}
	else if(format_input==INPUT_FILE_FORMAT_2){
		return str_path+"/"+names_vector[i];
	}
	else if(format_input==INPUT_FILE_FORMAT_3){
		return names_vector[i];
	}
	else{
		std::cerr<<"Error: format incorrect"<<std::endl;
		exit(EXIT_FAILURE);
	}
}
std::string CInputData::getClass(int i) const{
	return classes_vector[i];
}
std::string CInputData::getPath() const{
	return str_path;
}
int CInputData::getNumObjects() const{
	return num_objects;
}
int CInputData::getFormat() const{
	return format_input;
}
void CInputData::addSuffixToDBPath(std::string suffix){
	std::string str_contour_path=str_path;
	while(str_contour_path[str_contour_path.length()-1]=='/'){
		str_contour_path.erase(str_contour_path.length()-1);
	}
	str_contour_path=str_contour_path+"_"+suffix;
	str_path=str_contour_path;
}
/*-------------------------------------------------------------------------------------------*/
ObjClassDist::ObjClassDist():str_obj(""), str_class(""), f_dist(0){
}

ObjClassDist::ObjClassDist(std::string _str_obj,
		std::string _str_class,
		float _f_dist):str_obj(_str_obj), str_class(_str_class), f_dist(_f_dist){
}

void ObjClassDist::setValues(std::string _str_obj,
		std::string _str_class, float _f_dist){
	str_obj=_str_obj;
	str_class=_str_class;
	f_dist=_f_dist;
}
std::string ObjClassDist::getObj() const{
	return str_obj;
}
std::string ObjClassDist::getClass() const{
	return str_class;
}
float ObjClassDist::getDist() const{
	return f_dist;
}
/*-------------------------------------------------------------------------------------------*/
GT_Ranking::GT_Ranking(const std::vector<ObjClassDist>& vec_ranking, const GT_Query& gt_query):
	f_near_AP(0), f_exact_AP(0), f_class_AP(0){
	int i=0;
	int n_exacts=0;
	int n_nears=0;
	int n_classes=0;
	std::map<std::string, bool> map_objs;
	f_rank=-1;
	for(auto rk:vec_ranking){
		if(map_objs.find(rk.getObj())==map_objs.end()){
			if(gt_query.isExactTo(rk.getObj())){
				if(f_rank==-1) f_rank=(i+1);
				f_exact_AP+=1.0/(i+1);
				n_exacts++;
			}
			if(gt_query.isNearTo(rk.getObj())){
				if(f_rank==-1) f_rank=(i+1);
				f_near_AP+=1.0/(i+1);
				n_nears++;
			}
			if(gt_query.belongsTo(rk.getClass())){
				f_class_AP+=1.0/(i+1);
				n_classes++;
			}
			i++;
			map_objs[rk.getObj()]=true;
		}
	}
	if(n_exacts!=gt_query.getNumberOfExact() ||
			n_nears!=gt_query.getNumberOfNear()+gt_query.getNumberOfExact()){
		std::cerr<<"Something wrong happens, n_exact!=n_exact-query or  n_near!=n_near-query"<<std::endl;
		std::cerr<<"n_exact: "<<n_exacts<<" != "<<gt_query.getNumberOfNear()<<std::endl;
		std::cerr<<"n_near: "<<n_nears<<" != "<<gt_query.getNumberOfNear()+gt_query.getNumberOfExact()<<std::endl;
		exit(EXIT_FAILURE);
	}
	if(n_exacts!=0)	f_exact_AP=f_exact_AP/n_exacts;
	if(n_nears!=0) f_near_AP=f_near_AP/n_nears;
	if(n_classes!=0)f_class_AP=f_class_AP/n_classes;
}
/*-------------------------------------------------------------------------------------------*/
GT_Ranking::GT_Ranking(std::string str_ranking_file, const GT_Query& gt_query):
	f_near_AP(0), f_exact_AP(0), f_class_AP(0){
	int i=0;
	int n_exacts=0;
	int n_nears=0;
	int n_classes=0;
	std::ifstream f_in(str_ranking_file);
	JUtil::jmsr_assert(f_in.is_open(),str_ranking_file+"can't be opened");
	std::string str_line;
	std::string str_obj;
	std::string str_class;
	std::stringstream s_str;
	f_rank=-1;
	float f_dist;
	i=0;
	while(std::getline(f_in, str_line)){
		s_str.clear();
		s_str.str(str_line);
		s_str>>str_obj>>str_class>>f_dist;
		if(gt_query.isExactTo(str_obj)){
			if(f_rank==-1) f_rank=(i+1);
			f_exact_AP+=1.0/(i+1);
			n_exacts++;
		}
		if(gt_query.isNearTo(str_obj)){
			if(f_rank==-1) f_rank=(i+1);
			f_near_AP+=1.0/(i+1);
			n_nears++;
		}
		if(gt_query.belongsTo(str_class)){
			f_class_AP+=1.0/(i+1);
			n_classes++;
		}
		i++;
	}
	f_in.close();
	if(n_exacts!=gt_query.getNumberOfExact() ||
			n_nears!=gt_query.getNumberOfNear()+gt_query.getNumberOfExact()){
		std::cerr<<"Something wrong happens, n_exact!=n_exact-query or  n_near!=n_near-query"<<std::endl;
		std::cerr<<"n_exact: "<<n_exacts<<" != "<<gt_query.getNumberOfNear()<<std::endl;
		std::cerr<<"n_near: "<<n_nears<<" != "<<gt_query.getNumberOfNear()+gt_query.getNumberOfExact()<<std::endl;
		exit(EXIT_FAILURE);
	}
	if(n_exacts!=0)	f_exact_AP=f_exact_AP/n_exacts;
	if(n_nears!=0) f_near_AP=f_near_AP/n_nears;
	if(n_classes!=0)f_class_AP=f_class_AP/n_classes;
}

float GT_Ranking::getExactAP() const{
	return f_exact_AP;
}

float GT_Ranking::getNearAP() const{
	return f_near_AP;
}

float GT_Ranking::getClassAP() const{
	return f_class_AP;
}

float GT_Ranking::getRank() const{
	return f_rank;
}

void GT_Ranking::save(const std::vector<ObjClassDist>& vec_ranking, std::string str_file){
	std::ofstream f_out(str_file);
	JUtil::jmsr_assert(f_out.is_open(),str_file+"can't be opened");
	for(auto rk:vec_ranking){
		f_out<<rk.getObj()<<"\t"<<rk.getClass()<<"\t"<<rk.getDist()<<std::endl;
	}
	f_out.close();
}
/*-------------------------------------------------------------------------------------------*/
GT_Query::GT_Query(std::string _query_name, std::string gt_query_file):
		query_name(_query_name), q_relevants(), q_classes(),
		n_nears(0), n_exacts(0), n_classes(0){

	std::ifstream f_in(gt_query_file);
	JUtil::jmsr_assert(f_in.is_open(),gt_query_file+" can't be opened!!");
	char chr_type;
	std::string str_value;
	std::string str_line;
	std::stringstream s_str;
	while(std::getline(f_in, str_line)) {
		s_str.clear();
		s_str.str(str_line);
		s_str>>chr_type>>str_value;
		if(chr_type=='C'){
			if(q_classes.find(str_value)==q_classes.end()){
				q_classes[str_value]=true;
				n_classes++;
			}
			else{
				std::cout<<"WARNING:"<< str_value<<" is duplicated"<<std::endl;
			}
		}
		else if(chr_type=='N'){
			if(q_relevants.find(str_value)==q_relevants.end()){
				q_relevants[str_value]=Q_NEAR;
				n_nears++;
			}
			else{
				std::cout<<"WARNING:"<< str_value<<" is duplicated"<<std::endl;
			}
		}
		else if(chr_type=='E'){
			if(q_relevants.find(str_value)==q_relevants.end()){
				q_relevants[str_value]=Q_EXACT;
				n_exacts++;
			}
			else{
				std::cout<<"WARNING:"<< str_value<<" is duplicated"<<std::endl;
			}
		}
		else{
			std::cout<<"type in ground-truth is not correct: "<<chr_type<<std::endl;
			exit(EXIT_FAILURE);
		}
	}
	f_in.close();
}
int GT_Query::getNumberOfExact() const{
	return n_exacts;
}
int GT_Query::getNumberOfNear() const{
	return n_nears;
}
int GT_Query::getNumberOfClasses() const{
	return n_classes;
}
bool GT_Query::isNearTo(std::string str_code) const{
	return (q_relevants.find(str_code)!=q_relevants.end());
}
bool GT_Query::isExactTo(std::string str_code) const{
	bool ans=false;
	if (q_relevants.find(str_code)!=q_relevants.end()){
		if(q_relevants.at(str_code)==Q_EXACT){
			ans=true;
		}
	}
	return ans;
}
std::string GT_Query::getName() const{
	return query_name;
}
//str_class may be a multi_class separated by commas
bool GT_Query::belongsTo(std::string str_class)const {
	std::vector<std::string> vec_rk_classes= JUtil::splitString(str_class, ',');
	bool ans=false;
	for(unsigned int i=0; (i<vec_rk_classes.size() && !ans); i++){
		ans=(q_classes.find(vec_rk_classes[i])!=q_classes.end());
	}
	return ans;
}
GT_Query::~GT_Query(){
}

void GT_Query::loadGT(std::string str_gt_dir, std::vector<GT_Query>& v_gt_queries){
	std::string str_query_list=str_gt_dir+"/list_query_sketches.txt";
	std::string str_gt_query_dir=str_gt_dir+"/query_sketches_gt";
	std::ifstream f_in(str_query_list);
	JUtil::jmsr_assert(f_in.is_open(), str_query_list+" can't be opened!!");
	std::string str_query;
	v_gt_queries.clear();
	while(std::getline(f_in, str_query)){
		std::cout<<str_query<<std::endl;
		v_gt_queries.push_back(GT_Query(str_query, str_gt_query_dir+"/"+str_query+".txt"));
	}
	std::cout<<"-> "<<v_gt_queries.size()<<" gt_queries"<<std::endl;
	f_in.close();
}
/*-------------------------------------------------------------------------------------------*/
ConfigFile::ConfigFile(std::string file, char separator){
	std::ifstream f_in(file);
	JUtil::jmsr_assert(f_in.is_open(), file+std::string(" could not be opened!!"));
	std::string str_line;
	std::vector<std::string> vec_string;
	while(std::getline(f_in, str_line)){
		if (str_line[0]!='#'){
			vec_string=JUtil::splitString(str_line, separator);
			if (vec_string.size()>=2){
				map_par_value[vec_string[0]]=vec_string[1];
			}
		}
	}
}

bool ConfigFile::isDefined(std::string param){
	return (map_par_value.find(param)!=map_par_value.end());
}

std::string ConfigFile::getValue(std::string param){
	std::string val="";
	if (isDefined(param)){
		val=map_par_value[param];
	}
	return val;
}
/*-------------------------------------------------------------------------------------------*/

void JUtil::loadDataset(std::string db_filename, float *&db, int *num_descriptors, int *size_des,
		std::vector<std::string> &v_obj_name ,std::vector<std::string> &v_obj_class, int file_format){
	std::cout<<"LOADING"<<std::endl;
	if(file_format==FILE_FORMAT_BINARY){
		loadDataset_binary(db_filename, db, num_descriptors, size_des, v_obj_name, v_obj_class);
	}
	else if(file_format==FILE_FORMAT_TEXT){
		loadDataset_text(db_filename, db, num_descriptors, size_des, v_obj_name, v_obj_class);
	}
	else{
		std::cerr<<"WARNING:  no dataset loaded "<<std::endl;
		exit(EXIT_FAILURE);
	}
}
/*-------------------------------------------------------------------------------------------*/
void JUtil::loadDataset_text(std::string db_filename, float *&db, int *num_descriptors, int *size_des,
		std::vector<std::string> &v_obj_name ,std::vector<std::string> &v_obj_class){
    std::ifstream file;
    file.open(db_filename.c_str());
    //assert(file.is_open)
    jmsr_assert(file.is_open(),std::string("file ") + db_filename + "cann't be opened");
    std::string obj_name;
    std::string obj_class;

    file>>*num_descriptors>>*size_des;
    int total_size_db=(*num_descriptors)*(*size_des);
    db=new float[total_size_db];

    v_obj_name.resize(*num_descriptors);
    v_obj_class.resize(*num_descriptors);

    int i_des=0, i_val=0;
    float val=0;

    for(i_des=0; i_des<*num_descriptors;i_des++){
        file>>obj_name>>obj_class;
        v_obj_name[i_des]=obj_name;
        v_obj_class[i_des]=obj_class;

        for(i_val=0; i_val<*size_des; i_val++){
            file>>val;
            db[i_des*(*size_des)+i_val]=val;
        }
    }
}
/*-------------------------------------------------------------------------------------------*/
void JUtil::loadDataset_binary(std::string db_filename, float *&db, int *num_descriptors, int *size_des,
		std::vector<std::string> &v_obj_name ,std::vector<std::string> &v_obj_class){
	std::ifstream file;
	std::cout<<"Loading binary dataset"<<std::endl;
	file.open(db_filename.c_str(), std::ios::in | std::ios::binary);
	jmsr_assert(file.is_open(),std::string("file ") + db_filename + " can't be opened");
	int _num_descriptors=0;
	int _size_des=0;
	file.read(reinterpret_cast<char*>(&_num_descriptors), sizeof(int));
	file.read(reinterpret_cast<char*>(&_size_des), sizeof(int));
	db=new float[_num_descriptors*_size_des];
	float *db_des=NULL;
	v_obj_name.clear();
	v_obj_name.resize(_num_descriptors);

	v_obj_class.clear();
	v_obj_class.resize(_num_descriptors);

	int i=0;
	int image_code_length=0;
	int image_class_length=0;
	char* c_image_code;
	char* c_image_class;
	for(i=0; i<_num_descriptors;i++){
		file.read(reinterpret_cast<char*>(&image_code_length), sizeof(int));
		file.read(reinterpret_cast<char*>(&image_class_length), sizeof(int));

		c_image_code=new char[image_code_length];
		c_image_class=new char[image_class_length];

		file.read(c_image_code, image_code_length);
		file.read(c_image_class, image_class_length);

		v_obj_name[i]=std::string(c_image_code, image_code_length);
		v_obj_class[i]=std::string(c_image_class, image_class_length);

	    db_des=db+i*_size_des;
	    file.read(reinterpret_cast<char*>(db_des), _size_des*sizeof(float));
	    delete[] c_image_code;
	    delete[] c_image_class;
	}
	file.close();
	*num_descriptors=_num_descriptors;
	*size_des=_size_des;
}
/*-------------------------------------------------------------------------------------------*/
std::vector<std::string> JUtil::splitString(const std::string& input,char delimiter){
    std::vector<std::string> result;
    std::string token;
    size_t pos1=input.find_first_not_of(delimiter);
    size_t pos2;
    while(pos1!=std::string::npos)
    {
    	pos2=input.find_first_of(delimiter, pos1+1);
    	if(pos2==std::string::npos) pos2=input.length();
    	token=input.substr(pos1,pos2-pos1);
    	result.push_back(token);
    	pos1=input.find_first_not_of(delimiter, pos2+1);;
    }
    return result;
}
/*-------------------------------------------------------------------------------------------*/
float JUtil::rad2deg(float rad){
	return 180*(rad/3.14159);
}
/*-------------------------------------------------------------------------------------------*/
float JUtil::deg2rad(float deg){
	return 3.14159*(deg/180.0);
}
/*-------------------------------------------------------------------------------------------*/
std::string JUtil::intToString(int number)
{
    std::stringstream ss;//create a stringstream
    ss << number;//add number to the stream
    return ss.str();//return a string with the contents of the stream
}
//str_input words separated by whitespaces
std::string JUtil::eraseDuplicatedWords(std::string str_input){
	std::string str_out("");
	if(str_input.compare("")!=0){
		std::vector<std::string> vec_words=JUtil::splitString(str_input,' ');
		std::map<std::string, bool> word2bool;
		for(size_t i_class=0; i_class<vec_words.size();i_class++){
			if(word2bool.find(vec_words[i_class])==word2bool.end()){
				str_out=str_out+" "+vec_words[i_class];
				word2bool[vec_words[i_class]]=true;
			}
		}
	}
	return str_out;
}
/*-------------------------------------------------------------------------------------------*/
int JUtil::getMatlabSizeBytes(int type){
	unsigned char n_bytes=0;
	if(type==miINT8 || type==miUINT8) n_bytes=1;
	else if(type==miINT16 || type==miUINT16) n_bytes=2;
	else if(type==miINT32 || type==miUINT32) n_bytes=4;
	else if(type==miSINGLE || type==miDOUBLE) n_bytes=8;
	return n_bytes;
}

void JUtil::matlab2Mat(std::string mat_file, cv::Mat& mat){
	std::ifstream f_in(mat_file.c_str(), std::ios::in | std::ios::binary);
	assert(f_in.is_open());
	char *header=new char[128];
	char *first_part=new char[24];
	char *variable_name;
	char *flag_1=new char[4];
	char *flag_2=new char[4];
	char *flag_4=new char[4];

	for(int i=0; i<4; i++){
			flag_1[i]=0;
			flag_2[i]=0;
			flag_4[i]=0;
		}

	int data_type=0, data_size=0;
	int matrix_type=0, matrix_size=0;
	int n_rows=0, n_cols=0;
	f_in.read(header, 128);
	f_in.read(reinterpret_cast<char*>(&data_type), 4);
	f_in.read(reinterpret_cast<char*>(&data_size), 4);

	if(data_type!=miMATRIX){
		std::cerr<<"Data type "<<data_type<<" is not a correct type!!"<<std::endl;
		exit(EXIT_FAILURE);
	}

	f_in.read(first_part, 24);
	f_in.read(reinterpret_cast<char*>(&n_rows), 4);
	std::cout<<"n_rows: "<<n_rows<<std::endl;
	f_in.read(reinterpret_cast<char*>(&n_cols), 4);
	std::cout<<"n_cols: "<<n_cols<<std::endl;
	//---------------------------------------------------Reading matrix name
	f_in.read(flag_1, 2);
	std::cout<<"flag_1:"<<*(reinterpret_cast<int*>(flag_1))<<std::endl;
	f_in.read(flag_2, 2);
	int name_length=0;
	name_length=*(reinterpret_cast<int*>(flag_2));
	std::cout<<"name_length:"<<name_length<<std::endl;
	if(name_length==0)
	{
		f_in.read(flag_4, 4);
		name_length=*(reinterpret_cast<int*>(flag_4));
		std::cout<<name_length<<std::endl;
		name_length=ceil(name_length/(float)8)*8; //read 8-byte blocks
	}
	else{
		name_length=4;
	}
	variable_name=new char[name_length];
	f_in.read(variable_name, name_length);
	std::cout<<"name: "<<variable_name<<std::endl;
	//----------------------------------------------------
	f_in.read(reinterpret_cast<char*>(&matrix_type), 4);
	std::cout<<"type: "<<matrix_type<<std::endl;
	f_in.read(reinterpret_cast<char*>(&matrix_size), 4);

	std::cout<<"size_data: "<<matrix_size<<std::endl; //matrix_size in bytes
	//unsigned char n_bytes_per_value=getMatlabSizeBytes(matrix_type);
	char *data=new char[matrix_size];
	f_in.read(data,matrix_size);

	if(matrix_type==miDOUBLE){

		//double *formated_data=reinterpret_cast<double*>(data);
		mat.create(n_cols, n_rows, CV_64FC1);
		std::copy(data, data+matrix_size,mat.data);
		delete [] data;
		mat=mat.t();
	}
	if(matrix_type==miSINGLE){
		//float *formated_data=reinterpret_cast<float*>(data);
		mat.create(n_cols, n_rows, CV_32FC1);
		std::copy(data, data+matrix_size,mat.data);
		delete [] data;
		mat=mat.t();
	}
	if(matrix_type==miUINT8){

		//unsigned char *formated_data=reinterpret_cast<unsigned char*>(data);
		mat.create(n_cols, n_rows, CV_8UC1);
		std::copy(data, data+matrix_size,mat.data);
		delete [] data;
		mat=mat.t();
	}
	if(matrix_type==miINT8){

		//char *formated_data=reinterpret_cast<char*>(data);
		mat.create(n_cols, n_rows, CV_8SC1);
		std::copy(data, data+matrix_size,mat.data);
		delete [] data;
		mat=mat.t();
	}
	if(matrix_type==miUINT16){

		//unsigned short *formated_data=reinterpret_cast<unsigned short*>(data);
		mat.create(n_cols, n_rows, CV_16UC1);
		std::copy(data, data+matrix_size,mat.data);
		delete [] data;
		mat=mat.t();
	}
	if(matrix_type==miINT16){
		//short *formated_data=reinterpret_cast<short*>(data);
		mat.create(n_cols, n_rows, CV_16SC1);
		std::copy(data, data+matrix_size,mat.data);
		delete [] data;
		mat=mat.t();
	}
	if(matrix_type==miINT32){
		//int *formated_data=reinterpret_cast<int*>(data);
		mat.create(n_cols, n_rows, CV_32SC1);
		std::copy(data, data+matrix_size,mat.data);
		delete [] data;
		mat=mat.t();
	}
	f_in.close();
	delete[] header;
	delete[] first_part;
	delete[] variable_name;
	delete[] flag_1;
	delete[] flag_2;
	delete[] flag_4;
}

/*-------------------------------------------------------------------------------------------*/
std::vector<std::vector<int> > JUtil::getSubsets(int N)
{
    //N<=6, this function support subsets for N<=6, in other cases we will increase the number of bits
    //This function is interesing since a subset is represented by the position of bits 1 in the binary
    //representation of each number from 1 to 2^N
    std::vector<std::vector<int> > subsets;
    int bit[]={1,2,4,8,16,32};
    int n_subsets=pow(2,N);
    for(int i=0;i<n_subsets;i++)
    {
        std::vector<int> subset;
        for(int bit_p=0;bit_p<N;bit_p++)
        {
            if(bit[bit_p] & i) subset.push_back(bit_p+1);
        }
        subsets.push_back(subset);
    }
    return subsets;
}
/*-------------------------------------------------------------------------------------------*/
void JUtil::readCInputData(std::string input_file, CInputData& c_input_data){
	std::ifstream f_in(input_file.c_str());
	JUtil::jmsr_assert(f_in.is_open(), input_file + " can't be opened!");
	std::string id_image;
	std::string name_image;
	std::string class_image;
	std::string str_line;
	std::vector<std::string> vec_str;
	std::cout<<c_input_data.getFormat()<<std::endl;
	if (c_input_data.getFormat()==INPUT_FILE_FORMAT_3){
		while(std::getline(f_in, str_line)){
			vec_str=JUtil::splitString(str_line,'\t');
			JUtil::jmsr_assert(vec_str.size()==3, input_file + " in incorrect format, it requires 3-field lines >>" +str_line);
			id_image=vec_str[0];
			name_image=vec_str[1];
			class_image=vec_str[2];
			c_input_data.addData(id_image, name_image, class_image);
		}
	}
	else{
		int n_objs;
		std::string str_path;
		std::string str_n_objs;
		std::getline(f_in, str_n_objs);
		JUtil::jmsr_assert(!str_n_objs.empty(), input_file + " in incorrect format, <n_objs> is missed");
		n_objs=atoi(str_n_objs.c_str());
		std::getline(f_in, str_path);
		c_input_data.resize(n_objs);
		c_input_data.setPath(str_path);
		for(int i=0; i<n_objs;i++){
			std::getline(f_in, str_line);
			vec_str=JUtil::splitString(str_line,'\t');
			JUtil::jmsr_assert(vec_str.size()==3, input_file + " in incorrect format, it requires 3-field lines");
			id_image=vec_str[0];
			name_image=vec_str[1];
			class_image=vec_str[2];
			c_input_data.setData(id_image, name_image, class_image, i);
		}
	}
	f_in.close();
}
/*-------------------------------------------------------------------------------------------*/
void JUtil::readQueryData(std::string q_file, QueryData& input_data){
	std::ifstream f_in(q_file.c_str());
	assert(f_in.is_open());
	int n_objs;
	std::string str_path;
	f_in>>n_objs>>str_path;
	input_data.resize(n_objs);
	input_data.setPath(str_path);
	std::string  name_image;
	std::string  class_image;
	std::string  target_image;
	for(int i=0; i<n_objs;i++){
		f_in>>name_image>>class_image>>target_image;
		input_data.setData(name_image, class_image, target_image, i);
	}
	f_in.close();
}
/*-------------------------------------------------------------------------------------------*/
bool JUtil::isRGB(cv::Mat image){
	return (image.channels()==3);
}
/*-------------------------------------------------------------------------------------------*/
bool JUtil::isRGBA(cv::Mat image){
	return (image.channels()==4);
}
/*-------------------------------------------------------------------------------------------*/
void JUtil::saveDescriptors(CInputData& db_data, float *des, int size_des, std::string str_file, int file_format){
	int n_objects=db_data.getNumObjects();
	if(file_format==FILE_FORMAT_BINARY){
		std::ofstream f_out(str_file, std::ios::out | std::ios::binary);
		assert(f_out.is_open());
		f_out.write(reinterpret_cast<char*>(&n_objects), sizeof(int));
		f_out.write(reinterpret_cast<char*>(&size_des), sizeof(int));
		f_out.write(reinterpret_cast<char*>(des), n_objects*size_des*sizeof(float));
		std::string id;
		std::string clase;
		int id_size;
		int clase_size;
		for(int i=0; i<n_objects; i++){
			id=db_data.getId(i);
			clase=db_data.getClass(i);
			id_size=id.size();
			clase_size=clase.size();
			f_out.write(reinterpret_cast<char*>(&id_size), sizeof(int));
			f_out.write(id.c_str(), id_size);
			f_out.write(reinterpret_cast<char*>(&clase_size), sizeof(int));
			f_out.write(clase.c_str(), clase_size);
		}
		f_out.close();
	}
	else // FILE_FORMAT_TEXT
	{
		std::ofstream f_out(str_file);
		assert(f_out.is_open());
		f_out<<n_objects<<" "<<size_des<<std::endl;
		std::string id;
		std::string clase;
		for(int i=0; i<n_objects; i++){
			id=db_data.getId(i);
			clase=db_data.getClass(i);
			f_out<<id<<" "<<clase<<" ";
			for(int j=0; j<size_des-1;j++){
				f_out<<des[i*size_des+j]<<" ";
			}
			f_out<<des[i*size_des+(size_des-1)]<<std::endl;
		}
		f_out.close();
	}
}
/*-------------------------------------------------------------------------------------------*/
float* JUtil::readDescriptors(std::string str_file, int *_n_objects, int* _size_des,
			std::vector<std::string>& obj_ids, std::vector<std::string>& obj_classes, int file_format){

	float* des;
	int n_objects=0;
	int size_des=0;
	if (file_format==FILE_FORMAT_BINARY){
		std::ifstream f_in(str_file, std::ios::in | std::ios::binary);
		assert(f_in.is_open());
		f_in.read(reinterpret_cast<char*>(&n_objects), sizeof(int));
		f_in.read(reinterpret_cast<char*>(&size_des), sizeof(int));
		des=new float[n_objects*size_des];
		f_in.read(reinterpret_cast<char*>(des), n_objects*size_des*sizeof(float));
		obj_ids.resize(n_objects);
		obj_classes.resize(n_objects);
		int id_size;
		int clase_size;
		char* buffer;
		std::string id;
		std::string clase;
		for(int i=0; i<n_objects;i++){
			f_in.read(reinterpret_cast<char*>(&id_size), sizeof(int));
			buffer=new char[id_size];
			f_in.read(buffer, id_size);
			obj_ids[i].assign(buffer, id_size);
			delete[] buffer;
			f_in.read(reinterpret_cast<char*>(&clase_size), sizeof(int));
			buffer=new char[clase_size];
			f_in.read(buffer, clase_size);
			obj_classes[i].assign(buffer, clase_size);
			delete[] buffer;
		}
		f_in.close();
	}
	else{
		std::ifstream f_in(str_file);
		assert(f_in.is_open());
		f_in>>n_objects>>size_des;
		obj_ids.resize(n_objects);
		obj_classes.resize(n_objects);
		des=new float[n_objects*size_des];
		std::string id;
		std::string clase;
		for(int i=0; i<n_objects;i++){
			f_in>>id>>clase;
			obj_ids[i]=id;
			obj_classes[i]=clase;
			for(int j=0; j<size_des;j++){
				f_in>>des[i*size_des+j];
			}
		}
		f_in.close();
	}
	*_n_objects=n_objects;
	*_size_des=size_des;
	return des;
}
/*-------------------------------------------------------------------------------------------*/
float* JUtil::readDescriptors(std::string str_file, int *_n_objects, int* _size_des,
			std::vector<std::string>& obj_ids, std::vector<std::string>& obj_classes,
			std::map<std::string, int>& class2idx, std::vector<std::string>& namesByClass,
			int file_format){

	float* des;
	int n_objects=0;
	int size_des=0;
	int idx_class=0;
	if (file_format==FILE_FORMAT_BINARY){
		std::ifstream f_in(str_file, std::ios::in | std::ios::binary);
		assert(f_in.is_open());
		f_in.read(reinterpret_cast<char*>(&n_objects), sizeof(int));
		f_in.read(reinterpret_cast<char*>(&size_des), sizeof(int));
		des=new float[n_objects*size_des];
		f_in.read(reinterpret_cast<char*>(des), n_objects*size_des*sizeof(float));
		obj_ids.resize(n_objects);
		obj_classes.resize(n_objects);
		class2idx.clear();
		namesByClass.clear();

		int id_size;
		int clase_size;
		char* buffer;
		std::string id;
		std::string clase;
		for(int i=0; i<n_objects;i++){
			f_in.read(reinterpret_cast<char*>(&id_size), sizeof(int));
			buffer=new char[id_size];
			f_in.read(buffer, id_size);
			obj_ids[i].assign(buffer, id_size);
			delete[] buffer;
			f_in.read(reinterpret_cast<char*>(&clase_size), sizeof(int));
			buffer=new char[clase_size];
			f_in.read(buffer, clase_size);
			obj_classes[i].assign(buffer, clase_size);
			delete[] buffer;
			//-----------------------------------updating class2idx
			if(class2idx.find(obj_classes[i])==class2idx.end()){
				idx_class=(int)namesByClass.size();
				class2idx[obj_classes[i]]=idx_class;
				namesByClass.push_back(obj_classes[i]);
			}
			else{
				idx_class=class2idx[obj_classes[i]];
				namesByClass[idx_class]=namesByClass[idx_class]+"\t"+obj_ids[i];
			}
			//-----------------------------------
		}
		f_in.close();
	}
	else{
		std::ifstream f_in(str_file);
		assert(f_in.is_open());
		f_in>>n_objects>>size_des;
		obj_ids.resize(n_objects);
		obj_classes.resize(n_objects);
		des=new float[n_objects*size_des];
		std::string id;
		std::string clase;
		for(int i=0; i<n_objects;i++){
			f_in>>id>>clase;
			obj_ids[i]=id;
			obj_classes[i]=clase;
			for(int j=0; j<size_des;j++){
				f_in>>des[i*size_des+j];
			}
			//-----------------------------------updating class2idx
			if(class2idx.find(obj_classes[i])==class2idx.end()){
				idx_class=(int)namesByClass.size();
				class2idx[obj_classes[i]]=idx_class;
				namesByClass.push_back(obj_classes[i]);
			}
			else{
				idx_class=class2idx[obj_classes[i]];
				namesByClass[idx_class]=namesByClass[idx_class]+"\t"+obj_ids[i];
			}
			//-----------------------------------
		}
		f_in.close();
	}
	*_n_objects=n_objects;
	*_size_des=size_des;
	return des;
}
/*-------------------------------------------------------------------------------------------*/
void JUtil::saveMat(std::string str_file, cv::Mat& mat){
	int n_rows=mat.rows;
	int n_cols=mat.cols;
	int n_bytes=0;
	int depth=mat.depth();
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
	std::ofstream f_out(str_file.c_str(), std::ios::out | std::ios::binary);
	assert(f_out.is_open());
	f_out.write(reinterpret_cast<char*>(&n_rows), sizeof(int));
	f_out.write(reinterpret_cast<char*>(&n_cols), sizeof(int));
	f_out.write(reinterpret_cast<char*>(&depth), sizeof(int));
	f_out.write(reinterpret_cast<char*>(mat.data), n_rows*n_cols*n_bytes);
	f_out.close();
}
/*-------------------------------------------------------------------------------------------*/
float* JUtil::readMat(std::string str_file, int target_depth, int* _n_rows, int* _n_cols){
	std::ifstream f_in(str_file.c_str(), std::ios::in | std::ios::binary);
	assert(f_in.is_open());
	int n_rows=0;
	int n_cols=0;
	int n_bytes=0;
	int depth=0;
	f_in.read(reinterpret_cast<char*>(&n_rows), sizeof(int));
	f_in.read(reinterpret_cast<char*>(&n_cols), sizeof(int));
	f_in.read(reinterpret_cast<char*>(&depth), sizeof(int));
	assert(target_depth==depth);
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
	//template assert(n_bytes==sizeof(T));
	float* data=new float[n_rows*n_cols];
	f_in.read(reinterpret_cast<char*>(data), n_rows*n_cols*n_bytes);
	*_n_rows=n_rows;
	*_n_cols=n_cols;
	f_in.close();
	return data;
}
/*-------------------------------------------------------------------------------------------*/
void JUtil::readMat(std::string str_file, cv::Mat& mat){
	std::ifstream f_in(str_file.c_str(), std::ios::in | std::ios::binary);
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
	mat.create(n_rows, n_cols, depth);
	f_in.read(reinterpret_cast<char*>(mat.data), n_rows*n_cols*n_bytes);
	f_in.close();

}
/*-------------------------------------------------------------------------------------------*/
bool JUtil::isValid(int x, int y, int min_x,int max_x, int min_y, int max_y)
{
    bool r=false;
    if(x>=min_x && x<=max_x && y>=min_y && y<=max_y) r=true;
    else r=false;
    return r;
}
/*-------------------------------------------------------------------------------------------*/
void JUtil::computeNormalizedPrecision(std::vector<float>& prec_vec, float PR[11]){
	setArray<float>(PR,11,0);
	int n_size=(int)prec_vec.size();
	int recall_norm=10;
	float recall_i=0;
	float max_prec=prec_vec[n_size-1];
	int i=n_size-1;
	while(i>=0){
		recall_i=(i+1)/(float)n_size;
		while(i>=0 && recall_i>=recall_norm*0.1){
			if(prec_vec[i]>max_prec){
				max_prec=prec_vec[i];
			}
			i--;
			recall_i=(i+1)/(float)n_size;
		}
		PR[recall_norm]=max_prec;
		recall_norm--;
	}
	while(recall_norm>=0){
		PR[recall_norm]=max_prec;
		recall_norm--;
	}
}
/*-------------------------------------------------------------------------------------------*/
float JUtil::getKernelProb(float* des, float* center, float* std, int dim){
	float det=1;
	float dif=0;
	float prob=0;
	float eps=1e-10;
	float d=0;
	for(int i=0; i<dim; i++){
		d=std[i]+0.1;
		det*=d;
		dif+=((des[i]-center[i])*(des[i]-center[i]))/(d*d);
	}
	std::cout<<"KERNEL"<<std::endl;

	float factor=pow(2*3.14159, dim*0.5)*std::sqrt(det);
	std::cout<<det<<" "<<dif<<" "<<factor<<std::endl;
	factor=1.0/(factor+eps);
	prob=exp(-0.5*dif);
	return prob;
}
/*-------------------------------------------------------------------------------------------*/
float JUtil::getKernelProbDist(float dist, float mu, float sigma){
	float factor=1.0/(sigma*sqrt(2*3.14159));
	float prob=factor*exp(-0.5*(dist-mu)*(dist-mu)/(sigma*sigma));
	return prob;
}
/*-------------------------------------------------------------------------------------------*/
void JUtil::jmsr_assert(bool assertion, std::string mss){
	if(!assertion){
		throw std::runtime_error(std::string("Runtime error: ") + mss);
	}
	//assert(assertion);
}
/*-------------------------------------------------------------------------------------------*/
template <class T>  void JUtil::setArray(T *vector, int n_size, T val){
	for(int i=0; i<n_size; i++){
		vector[i]=val;
	}
}
template void JUtil::setArray<float>(float* vector, int n_size, float val);
template void JUtil::setArray<int>(int* vector, int n_size, int val);
template void JUtil::setArray<unsigned char>(unsigned char* vector, int n_size, unsigned char val);
/*-------------------------------------------------------------------------------------------*/
std::string JUtil::lineArgumentsToString(int nargs, char* vargs[]){
	int i=0;
	std::string str("");
	for(i=0; i<nargs; i++){
		str+=" "+std::string(vargs[i]);
	}
	return str;
}
/*-------------------------------------------------------------------------------------------*/
std::string JUtil::getInputValue(const std::string& input, std::string param){
	std::string sep("=");
	size_t pos=input.find(param+sep);
	if (pos==std::string::npos){
		std::cout<<"Warning: param "<<param<<" not found"<<std::endl;
		return "";
	}
	else{
		size_t pos_2=input.find(sep,pos);
		if (pos_2==std::string::npos){
				std::cout<<"Warning: param value for "<<param<<" not found"<<std::endl;
				return "";
		}
		else{
			size_t pos_3=input.find(" ",pos_2);
			if (pos_3==std::string::npos){
				pos_3=input.size();
			}
			return input.substr(pos_2+1,pos_3-(pos_2+1));
		}
	}
}
/*-------------------------------------------------------------------------------------------*/
std::string JUtil::getBasename(std::string input){
	std::string basename;
	size_t pos_slash=input.find_last_of('/');
	if (pos_slash==std::string::npos){
		pos_slash=-1;
	}
	basename=input.substr(pos_slash+1);
	return basename;
}

/*-------------------------------------------------------------------------------------------*/
std::string JUtil::getDirname(std::string input){
	std::string dirname;
	size_t pos_slash=input.find_last_of('/');
	if (pos_slash==std::string::npos){
		dirname=".";
	}
	else{
		dirname=input.substr(0,pos_slash);
	}
	return dirname;
}
/*-------------------------------------------------------------------------------------------*/
std::string JUtil::deleteExtension(std::string input){
	std::string out;
	out=input.substr(0,input.find_last_of('.'));
	return out;
}
/*-------------------------------------------------------------------------------------------*/
float JUtil::getEntropy(float* probs, int dim, bool normalized){
	float entropy=0;
	for(int i=0; i<dim; i++){
		if(probs[i]!=0){
			entropy+=probs[i]*std::log2(probs[i]);
		}
	}
	entropy=(-1)*entropy;
	if(normalized){
		float max_entropy=(-1)*std::log2(1.0/static_cast<float>(dim));
		entropy=entropy/(max_entropy);
	}
	return entropy;
}
/*-------------------------------------------------------------------------------------------*/
bool JUtil::file_is_readeable(std::string str_file){
	std::ifstream f_in(str_file);
	if (f_in.is_open()){
		f_in.close();
		return true;
	}
	else{
		return false;
	}
}
/*-------------------------------------------------------------------------------------------*/

template <class T> void JUtil::buffer_resize(T *&array, int* _cur_size, bool shrink){
	int cur_size=*_cur_size;
	T* new_array;
	if(shrink){
		new_array=new T[cur_size];
	}
	else{
		new_array=new T[cur_size*2]; //if !shrink=expand
		*_cur_size=cur_size*2;
	}
	std::memcpy(new_array, array, cur_size*sizeof(T));
	delete[] array;
	array=new_array;
}

template void JUtil::buffer_resize<float>(float *&array, int* _cur_size, bool shrink);

/*-------------------------------------------------------------------------------------------*/
float JUtil::getGaussianWeight(float d, float sigma){
		float g=0;
		g=std::exp(-std::abs(d)/(sigma*sigma));
		return g;
	}
/*-------------------------------------------------------------------------------------------*/
template <class T>  void JUtil::findMaxMin(T* v, unsigned int size,
													T* max_v, unsigned int* idx_max,
													T* min_v, unsigned int* idx_min,
													int start_pos, int end_pos){
	if(size>0){

		int start=0;
		int end=size-1;
		if (start_pos>-1 && start_pos<=end_pos){
			start=start_pos;
			end=end_pos;
		}
		*max_v=v[start];
		*min_v=v[start];
		*idx_max=start;
		*idx_min=start;
		for(int i=start+1; i<=end; i++){
			if(v[i]>*max_v){
				*max_v=v[i];
				*idx_max=i;
			}
			if(v[i]<*min_v){
				*min_v=v[i];
				*idx_min=i;
			}
		}
	}
}
/*-------------------------------------------------------------------------------------------*/
template void JUtil::findMaxMin(float* v, unsigned int size,
									float* max_v, unsigned int* idx_max,
									float* min_v, unsigned int* idx_min,
									int start_pos, int end_pos);
/*-------------------------------------------------------------------------------------------*/
std::string JUtil::str_replace(std::string &s,
                      const std::string &toReplace,
                      const std::string &replaceWith)
{
	if (s.find(toReplace)!=std::string::npos)
		return(s.replace(s.find(toReplace), toReplace.length(), replaceWith));
	else
		return s;
}
/*-------------------------------------------------------------------------------------------*/
void JUtil::colage(const std::vector<std::string>& vec_im, cv::Mat& mat_out, cv::Size size,
					int n_rows, int n_cols,
					int h_space, int v_space, bool first_is_query){

	mat_out.create(size, CV_8UC3);
	int n_images=vec_im.size();
	JUtil::jmsr_assert((n_rows*n_cols)>=n_images, "grid<n_images");
	int image_width=(size.width-(n_cols+1)*h_space)/n_cols;
	int image_height=(size.height-(n_rows+1)*v_space)/n_rows;
	cv::Mat mat_image;
	cv::Rect rect(0,0,image_width, image_height);
	int row=0, col=0;
	for(int i=0; i<n_images; i++){
		row=i/n_cols;
		col=i%n_cols;
		mat_image=cv::imread(vec_im[i]);
		cv::resize(mat_image, mat_image, cv::Size(image_width, image_height));
		rect.x=col*image_width+h_space;
		rect.y=row*image_height+v_space;
		mat_image.copyTo(mat_out(rect));
	}
	if(first_is_query){
		rect.y=v_space;
		rect.x=h_space;
		cv::rectangle(mat_out, rect, cv::Scalar(0,0,255), 2);
	}

}

bool JUtil::fileExists(const std::string& str_filename){
	std::ifstream f_in(str_filename);
	if (f_in.is_open()){
		f_in.close();
		return true;

	}
	return false;
}

