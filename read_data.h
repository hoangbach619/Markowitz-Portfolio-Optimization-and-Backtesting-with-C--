#ifndef READ_DATA_H
#define READ_DATA_H

#include <string>
#include <vector>

double string_to_double(const std::string& s);
void readData(double** data, const std::string& fileName);

void computeMean(double** data,int numberAssets,int t_start,int window,std::vector<double>& mean);

void computeCovariance(double** data,int numberAssets,int t_start,int window,const std::vector<double>& mean,std::vector<std::vector<double> >& cov);

void conjugateGradient(const std::vector<std::vector<double> >& Q,const std::vector<double>& b,std::vector<double>& x,double epsilon = 1e-6);

#endif
