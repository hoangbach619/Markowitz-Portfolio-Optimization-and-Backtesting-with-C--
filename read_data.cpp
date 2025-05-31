#include "read_data.h"
#include "csv.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <numeric>


using namespace std;


//g++ -c read_data.cpp
// g++ -c csv.cpp
// g++ -o portfolioSolver csv.o read_data.o
// ./portfolioSolver


double string_to_double(const std::string& s) {
    istringstream i(s);
    double x;
    if (!(i >> x))
        return 0.0;
    return x;
}

void readData(double **data, const std::string& fileName)
{
	char tmp[20];
	ifstream file (strcpy(tmp, fileName.c_str()));
	Csv csv(file);
	string line;
	if (file.is_open())
	{
		int i=0;
		while (csv.getline(line) != 0) {
         	for (int j = 0; j < csv.getnfield(); j++)
            {
               double temp=string_to_double(csv.getfield(j));
               //cout << "Asset " << j << ", Return "<<i<<"="<< temp<<"\n";
               data[j][i]=temp;
            }
            i++;
		}
		
		file.close();
	}
	else {cout <<fileName <<" missing\n";exit(0);}
                                                                 }
void computeMean(double** data,int numberAssets,int t_start,int window,vector<double>& mean) {
    for (int i = 0; i < numberAssets; ++i) {
        double sum = 0.0;
        for (int j = 0; j < window; ++j)
            sum += data[i][t_start + j];
        mean[i] = sum / window;
    }
}

void computeCovariance(double** data,int numberAssets,int t_start,int window,const vector<double>& mean,vector<vector<double> >& cov) {
    for (int i = 0; i < numberAssets; ++i) {
        for (int j = 0; j < numberAssets; ++j) {
            double sum = 0.0;
            for (int k = 0; k < window; ++k) {
                double ri = data[i][t_start + k] - mean[i];
                double rj = data[j][t_start + k] - mean[j];
                sum += ri * rj;
            }
            cov[i][j] = sum / (window - 1);
        }
    }
}

void conjugateGradient(const vector< vector<double> >& Q,const vector<double>& b,vector<double>& x,double epsilon) {
    int n = Q.size();

    vector<double> r(n); 
    vector<double> p(n); 
    vector<double> Qp(n); 

    // r0 = b - Q * x0
    for (int i = 0; i < n; ++i) {
        double Qxi = 0.0;
        for (int j = 0; j < n; ++j){
            Qxi += Q[i][j] * x[j];
        }
        r[i] = b[i] - Qxi;
        p[i] = r[i]; 
    }

    double rs_old = 0.0;
    for (int i = 0; i < n; ++i){
        rs_old += r[i] * r[i];
    
    }

    for (int k = 0; k < n; ++k) {
        
        // Qp_k = Q * p_k
        for (int i = 0; i < n; ++i) {
            Qp[i] = 0.0;
            for (int j = 0; j < n; ++j)
                Qp[i] += Q[i][j] * p[j];
        }

        double alpha_num = rs_old;
        double alpha_den = 0;
        for (int i = 0; i < n; ++i){
            alpha_den += p[i] * Qp[i];
        }
        double alpha = alpha_num / alpha_den;

        //  x = x + alpha * p
        for (int i = 0; i < n; ++i)
            x[i] = x[i] + alpha * p[i];

        // r = r - alpha * Qp
        for (int i = 0; i < n; ++i){
            r[i] -= alpha * Qp[i];
        }


        double rs_new = 0.0;
        for (int i = 0; i < n; ++i){
            rs_new += r[i] * r[i];
        }

        if (sqrt(rs_new) < epsilon){
            break;
        }
        // beta_k = (r_(k+1)^T * r_(k+1)) / (r_k^T * r_k) 
        double beta = rs_new / rs_old;

        // p_(k+1) = r_(k+1) + beta_k * p_k
        for (int i = 0; i < n; ++i)
            p[i] = r[i] + beta * p[i];

        rs_old = rs_new;
    }
}

int main() {
    const int numberAssets = 83;
    const int numberReturns = 700;
    const int window = 100;
    const int step = 12;
    const int numberTargets = 20;

    double** returnMatrix = new double*[numberAssets]; // a matrix to store the return data
    //allocate memory for return data
    for (int i = 0; i < numberAssets; ++i)
        returnMatrix[i] = new double[numberReturns];
    
    //read the data from the file and store it into the return matrix
    string fileName="asset_returns.csv";
    readData(returnMatrix, fileName);
    // returnMatrix[i][j] stores the asset i, return j value

    vector<double> meanInSample(numberAssets);
    vector< vector<double> > inCov(numberAssets, vector<double>(numberAssets));

    ofstream results("results.csv");
    results << "window,target,Return,Risk, Residual\n";


    // Rolling backtest
    int rollingWindow = 0;
    for (int t_start = 0; t_start + window + step <= numberReturns; t_start += step) {
        computeMean(returnMatrix, numberAssets, t_start, window, meanInSample);
        computeCovariance(returnMatrix, numberAssets, t_start, window, meanInSample, inCov);
        for (int t = 0; t <= numberTargets; ++t) {
            double target = t * 0.10 / numberTargets;
            int dimension = numberAssets + 2;
        
           
            vector< vector<double> > Q(dimension, vector<double>(dimension, 0.0));
            vector<double> b(dimension, 0.0);
           

            for (int i = 0; i < numberAssets; ++i)
                for (int j = 0; j < numberAssets; ++j)
                    Q[i][j] = inCov[i][j];

            for (int i = 0; i < numberAssets; ++i) {
                double ri = meanInSample[i];
                Q[i][numberAssets] = -ri;
                Q[i][numberAssets + 1] = -1.0;
                Q[numberAssets][i] = -ri;
                Q[numberAssets + 1][i] = -1.0;
            }
            b[numberAssets] = -target;
            b[numberAssets + 1] = -1.0;
            vector<double> weights(dimension, 1.0);
            conjugateGradient(Q, b, weights, 1e-6);




            // CSV Export for validation
            if (fabs(target - 0.03) < 1e-6) { 
                ofstream outW("weights.csv");
                for (int i = 0; i < dimension; ++i)
                    outW << weights[i] << "\n";
                outW.close();

                ofstream outB("b.csv");
                for (int i = 0; i < dimension; ++i)
                    outB << b[i] << "\n";
                outB.close();

                ofstream outQ("Q.csv");
                for (int i = 0; i < dimension; ++i) {
                    for (int j = 0; j < dimension; ++j) {
                        outQ << Q[i][j];
                        if (j < dimension - 1) outQ << ",";
                    }
                    outQ << "\n";
                }
                outQ.close();
            }
            cout << "[CG] sample weights: ";
            for (int i = 0; i < 5; ++i) cout << weights[i] << "  ";
            cout << "\n";

            // Check
        
            double sumw = 0.0;
            for (int i = 0; i < numberAssets; ++i)
                sumw += weights[i];
            cout << "Sum of weight = " << sumw << "\n";

            double actualReturn = 0.0;
            for (int i = 0; i < numberAssets; ++i)
                actualReturn += meanInSample[i] * weights[i];
            cout << "Actual return = " << actualReturn << ", Target = " << target << "\n";


            vector<double> Qx(dimension, 0.0);
            for (int i = 0; i < dimension; ++i)
                for (int j = 0; j < dimension; ++j)
                    Qx[i] += Q[i][j] * weights[j];
            double residual = 0.0;
            for (int i = 0; i < dimension; ++i)
                residual += pow(Qx[i] - b[i], 2);
            residual = sqrt(residual);
            cout << "Residual 'Qx - b' = " << residual << "\n";

            // Out-of-sample 
            vector<double> meanOutSample(numberAssets);
            vector< vector<double> > outCov(numberAssets, vector<double>(numberAssets));
            computeMean(returnMatrix, numberAssets, t_start + window, step, meanOutSample);
            computeCovariance(returnMatrix, numberAssets, t_start + window, step, meanOutSample, outCov);
        
            double realizedReturn = 0.0;
            for (int i = 0; i < numberAssets; ++i)
                realizedReturn += meanOutSample[i] * weights[i];
            double realizedVar = 0.0;
            for (int i = 0; i < numberAssets; ++i)
                for (int j = 0; j < numberAssets; ++j)
                    realizedVar += weights[i] * outCov[i][j] * weights[j];
            double realizedRisk = sqrt(realizedVar);

            results << rollingWindow << "," << target << "," << realizedReturn << "," << realizedRisk << "," << residual << "\n";

            cout << "Window["<< t_start << "] target="<< target
                 << ", OOS_r="<< realizedReturn
                 << ", sigma="<< realizedRisk << "\n";
        }
        rollingWindow++;

    }

    results.close();

    // Cleanup
    for (int i = 0; i < numberAssets; ++i)
        delete[] returnMatrix[i];
    delete[] returnMatrix;

    return 0;
}
