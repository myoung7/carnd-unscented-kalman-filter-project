#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    
    VectorXd rmse(4);
    rmse << 0,0,0,0;
    
    if(estimations.size() != ground_truth.size() || estimations.size() == 0){
        cout << "Error Tools:CalculateRMSE() - Invalid estimation or ground_truth data" << endl;
        return rmse;
    }
    
    int n=estimations.size();
    
    for(int i=0;i<n;i++)
    {
        VectorXd residual = estimations[i]-ground_truth[i];
        residual = residual.array() * residual.array();
        rmse += residual;
    }
    
    rmse /= n;
    rmse = sqrt(rmse.array());
    
    return rmse;
    
}
