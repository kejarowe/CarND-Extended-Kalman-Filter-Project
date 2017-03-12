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

	if (estimations.size() == 0 || estimations.size() != ground_truth.size())
	    return rmse;

	//accumulate squared residuals
	VectorXd this_term(4);
	for(int i=0; i < estimations.size(); ++i){
	    this_term << 0,0,0,0;
	    this_term = estimations[i] - ground_truth[i];
	    this_term = this_term.array()*this_term.array();
        rmse += this_term;
	}

	//calculate the mean
	rmse /= estimations.size();

	//calculate the squared root
	rmse = rmse.array().sqrt();

	//return the result
	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

	MatrixXd Hj(3,4);
	//recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	//check division by zero
	if ((px == 0) && (py == 0))
	    return Hj.setZero(3,4);

	//compute the Jacobian matrix
    float mag = px*px+py*py;
    float root = sqrt(mag);
    float mag_32 = pow(mag,1.5);
	Hj << px/root, py/root, 0, 0,
	    -py/mag,px/mag,0,0,
	    py*(vx*py-vy*px)/mag_32,px*(vy*px-vx*py)/mag_32,px/root,py/root;
	    

	return Hj;
}
