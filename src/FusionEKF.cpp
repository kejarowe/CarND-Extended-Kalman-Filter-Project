#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225,      0,
                   0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09,      0,    0,
                 0, 0.0009,    0,
                 0,      0, 0.09;

  //measurement matrix - laser
  H_laser_ << 1,0,0,0,
              0,1,0,0;

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  static MatrixXd Q_nu = MatrixXd(2,2);
  static MatrixXd G = MatrixXd(4,2);

  if (!is_initialized_) {
    /**
    TODO:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement
    //cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      //unpack measurement
      float rho = measurement_pack.raw_measurements_(0);
      float phi = measurement_pack.raw_measurements_(1);
      float rho_dot = measurement_pack.raw_measurements_(2);

      //calculate initial state
      ekf_.x_ << rho * cos(phi), rho * sin(phi), rho_dot * cos(phi), rho_dot * sin(phi); //phi_dot is unknown, so assume 0

      //init measurement jacobian
      Hj_ = tools.CalculateJacobian(ekf_.x_);

      //init state covariance
      ekf_.P_ = Hj_.transpose() * R_radar_ * Hj_;


    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      ekf_.x_ = H_laser_.transpose() * measurement_pack.raw_measurements_;
      ekf_.P_ = H_laser_.transpose() * R_laser_ * H_laser_;   
    }

    //the initial transition matrix F_
    ekf_.F_ = MatrixXd(4, 4);
    ekf_.F_ << 1, 0, 1, 0,
               0, 1, 0, 1,
               0, 0, 1, 0,
               0, 0, 0, 1;

    //using sigma_ax = sigma_ay = 9
    Q_nu << 9, 0,
            0, 9;

    previous_timestamp_ = measurement_pack.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
   TODO:
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use sigma_ax = 9 and sigma_ay = 9 for your Q matrix.
   */

  double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  ekf_.F_(0,2) = dt;
  ekf_.F_(1,3) = dt;

  double dt_sq_div_2 = dt * dt / 2.0;

  G << dt_sq_div_2,           0,
                 0, dt_sq_div_2,
                 dt,          0,
                 0,          dt;

  ekf_.Q_ = G * Q_nu * G.transpose();

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);

    // Measurement Jacobian is invalid, skip the measurement
    if (ekf_.H_.sum() == 0){
      cout << "skipped radar update" << endl;
      return;
    }
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  //cout << "x_ = " << ekf_.x_ << endl;
  //cout << "P_ = " << ekf_.P_ << endl;
}
