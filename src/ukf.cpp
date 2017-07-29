#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */

const int STATE_DIMENSION             = 5;
const int AUG_STATE_DIMENSION         = 7;
const int LASER_MEASUREMENT_DIMENSION = 2;
const int RADAR_MEASUREMENT_DIMENSION = 3;

const double ACCURACY_CHECK_VALUE = 0.001;

const double DT_TIME_CONVERSION = 1000000.0;

const double STD_A_NOISE     = 1.5;
const double STD_YAWDD_NOISE = 0.5;

const double STD_LASER_PX  = 0.15;
const double STD_LASER_PY  = 0.15;

const double STD_RADAR_R   = 0.3;
const double STD_RADAR_PHI = 0.03;
const double STD_RADAR_RD  = 0.3;

const bool USE_LASER = true;
const bool USE_RADAR = true;

UKF::UKF() {
    
    n_x_   = STATE_DIMENSION;
    n_aug_ = AUG_STATE_DIMENSION;
    
    n_aug_sig_ = 2 * n_aug_ + 1;
    
    is_initialized_ = false;
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = USE_LASER;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = USE_RADAR;

    // initial state vector
    x_ = VectorXd::Zero(n_x_);

    // initial covariance matrix
    P_ = MatrixXd::Zero(n_x_, n_x_);
    
    Xsig_pred_ = MatrixXd::Zero(n_x_, n_aug_sig_);

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = STD_A_NOISE;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = STD_YAWDD_NOISE;

    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = STD_LASER_PX;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = STD_LASER_PY;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = STD_RADAR_R;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = STD_RADAR_PHI;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = STD_RADAR_RD;
    
    weights_ = VectorXd::Zero(n_aug_sig_);
    
    lambda_ = 3 - n_aug_;

    /**
     TODO:

     Complete the initialization. See ukf.h for other member properties.

    Hint: one or more values initialized above might be wildly off...
     */
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
    
    bool isRadar = meas_package.sensor_type_ == MeasurementPackage::RADAR;
    bool isLaser = meas_package.sensor_type_ == MeasurementPackage::LASER;
    
    //Initialize the state if haven't done so.
    if (!is_initialized_) {
        
        float px, py, v, psi, psidot;
        previous_time_stamp_ = meas_package.timestamp_;
        
        if (isRadar) {
            float rho     = meas_package.raw_measurements_[0];
            float phi     = meas_package.raw_measurements_[1];
            float psi_dot = meas_package.raw_measurements_[2];
            
            px = rho*cos(phi);
            py = rho*sin(phi);
            v = sqrt(pow(rho * cos(psi_dot),2) + pow(rho * sin(psi_dot),2));
            x_ << px, py, v, 0, 0;
            
        } else if (isLaser) {
            px = meas_package.raw_measurements_[0];
            py = meas_package.raw_measurements_[1];
            x_ << px, py, 0, 0, 0;
        }
        
        P_ << 1,0,0,0,0,
              0,1,0,0,0,
              0,0,1,0,0,
              0,0,0,1,0,
              0,0,0,0,1;
        
        weights_(0) = (lambda_) / (double)(lambda_ + AUG_STATE_DIMENSION);
        for(int i=1; i<weights_.size(); i++) {
            weights_(i) = (0.5) / (lambda_ + AUG_STATE_DIMENSION);
        }
        
        is_initialized_ = true;
        return;
    }
    
    //Prediction
    float dt = (meas_package.timestamp_ - previous_time_stamp_) / DT_TIME_CONVERSION;
    previous_time_stamp_ = meas_package.timestamp_;
    
    Prediction(dt);
    
    if (isRadar && use_radar_) {
        UpdateRadar(meas_package);
    } else if (isLaser && use_laser_) {
        UpdateLidar(meas_package);
    }
    
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
    
    VectorXd x_aug    = VectorXd::Zero(n_aug_);
    VectorXd x_calc   = VectorXd::Zero(n_x_);
    MatrixXd P_calc   = MatrixXd::Zero(n_x_, n_x_);
    MatrixXd P_aug    = MatrixXd::Zero(n_aug_, n_aug_);
    MatrixXd Xsig_aug = MatrixXd::Zero(n_aug_, n_aug_sig_);
    
    x_aug.head(n_x_) = x_;
    
    P_aug.topLeftCorner(n_x_,n_x_) = P_;
    P_aug(n_x_,n_x_)     = pow(std_a_, 2);
    P_aug(n_x_+1,n_x_+1) = pow(std_yawdd_,2);
    
    MatrixXd L = P_aug.llt().matrixL();
    
    Xsig_aug.col(0) = x_aug;
    for (int i=0; i<n_aug_; i++)
    {
        Xsig_aug.col(i+1)        = x_aug + (sqrt(lambda_+n_aug_) * L.col(i));
        Xsig_aug.col(i+1+n_aug_) = x_aug - (sqrt(lambda_+n_aug_) * L.col(i));
    }
    
    for(int j=0; j<n_aug_sig_; j++)
    {
        VectorXd prediction = VectorXd::Zero(n_x_);
        VectorXd process_model = VectorXd::Zero(n_x_);
        VectorXd process_noise_covariance = VectorXd::Zero(n_x_);
        
        VectorXd aug_state = Xsig_aug.col(j).head(n_x_);
        VectorXd aug_noise = Xsig_aug.col(j).tail(2);
        
        float px_k       = aug_state(0);
        float py_k       = aug_state(1);
        float v_k        = aug_state(2);
        float psi_k      = aug_state(3);
        float psidot_k   = aug_state(4);
        
        float mu_a_k   = aug_noise(0);
        float mu_yawdd = aug_noise(1);
        
        float f_px_k, f_py_k;
         
        if (psidot_k < ACCURACY_CHECK_VALUE) {
            
            f_px_k = v_k * cos(psi_k) * delta_t;
        
            f_py_k = v_k * sin(psi_k) * delta_t;
            
        } else {
            f_px_k = (v_k / psidot_k) *
                     (sin(psi_k + psidot_k*delta_t) - sin(psi_k));
        
            f_py_k = (v_k / psidot_k) *
                     (-1 * cos(psi_k + psidot_k*delta_t) + cos(psi_k));
        }
        
        float f_v_k      = 0;
        float f_psi_k    = psidot_k * delta_t;
        float f_psidot_k = 0;
        
        while(f_psi_k > M_PI){
            f_psi_k -= 2*M_PI;
        }
        
        while(f_psi_k < -M_PI){
            f_psi_k += 2*M_PI;
        }
        
        process_model << f_px_k,
                         f_py_k,
                         f_v_k,
                         f_psi_k,
                         f_psidot_k;
        
        float noise_px_k     = (0.5) * pow(delta_t, 2) * cos(psi_k) * mu_a_k;
        float noise_py_k     = (0.5) * pow(delta_t, 2) * sin(psi_k) * mu_a_k;
        float noise_v_k      = delta_t * mu_a_k;
        float noise_psi_k    = (0.5) * pow(delta_t, 2) * mu_yawdd;
        float noise_psidot_k = delta_t * mu_yawdd;
        
        process_noise_covariance << noise_px_k,
                                    noise_py_k,
                                    noise_v_k,
                                    noise_psi_k,
                                    noise_psidot_k;
        
        prediction = aug_state + process_model + process_noise_covariance;
        
        Xsig_pred_.col(j) = prediction;
    }
    
    for(int k=0; k<n_aug_sig_; k++)
    {
        x_calc += weights_(k) * Xsig_pred_.col(k);
    }
    
    for(int l=0; l<n_aug_sig_; l++)
    {
        VectorXd a = (Xsig_pred_.col(l) - x_calc);
        
        float psi = a(3);
        
        while(psi > M_PI){
            psi -= 2*M_PI;
        }
        
        while(psi < -M_PI){
            psi += 2*M_PI;
        }
        P_calc += weights_(l) * a * a.transpose();
    }
    
    x_ = x_calc;
    P_ = P_calc;
    
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
    
    float z1 = meas_package.raw_measurements_[0];
    float z2 = meas_package.raw_measurements_[1];
    MatrixXd I = MatrixXd::Zero(STATE_DIMENSION,
                                STATE_DIMENSION);
    MatrixXd H = MatrixXd::Zero(LASER_MEASUREMENT_DIMENSION,
                                STATE_DIMENSION);
    VectorXd z = VectorXd::Zero(LASER_MEASUREMENT_DIMENSION);
    MatrixXd R = MatrixXd::Zero(LASER_MEASUREMENT_DIMENSION,
                                LASER_MEASUREMENT_DIMENSION);
    VectorXd y = VectorXd::Zero(2);
    
    z <<    z1,
            z2;
    
    I <<    1,0,0,0,0,
            0,1,0,0,0,
            0,0,1,0,0,
            0,0,0,1,0,
            0,0,0,0,1;
    
    H <<    1,0,0,0,0,
            0,1,0,0,0;
    
    R(0,0) = pow(std_laspx_,2);
    R(1,1) = pow(std_laspy_,2);
    
    y = z - H * x_;
    
    MatrixXd S = H * P_ * H.transpose() + R;
    MatrixXd K = P_ * H.transpose() * S.inverse();
    x_ = x_ + K * y;
    P_ = (I - K * H) * P_;
    
    VectorXd zsub = y;
    
    float NIS = zsub.transpose() * S.inverse() * zsub;
    
    cout << "NIS (LIDAR): " << NIS << endl;
    cout << "x_ = " << x_ << endl;
    cout << "P_ = " << P_ << endl << endl;
    
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
    
    int n_z = 3;
    
    double z1 = meas_package.raw_measurements_[0];
    double z2 = meas_package.raw_measurements_[1];
    double z3 = meas_package.raw_measurements_[2];
    
    MatrixXd Zsig   = MatrixXd::Zero(RADAR_MEASUREMENT_DIMENSION, 2 * n_aug_ + 1);
    MatrixXd S      = MatrixXd::Zero(n_z,n_z);
    VectorXd z_pred  = VectorXd::Zero(n_z);
    VectorXd z       = VectorXd::Zero(n_z);
    
    z <<    z1,
            z2,
            z3;
    
    for (int i=0; i<n_aug_sig_; i++) {
        VectorXd sigma_point = Xsig_pred_.col(i);
        
        double sig_px  = sigma_point(0);
        double sig_py  = sigma_point(1);
        double sig_v   = sigma_point(2);
        double sig_psi = sigma_point(3);
        
        double rho    = sqrt(pow(sig_px,2)+pow(sig_py,2));
        double phi    = atan2(sig_py,sig_px);
        double rhodot = (sig_px*cos(sig_psi)*sig_v+sig_py*sin(sig_psi)*sig_v) /
                        sqrt(pow(sig_px,2)+pow(sig_py,2));
        
        VectorXd measure_sig = VectorXd(RADAR_MEASUREMENT_DIMENSION);
        measure_sig <<  rho,
                        phi,
                        rhodot;
        
        Zsig.col(i) = measure_sig;
    }
    
    //calculate mean predicted measurement
    for(int j=0; j<n_aug_sig_; j++){
        z_pred += weights_(j) * Zsig.col(j);
    }
    
    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd::Zero(STATE_DIMENSION, n_z);
    
    for(int m=0; m<n_aug_sig_; m++){
        VectorXd z_diff = Zsig.col(m)-z_pred;
        while(z_diff(1) > M_PI){
            z_diff(1) -= 2*M_PI;
        }
        
        while(z_diff(1) < -M_PI){
            z_diff(1) += 2*M_PI;
        }
        
        VectorXd x_diff = Xsig_pred_.col(m)-x_;
        while(x_diff(3) > M_PI){
            x_diff(3) -= 2*M_PI;
        }
        
        while(x_diff(3) < -M_PI){
            x_diff(3) += 2*M_PI;
        }
        
        Tc += weights_(m) * x_diff * z_diff.transpose();
        S  += weights_(m) * z_diff * z_diff.transpose();
    }
    
    MatrixXd R = MatrixXd::Zero(3,3);
    R(0,0) = pow(std_radr_,2);
    R(1,1) = pow(std_radphi_,2);
    R(2,2) = pow(std_radrd_,2);
    
    S += R;
    
    //calculate Kalman gain K;
    MatrixXd K = MatrixXd::Zero(STATE_DIMENSION, STATE_DIMENSION);
    K = Tc * S.inverse();
    
    //update state mean and covariance matrix
    
    VectorXd z_diff = z - z_pred;
    
    while(z_diff(1) > M_PI){
        z_diff(1) -= 2*M_PI;
    }
    
    while(z_diff(1) < -M_PI){
        z_diff(1) += 2*M_PI;
    }
    
    x_ = x_ + K * (z_diff);
    P_ = P_ - K * S * K.transpose();
    
    //Calculate NIS
    
    float NIS = z_diff.transpose() * S.inverse() * z_diff;
    
    cout << "NIS (RADAR): " NIS << endl;
    cout << "x_ = " << x_ << endl;
    cout << "P_ = " << P_ << endl << endl;
}
