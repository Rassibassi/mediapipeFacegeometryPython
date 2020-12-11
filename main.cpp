// Many parts taken from github.com/google/mediapipe 
//
// Copyright 2020 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//
// g++ -o main main.cpp `pkg-config --libs --cflags opencv` -I/data/libraries/eigen-3.3.9 -L/usr/local/lib -lcnpy -lz --std=c++11
//

#include <iostream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "cnpy.h"

using namespace std;
using namespace Eigen;

void cnpy2eigen(string data_fname, MatrixXd& mat_out){
    cnpy::NpyArray npy_data = cnpy::npy_load(data_fname);
    // double* ptr = npy_data.data<double>();

    int data_row = npy_data.shape[0];
    int data_col = npy_data.shape[1];

    double* ptr = static_cast<double *>(malloc(data_row * data_col * sizeof(double)));

    memcpy(ptr, npy_data.data<double>(), data_row * data_col * sizeof(double));

    cv::Mat dmat = cv::Mat(cv::Size(data_col, data_row), CV_64F, ptr); // CV_64F is equivalent double

    new (&mat_out) Map<Matrix<double,Dynamic,Dynamic>>(reinterpret_cast<double *>(dmat.data), data_col, data_row);
}

void save_matrix(string s, MatrixXf M){
    const long unsigned int rows = M.rows();
    const long unsigned int cols = M.cols();

    vector<float> raw_data(rows * cols);

    if(cols == 468){
        Matrix<float, 3, 468>::Map(raw_data.data()) = M;
    }else{
        Matrix<float, 3, 3>::Map(raw_data.data()) = M;
    }
    
    cnpy::npy_save(s, &raw_data[0], {rows, cols}, "w");
}

void log_matrix(string s, MatrixXf M, int n){
    cout << s << endl;
    cout << "rows/cols: " << M.rows() << "/" << M.cols() << endl;
    cout << M.leftCols(n) << endl << endl;

    if( (M.rows() == 3 && M.cols() == 468) || (M.rows() == 3 && M.cols() == 3)){
        save_matrix(s + "_cpp.npy", M);
    }
}

void log_vector(string s, VectorXf V, int n){
    cout << s << endl;
    cout << V.head(n) << endl << endl;
}

void log_float(string s, float f){
    cout << s << endl;
    cout << f << endl << endl;
}

void ProjectXY(float pcf_right,
               float pcf_left,
               float pcf_top,
               float pcf_bottom,
               Matrix3Xf& landmarks) {
    float x_scale = pcf_right - pcf_left;
    float y_scale = pcf_top - pcf_bottom;
    float x_translation = pcf_left;
    float y_translation = pcf_bottom;

    // if (origin_point_location_ == OriginPointLocation::TOP_LEFT_CORNER) {
    //   landmarks.row(1) = 1.f - landmarks.row(1).array();
    // }

    landmarks =
        landmarks.array().colwise() * Array3f(x_scale, y_scale, x_scale);
    landmarks.colwise() += Vector3f(x_translation, y_translation, 0.f);
  }

static void ChangeHandedness(Matrix3Xf& landmarks) {
    landmarks.row(2) *= -1.f;
  }

static void UnprojectXY(float pcf_near, Matrix3Xf& landmarks) {
    landmarks.row(0) =
        landmarks.row(0).cwiseProduct(landmarks.row(2)) / pcf_near;
    landmarks.row(1) =
        landmarks.row(1).cwiseProduct(landmarks.row(2)) / pcf_near;
  }

static void MoveAndRescaleZ(float pcf_near,
                            float depth_offset, float scale,
                            Matrix3Xf& landmarks) {
    landmarks.row(2) =
        (landmarks.array().row(2) - depth_offset + pcf_near) / scale;
  }

// headers
void SolveWeightedOrthogonalProblem(
      const Matrix3Xf& source_points,
      const Matrix3Xf& target_points,
      const VectorXf& point_weights,
      Matrix4f& transform_mat);
static VectorXf ExtractSquareRoot(const VectorXf& point_weights);
static void InternalSolveWeightedOrthogonalProblem(
      const Matrix3Xf& sources, const Matrix3Xf& targets,
      const VectorXf& sqrt_weights, Matrix4f& transform_mat);
static void ComputeOptimalRotation(
      const Matrix3f& design_matrix, Matrix3f& rotation);
static float ComputeOptimalScale(
      const Matrix3Xf& centered_weighted_sources,
      const Matrix3Xf& weighted_sources,
      const Matrix3Xf& weighted_targets,
      const Matrix3f& rotation);
static Matrix4f CombineTransformMatrix(const Matrix3f& r_and_s,
                                       const Vector3f& t);
// headers end

float EstimateScale(Matrix3Xf canonical_metric_landmarks,
                    Matrix3Xf& landmarks,
                    VectorXf landmark_weights) {
    Matrix4f transform_mat;
    SolveWeightedOrthogonalProblem(canonical_metric_landmarks, landmarks, landmark_weights, transform_mat);
    
    cout << "TRANSFORM_MAT" << endl;
    cout << transform_mat << endl;

    return transform_mat.col(0).norm();
  }

void SolveWeightedOrthogonalProblem(
      const Matrix3Xf& source_points,
      const Matrix3Xf& target_points,
      const VectorXf& point_weights,
      Matrix4f& transform_mat) {

    // TODO include these validators
    // ValidateInputPoints(source_points, target_points);
    // ValidatePointWeights(source_points.cols(), point_weights);

    // Extract square root from the point weights.
    VectorXf sqrt_weights = ExtractSquareRoot(point_weights);
    log_vector("SolveWeightedOrthogonalProblem:sqrt_weights", sqrt_weights, 3);

    // Try to solve the WEOP problem.
    InternalSolveWeightedOrthogonalProblem(source_points, target_points, sqrt_weights, transform_mat);
  }

static VectorXf ExtractSquareRoot(const VectorXf& point_weights) {
    VectorXf sqrt_weights(point_weights);
    for (int i = 0; i < sqrt_weights.size(); ++i) {
      sqrt_weights(i) = sqrt(sqrt_weights(i));
    }

    return sqrt_weights;
}

static void InternalSolveWeightedOrthogonalProblem(
      const Matrix3Xf& sources, const Matrix3Xf& targets,
      const VectorXf& sqrt_weights, Matrix4f& transform_mat) {

    log_matrix("sources", sources, 3);
    log_matrix("targets", targets, 3);

    // tranposed(A_w).
    Matrix3Xf weighted_sources =
        sources.array().rowwise() * sqrt_weights.array().transpose();
    // tranposed(B_w).
    Matrix3Xf weighted_targets =
        targets.array().rowwise() * sqrt_weights.array().transpose();

    log_matrix("weighted_sources", weighted_sources, 3);
    log_matrix("weighted_targets", weighted_targets, 3);

    // w = tranposed(j_w) j_w.
    float total_weight = sqrt_weights.cwiseProduct(sqrt_weights).sum();
    log_float("total_weight", total_weight);

    // Let C = (j_w tranposed(j_w)) / (tranposed(j_w) j_w).
    // Note that C = tranposed(C), hence (I - C) = tranposed(I - C).
    //
    // tranposed(A_w) C = tranposed(A_w) j_w tranposed(j_w) / w =
    // (tranposed(A_w) j_w) tranposed(j_w) / w = c_w tranposed(j_w),
    //
    // where c_w = tranposed(A_w) j_w / w is a k x 1 vector calculated here:
    Matrix3Xf twice_weighted_sources =
        weighted_sources.array().rowwise() * sqrt_weights.array().transpose();
    Vector3f source_center_of_mass =
        twice_weighted_sources.rowwise().sum() / total_weight;
    log_vector("source_center_of_mass", source_center_of_mass, 3);

    // tranposed((I - C) A_w) = tranposed(A_w) (I - C) =
    // tranposed(A_w) - tranposed(A_w) C = tranposed(A_w) - c_w tranposed(j_w).
    Matrix3Xf centered_weighted_sources =
        weighted_sources - source_center_of_mass * sqrt_weights.transpose();

    log_matrix("centered_weighted_sources", centered_weighted_sources, 3);

    Matrix3f rotation;
    Matrix3f design_matrix = weighted_targets * centered_weighted_sources.transpose();
    log_matrix("design_matrix", design_matrix, 3);
    ComputeOptimalRotation(design_matrix, rotation);    

    float scale = ComputeOptimalScale(centered_weighted_sources, weighted_sources, weighted_targets, rotation);
    log_float("scale", scale);

    // R = c tranposed(T).
    Matrix3f rotation_and_scale = scale * rotation;
    log_matrix("rotation_and_scale", rotation_and_scale, 3);

    // Compute optimal translation for the weighted problem.

    // tranposed(B_w - c A_w T) = tranposed(B_w) - R tranposed(A_w) in (54).
    const auto pointwise_diffs =
        weighted_targets - rotation_and_scale * weighted_sources;
    log_matrix("pointwise_diffs", pointwise_diffs, 3);
    // Multiplication by j_w is a respectively weighted column sum.
    // (54) from the paper.
    const auto weighted_pointwise_diffs =
        pointwise_diffs.array().rowwise() * sqrt_weights.array().transpose();
    log_matrix("weighted_pointwise_diffs", weighted_pointwise_diffs, 3);
    Vector3f translation =
        weighted_pointwise_diffs.rowwise().sum() / total_weight;
    log_vector("translation", translation, 3);

    transform_mat = CombineTransformMatrix(rotation_and_scale, translation);

    log_matrix("transform_mat", transform_mat.topLeftCorner(3,3), 3);
  }

static void ComputeOptimalRotation(
      const Matrix3f& design_matrix, Matrix3f& rotation) {

    float design_matrix_norm = design_matrix.norm();
    log_float("design_matrix_norm", design_matrix_norm);

    if( design_matrix_norm < 1e-9f) {
        cout << "Design matrix norm is too small!" << endl << endl;
    }

    JacobiSVD<Matrix3f> svd(
        design_matrix, ComputeFullU | ComputeFullV);

    Matrix3f postrotation = svd.matrixU();
    Matrix3f prerotation = svd.matrixV().transpose();

    // Disallow reflection by ensuring that det(`rotation`) = +1 (and not -1),
    // see "4.6 Constrained orthogonal Procrustes problems"
    // in the Gower & Dijksterhuis's book "Procrustes Analysis".
    // We flip the sign of the least singular value along with a column in W.
    //
    // Note that now the sum of singular values doesn't work for scale
    // estimation due to this sign flip.
    if (postrotation.determinant() * prerotation.determinant() <
        static_cast<float>(0)) {
      postrotation.col(2) *= static_cast<float>(-1);
    }

    log_matrix("postrotation", postrotation, 3);
    log_matrix("prerotation", prerotation, 3);

    // Transposed (52) from the paper.
    rotation = postrotation * prerotation;

    log_matrix("rotation", rotation, 3);
  }

static float ComputeOptimalScale(
      const Matrix3Xf& centered_weighted_sources,
      const Matrix3Xf& weighted_sources,
      const Matrix3Xf& weighted_targets,
      const Matrix3f& rotation) {
    // tranposed(T) tranposed(A_w) (I - C).
    const auto rotated_centered_weighted_sources =
        rotation * centered_weighted_sources;
    // Use the identity trace(A B) = sum(A * B^T)
    // to avoid building large intermediate matrices (* is Hadamard product).
    // (53) from the paper.
    float numerator =
        rotated_centered_weighted_sources.cwiseProduct(weighted_targets).sum();
    float denominator =
        centered_weighted_sources.cwiseProduct(weighted_sources).sum();

    if( denominator < 1e-9f) {
        cout << "Scale expression denominator is too small!" << endl << endl;
    }
    if( numerator / denominator < 1e-9f) {
        cout << "Scale is too small!" << endl << endl;
    }

    return numerator / denominator;
  }

static Matrix4f CombineTransformMatrix(const Matrix3f& r_and_s,
                                       const Vector3f& t) {
    Matrix4f result = Matrix4f::Identity();
    result.leftCols(3).topRows(3) = r_and_s;
    result.col(3).topRows(3) = t;

    return result;
  }

int main(){
    // {'bottom': -0.8934218078764548,
    // 'far': 10000,
    // 'fov_y': 1.4583376618561699,
    // 'left': -0.5025497669305058,
    // 'near': 1,
    // 'right': 0.5025497669305058,
    // 'top': 0.8934218078764548}

    float pcf_bottom = -0.8934218078764548;
    float pcf_far = 10000;
    float fov_y = 1.4583376618561699;
    float pcf_left = -0.5025497669305058;
    float pcf_near = 1.0;
    float pcf_right = 0.5025497669305058;
    float pcf_top = 0.8934218078764548;    

    MatrixXd canonical_metric_landmarks_d;
    cnpy2eigen("canonical_metric_landmarks.npy", canonical_metric_landmarks_d);
    canonical_metric_landmarks_d.transposeInPlace();
    Matrix3Xf canonical_metric_landmarks = canonical_metric_landmarks_d.cast <float> ();
    log_matrix("canonical_metric_landmarks", canonical_metric_landmarks, 3);

    MatrixXd landmark_weights_d;
    cnpy2eigen("landmark_weights.npy", landmark_weights_d);
    landmark_weights_d.transposeInPlace();
    MatrixXf landmark_weights_M = landmark_weights_d.cast <float> ();
    VectorXf landmark_weights = landmark_weights_M.row(0);
    log_vector("landmark_weights", landmark_weights, 3);
    
    MatrixXd screen_landmarks_d;
    cnpy2eigen("landmarks.npy", screen_landmarks_d);
    screen_landmarks_d.transposeInPlace();
    Matrix3Xf screen_landmarks = screen_landmarks_d.cast <float> ();
    log_matrix("landmarks", screen_landmarks, 3);

    ProjectXY(pcf_right, pcf_left, pcf_top, pcf_bottom, screen_landmarks);
    const float depth_offset = screen_landmarks.row(2).mean();
    log_float("depth_offset", depth_offset);

    Matrix3Xf intermediate_landmarks(screen_landmarks);
    ChangeHandedness(intermediate_landmarks);

    float first_iteration_scale = EstimateScale(canonical_metric_landmarks,intermediate_landmarks,landmark_weights);
    log_float("first_iteration_scale", first_iteration_scale);

    // 2nd iteration: unproject XY using the scale from the 1st iteration.
    intermediate_landmarks = screen_landmarks;
    MoveAndRescaleZ(pcf_near, depth_offset, first_iteration_scale, intermediate_landmarks);
    UnprojectXY(pcf_near, intermediate_landmarks);
    ChangeHandedness(intermediate_landmarks);
    float second_iteration_scale = EstimateScale(canonical_metric_landmarks,intermediate_landmarks,landmark_weights);
    log_float("second_iteration_scale", second_iteration_scale);

    // Use the total scale to unproject the screen landmarks.
    float total_scale = first_iteration_scale * second_iteration_scale;
    MoveAndRescaleZ(pcf_near, depth_offset, total_scale, screen_landmarks);
    UnprojectXY(pcf_near, screen_landmarks);
    ChangeHandedness(screen_landmarks);

    // At this point, screen landmarks are converted into metric landmarks.
    Matrix3Xf& metric_landmarks = screen_landmarks;
    log_matrix("metric_landmarks", metric_landmarks, 3);

    return 0;
}