/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles
  
  
  std::default_random_engine gen;
  
  // Standard deviations for x, y, and theta
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];  
  
  // Normal (Gaussian) distribution for x, y and theta
  normal_distribution<double> dist_x(x, std_x); 
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);
  
  
  for (int i = 0; i < num_particles; ++i) {
    Particle particle_init;    
    particle_init.id = i;
    particle_init.x = dist_x(gen);
    particle_init.y = dist_y(gen);
    particle_init.theta = dist_theta(gen);
    particle_init.weight = 1.0;
    
    // particles vector is used by main.cpp
    particles.push_back(particle_init);
    
    
  }
   is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  
  
  std::default_random_engine gen;
  
  for (auto &particle : particles){
    
    double x_0 = particle.x;
    double y_0 = particle.y;
    double theta_0 = particle.theta;
    
    // yaw rate != 0
    if (fabs(yaw_rate) > 1e-5){
      particle.x = x_0 + (velocity / yaw_rate) * (sin(theta_0 + yaw_rate * delta_t) - (sin(theta_0)));
      particle.y = y_0 + (velocity / yaw_rate) * (cos(theta_0) - cos(theta_0 + (yaw_rate * delta_t)));
      particle.theta = theta_0 + (yaw_rate * delta_t);                                                
                                                      
    }else {
      // yaw rate = 0 
      particle.x =  x_0 + velocity * delta_t * cos(theta_0);
      particle.y =  y_0 + velocity * delta_t * sin(theta_0);
      particle.theta =  theta_0;
      
    }
                                                      
  
  
  // Standard deviations for x, y, and theta
  double std_x = std_pos[0];
  double std_y = std_pos[1];
  double std_theta = std_pos[2];  
  
  // Normal (Gaussian) distribution for x, y and theta
  normal_distribution<double> dist_x(particle.x, std_x); 
  normal_distribution<double> dist_y(particle.y, std_y);
  normal_distribution<double> dist_theta(particle.theta, std_theta);             
                                                      
  particle.x = dist_x(gen);
  particle.y = dist_y(gen);
  particle.theta = dist_theta(gen);                                                    
  
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (auto& obs : observations){
    double dist_inf = 1e3;
    
    for (auto& pred : predicted){
      // Computes the Euclidean distance between two 2D points.
      double d =  dist(obs.x, obs.y, pred.x, pred.y);
      if (d < dist_inf){
       dist_inf = d;
       obs.id = pred.id; 
      }
      
    }
    
  
  }
  

}



  


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {

   /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  double obs_x_temp, obs_y_temp;
  double landmark_x_temp, landmark_y_temp;
  bool found_close_landmark{false};
  double exponent, update_prob_temp;

  //Store calculations
  double normalizer = 2*M_PI*std_landmark[0]*std_landmark[1];
  double sigma_x_sqr_2 = 2*pow(std_landmark[0], 2);
  double sigma_y_sqr_2 = 2*pow(std_landmark[1], 2);
  
  
  double weights_sum = 0.0;

  // loop on all particles
  for (auto &particle : particles) {

    // transform observations in map frame at particle state
    vector<LandmarkObs> observations_in_map_frame;
    for (const auto &observation : observations) {
      obs_x_temp = particle.x + cos(particle.theta) * observation.x -
                   sin(particle.theta) * observation.y;
      obs_y_temp = particle.y + sin(particle.theta) * observation.x +
                   cos(particle.theta) * observation.y;
      observations_in_map_frame.push_back(
          LandmarkObs{observation.id, obs_x_temp, obs_y_temp});
    }

    // find landmarks in range of the particle
    vector<LandmarkObs> landmarks_in_range;
    double dist_landmark_particle;
    for (const auto &landmark : map_landmarks.landmark_list) {
      dist_landmark_particle = dist(particle.x, particle.y, landmark.x_f, landmark.y_f);
      if (dist_landmark_particle <= sensor_range) {
        landmarks_in_range.push_back(LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});
      }
    }

    // data association between mapped observations and landmarks
    dataAssociation(landmarks_in_range, observations_in_map_frame);

    // reset weight
    particle.weight = 1.0;

    // calculate weight
    for (const auto &observation : observations_in_map_frame) {

      // find closest landmark to observation
      landmark_x_temp = 0.0;
      landmark_y_temp = 0.0;
      for (const auto &landmark : landmarks_in_range) {
        if (observation.id == landmark.id) {
          landmark_x_temp = landmark.x;
          landmark_y_temp = landmark.y;
          found_close_landmark = true;
          break;
        }
      }

      // update weight only if found a close landmark to observation
      if (found_close_landmark) {
        exponent = -(pow(observation.x - landmark_x_temp, 2) /
                         (2 * M_PI * sigma_x_sqr_2 * sigma_y_sqr_2) +
                     pow(observation.y - landmark_y_temp, 2) /
                         (2 * M_PI * sigma_x_sqr_2 * sigma_y_sqr_2));
        update_prob_temp = exp(exponent) / normalizer ;
        update_prob_temp = std::max(update_prob_temp, 1.0e-4);
        particle.weight = particle.weight * update_prob_temp;
      }
    }

    weights_sum += particle.weight;
  }

  for (auto &particle : particles) {
    particle.weight = particle.weight / weights_sum;
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::random_device rd;
  std::mt19937 gen(rd());
  
  
  vector<double> particle_weights;
  for (const auto &particle : particles) {
    particle_weights.push_back(particle.weight);
  }
  
  std::discrete_distribution<> d(particle_weights.begin(), particle_weights.end());
  vector<Particle> resampled_particles;
  for (int i = 0; i < num_particles; ++i) {
    resampled_particles.push_back(particles[d(gen)]);
  }
  particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}