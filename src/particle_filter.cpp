/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  // set the number of particles
  num_particles = 50;

  // creates a normal (Gaussian) distribution for x, y and theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; i++){
    Particle par;
    par.x = dist_x(gen);
    par.y = dist_y(gen);
    par.theta = dist_theta(gen);
    par.weight = 1.0;
    par.id = i;

    particles.push_back(par);
  }

  //set the flag for initialization to true
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  // default_random_engine gen;

  // creates a normal (Gaussian) distribution for x, y and theta
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; i ++){
	  
    double theta = particles[i].theta;
	  
    if (fabs(yaw_rate) > 0.0001){
      particles[i].x += velocity / yaw_rate * (sin(theta + yaw_rate*delta_t) - sin(theta));
      particles[i].y += velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate*delta_t;
    }
    else{
      particles[i].x += velocity * cos(theta) *delta_t;
      particles[i].y += velocity * sin(theta) *delta_t;
    }
  
    //adding the gaussian noise
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);     
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  for (int i = 0; i < observations.size(); i++){
    //define the minimal distance to a very large number
    double min_dis = numeric_limits<double>::max();
    //define the index of the prediction, which is the nearest to this observation
    int idx;
    
    // find the nearest prediction to this observation
    for (int j = 0; j < predicted.size(); j++){
      double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if (distance < min_dis){
        min_dis = distance;
        idx = predicted[j].id;
      }
    }
    //save the index to this observation
    observations[i].id = idx;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
  for (int i = 0; i < num_particles; i++){
    // create the vector for saving the landmarks in the range
    double x = particles[i].x;
    double y = particles[i].y;
    double theta = particles[i].theta;

    vector<LandmarkObs> Landmarks_inRange;
    // find the landmarks in the sensor range to the particle
    for (int j = 0; j < map_landmarks.landmark_list.size(); j++){
      double x_landmarks = map_landmarks.landmark_list[j].x_f;
      double y_landmarks = map_landmarks.landmark_list[j].y_f;
      double dis = dist(x_landmarks, y_landmarks, x, y);
      // if the distance of the jth landmark to the ith particle is in the sensor range, then save this landmark's information
      if (dis <= sensor_range){
        Landmarks_inRange.push_back(LandmarkObs{map_landmarks.landmark_list[j].id_x, x_landmarks, y_landmarks});
      }
    }

    //transform the vehicle's coordinates to MAP's coordinates
    vector<LandmarkObs> Observation_MAP;
    for (int j = 0; j < observations.size(); j++){
      double xm = x + cos(theta) * observations[j].x - sin(theta) * observations[j].y;
      double ym = y + sin(theta) * observations[j].x + cos(theta) * observations[j].y;
      Observation_MAP.push_back(LandmarkObs{observations[j].id, xm, ym});
    }
    
    //for each observation get the nearest landmark and save the id
    dataAssociation(Landmarks_inRange, Observation_MAP);

    particles[i].weight = 1.0;
    //get the weight 
    for (int j = 0; j < Observation_MAP.size(); j++){
      double Map_x = Observation_MAP[j].x;
      double Map_y = Observation_MAP[j].y;

      double landmark_nearest_x, landmark_nearest_y;
      int id = Observation_MAP[j].id;
      int m = 0;
      bool F_landmark = false;

      // find the landmark that has this id
      while(!F_landmark && m < Landmarks_inRange.size()){
        if (Landmarks_inRange[m].id == id){
          F_landmark = true;
          double landmark_nearest_x = Landmarks_inRange[m].x;
          double landmark_nearest_y = Landmarks_inRange[m].y;
        }
        m++;
      }

      //calculating the weight
      double delta_x = Map_x - landmark_nearest_x;
      double delta_y = Map_y - landmark_nearest_y;

      double sig_x = std_landmark[0];
      double sig_y = std_landmark[1];
      particles[i].weight *= (1/(2 * M_PI * sig_x * sig_y)) * exp(-(delta_x * delta_x /(2 * sig_x * sig_x) + delta_y * delta_y /(2 * sig_y * sig_y)));
    }
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  //default_random_engine gen;
  vector<Particle> particles_resample;

  //vector<double> weights;
  double weight_max = numeric_limits<double>::min();
  for (int i = 0; i < num_particles; i++){
    weights.push_back(particles[i].weight);
    if (particles[i].weight > weight_max){
      weight_max = particles[i].weight;
    }
  } 

  // generate distribution 
  uniform_int_distribution<int> intdist(0, num_particles-1);
  uniform_real_distribution<double>  realdist(0.0, weight_max);

  // generate the random starting index for resampling
  auto index = intdist(gen);

  double beta = 0;
  for (int i = 0; i < num_particles; i++){
    beta += realdist(gen) * 2.0;
    while(beta > weights[index]){
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    particles_resample.push_back(particles[index]);
  }
  particles = particles_resample;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //particle.associations.clear();
    //particle.sense_x.clear();
    //particle.sense_y.clear();

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    //return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
