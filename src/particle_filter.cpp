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

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
    //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this
    //   method (and others in this file).

    if (is_initialized) return;
    num_particles = 20;
    
    default_random_engine gen;
    double std_x = std[0];
    double std_y = std[1];
    double std_theta = std[2];
    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);

    for (int index = 0; index < num_particles; ++index) {
        particles.push_back(
            Particle{
                index, dist_x(gen), dist_y(gen), dist_theta(gen), 1.0, {}, {}, {}
            });
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/
    default_random_engine gen;
    double std_x = std_pos[0];
    double std_y = std_pos[1];
    double std_theta = std_pos[2];
    normal_distribution<double> dist_x(0, std_x);
    normal_distribution<double> dist_y(0, std_y);
    normal_distribution<double> dist_theta(0, std_theta);

    for (int index = 0; index < num_particles; ++index) {
        double theta = particles[index].theta;
        if (fabs(yaw_rate) > 0.001) { // avoid division by zero
            particles[index].x +=
                velocity / yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
            particles[index].y +=
                velocity / yaw_rate * (cos(theta) - cos(theta +  yaw_rate * delta_t));
            particles[index].theta += yaw_rate * delta_t;
        } else {
            particles[index].x += velocity * delta_t * cos(theta);
            particles[index].y += velocity * delta_t * sin(theta);
        }
        // add noise
        particles[index].x += dist_x(gen);
        particles[index].y += dist_y(gen);
        particles[index].theta += dist_theta(gen);
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
    //   implement this method and use it as a helper during the updateWeights
    //   phase.

    // for each observation, find nearest neighbor in predicted, and associate
    for (auto & obs : observations) {
        int nearest_id;
        double nearest_distance_squared = std::numeric_limits<double>::max();
        for (auto const & pred : predicted) {
            double distance_squared = pow(obs.x - pred.x, 2) + pow(obs.y - pred.y, 2);
            if (distance_squared < nearest_distance_squared) {
                nearest_id = pred.id;
                nearest_distance_squared = distance_squared;
            }
        }
        obs.id = nearest_id;
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

    double std_landmark_x = std_landmark[0];
    double std_landmark_y = std_landmark[1];

    for (auto & particle : particles) {
        // find landmarks in range of particle on the map
        std::vector<LandmarkObs> landmarks_in_range;
        for (auto const & landmark : map_landmarks.landmark_list) {
            if (fabs(particle.x - landmark.x_f) <= sensor_range &&
                fabs(particle.y - landmark.y_f) <= sensor_range) {
                landmarks_in_range.push_back(LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});
            }
        }

        // apply transformation (rotation and translation)
        std::vector<LandmarkObs> transformed_observations;
        for (auto const & obs : observations) {
            double transformed_x =
                obs.x * cos(particle.theta) - obs.y * sin(particle.theta) + particle.x;
            double transformed_y =
                obs.x * sin(particle.theta) + obs.y * cos(particle.theta) + particle.y;
            transformed_observations.push_back(
                LandmarkObs{obs.id, transformed_x, transformed_y}
                );
        }

        // associated landmarks with observations
        dataAssociation(landmarks_in_range, transformed_observations);
        
        particle.weight = 1.0;

        for (auto const & transformed_obs : transformed_observations) {
            // retrieve predicted observation associated with transformed observation
            auto predicted_obs = std::find_if(
                landmarks_in_range.begin(), landmarks_in_range.end(),
                [& transformed_obs](LandmarkObs const& predicted)
                {
                    return predicted.id == transformed_obs.id;
                });
            if (predicted_obs != landmarks_in_range.end()) {
                // update weight
                particle.weight *= 1 / (2 * M_PI * std_landmark_x * std_landmark_y) *
                    exp(-(pow(transformed_obs.x - predicted_obs->x, 2)/(2*pow(std_landmark_x, 2)) +
                          pow(transformed_obs.y - predicted_obs->y, 2)/(2*pow(std_landmark_y, 2))));
            }
        }
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight. 
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    std::vector<Particle> resampled_particles;

    double total_weight = std::accumulate(particles.begin(), particles.end(), 0.0,
                                          [&](double total, Particle const & particle) {
                                              return total + particle.weight;
                                          });

    std::default_random_engine gen;
    std::uniform_real_distribution<double> weight_dist(0, total_weight);

    // resample particles.size() amount of times
    for (size_t i = 0; i < particles.size(); ++i) {
        double random_weight = weight_dist(gen);
        for (auto const & particle : particles) {
            random_weight -= particle.weight;
            if (random_weight <= 0) {
                resampled_particles.push_back(particle);
                break;
            }
        }
        // in case not resampled (floating point error etc)
        if (resampled_particles.size() != (i+1)) {
            resampled_particles.push_back(particles[particles.size() - 1]);
        }
    }
    particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                         const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
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
