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
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;
	std::default_random_engine gen;
	std::normal_distribution<double> ND_x(x, std[0]);
	std::normal_distribution<double> ND_y(y, std[1]);
	std::normal_distribution<double> ND_theta(theta, std[2]);

	for (int i = 0 ; i < num_particles; i++)
	{
		Particle p = {id:i, x:ND_x(gen), y:ND_y(gen), theta:ND_theta(gen), weight:1};
		particles.push_back(p);
		weights.push_back(1);		
	}
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
	for (int i = 0; i < num_particles; i++)
	{
		double x_hat = 0;
		double y_hat = 0;
		double theta_hat = 0;

		if (abs(yaw_rate) > 1e-5) 
		{
			theta_hat = particles[i].theta + yaw_rate * delta_t;
            x_hat = particles[i].x + velocity / yaw_rate * (sin(theta_hat) - sin(particles[i].theta));
            y_hat = particles[i].y + velocity / yaw_rate * (cos(particles[i].theta) - cos(theta_hat));            
        } 
		else 
		{   
			theta_hat = particles[i].theta;         
            x_hat = particles[i].x + velocity * delta_t * cos(particles[i].theta);
            y_hat = particles[i].y + velocity * delta_t * sin(particles[i].theta);			
		}

		// Initialize normal distributions centered on predicted values
        normal_distribution<double> ND_x(x_hat, std_pos[0]);
        normal_distribution<double> ND_y(y_hat, std_pos[1]);
        normal_distribution<double> ND_theta(theta_hat, std_pos[2]);

        // Update particle with noisy prediction
        particles[i].x	   = ND_x(gen);
        particles[i].y	   = ND_y(gen);
		particles[i].theta = ND_theta(gen);

	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (auto& obs : observations) 
	{
        double min_dist = numeric_limits<double>::max();

        for (auto& pred_obs : predicted) 
		{
            double d = dist(obs.x, obs.y, pred_obs.x, pred_obs.y);
            if (d < min_dist)
			{
                obs.id	 = pred_obs.id;
                min_dist = d;
            }
        }
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
	 // Gather std values for readability
    double std_x = std_landmark[0];
    double std_y = std_landmark[1];

    for (size_t i = 0; i < num_particles; ++i) {
        // List all landmarks within sensor range
        vector<LandmarkObs> predicted_landmarks;

        for (auto& map_landmark : map_landmarks.landmark_list) {
            
            double d = dist(particles[i].x, particles[i].y, map_landmark.x_f, map_landmark.y_f);
            if (d < sensor_range) {
                LandmarkObs l_pred = {id: map_landmark.id_i, x:map_landmark.x_f, y:map_landmark.y_f};  
                predicted_landmarks.push_back(l_pred);
            }
        }

        // List all observations in map coordinates
        vector<LandmarkObs> observed_landmarks_map_ref;
        for (size_t j = 0; j < observations.size(); ++j) {

            // Convert observation from particle(vehicle) to map coordinate system
            LandmarkObs rototranslated_obs;
            rototranslated_obs.x = cos(particles[i].theta) * observations[j].x - sin(particles[i].theta) * observations[j].y + particles[i].x;
            rototranslated_obs.y = sin(particles[i].theta) * observations[j].x + cos(particles[i].theta) * observations[j].y + particles[i].y;

            observed_landmarks_map_ref.push_back(rototranslated_obs);
        }

        // Get landmark associated ids
        dataAssociation(predicted_landmarks, observed_landmarks_map_ref);

        // Compute particle likelihood 
        double particle_likelihood = 1.0;

        double mu_x, mu_y;
        for (const auto& obs : observed_landmarks_map_ref) {

            // Find related map landmark for distribution centers
            for (auto& land: predicted_landmarks)
                if (obs.id == land.id) {
                    mu_x = land.x;
                    mu_y = land.y;
                    break;
                }

            double norm_factor = 2 * M_PI * std_x * std_y;
            double prob = exp( -( pow(obs.x - mu_x, 2) / (2 * std_x * std_x) + pow(obs.y - mu_y, 2) / (2 * std_y * std_y) ) );

            particle_likelihood *= prob / norm_factor;
        }

        particles[i].weight = particle_likelihood;

    } 

    // Normalize 
    double norm_factor = 0.0;
    for (const auto& particle : particles)
        norm_factor += particle.weight;
    
	int i = 0;    
    for (auto& particle : particles)
	{
		particle.weight /= (norm_factor + numeric_limits<double>::epsilon());
		weights[i++] = particle.weight;
	}
		
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	discrete_distribution<int> custom_distribution(weights.begin(), weights.end());
	vector<Particle> resample_particles;

	for(int i = 0; i < num_particles; i++)
	{
		resample_particles.push_back(particles[custom_distribution(gen)]);
	}
	particles = resample_particles;
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
