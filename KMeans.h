#pragma once
#include <iostream>
#include <vector>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;

/*
	K means clustering

		- This code is written to deeply understand the kmeans algorithm.

	1. Variable description

		- K: # of clusters
		- groups: A 2d vector containing the original dataset's index of each group's point.
		- centers: A row vector containing the center of each cluster.

	2. Function description

		- fit(const MatrixXd& df, int init, int n_init):

			This function takes three arguments and df is a dataset used to form clusters. init
			is an option to initialize centers. If it is zero, the function randomly select
			'K' number of rows from dataset and assign them as initail centers. If it is one,
			the function initialize centers using kmeans++ algorithm. This algorithm selects
			the furthest point from the existing centers. In gerneral, this is known to prevent
			a convergence to the local minimum, guaranteeing slightly better performance.

			The thrid argument represents the number of initializations. If it is one, the default
			value, the function forms one cluster set using only one center set. In other words,
			it starts at only one point(an initial set of centers) and end when it reaches
			the minimum(the optimal centers). If it is greater than one, the funtion forms multiple
			cluseter sets using mutiple center set, and selects the best one among them.

		- predict(const MatrixXd& df):

			This function simply assigns each point(each row of datset) to the closest cluster.
			It also calculate the sillhouette coefficient to evaluate the clustring result.
			The coefficient has a value between -1 to 1. The closer to 1, the better the clusters.
*/

class KMeans
{
private:
	int K;
	RowVectorXd* centers;
	vector<vector<int>> clusters;
public:
	KMeans(int n_clusters);
	~KMeans();
	void fit(const MatrixXd& X, int init, int n_init = 1);
	VectorXd predict(const MatrixXd& X);

	template<class T>
	friend double evaluate_model(const T& model, const MatrixXd& features, const VectorXd& labels);
};

namespace kms
{
	void single_fit(const MatrixXd& X, RowVectorXd*& centers, vector<vector<int>>& clusters,
		int K, int init, int seed = 0);

	void rand_init_center(const MatrixXd& X, RowVectorXd*& centers, int K, int seed);

	int unique_random(const vector<int>& unique, int range);

	void kmpp_init_center(const MatrixXd& X, RowVectorXd*& centers, int K, int seed);

	int nearest_center(const RowVectorXd& point, const RowVectorXd* centers, int K);

	double euclidean_norm(const RowVectorXd& p1, const RowVectorXd& p2);

	vector<vector<int>> to_clusters(const MatrixXd& X, const RowVectorXd* centers, int K);

	bool clusters_are_empty(const vector<vector<int>>& clusters);

	void update_centers(const MatrixXd& X, RowVectorXd*& centers, const vector<vector<int>>& clusters);

	void ensemble_fit(const MatrixXd& X, RowVectorXd*& centers, vector<vector<int>>& clusters,
		int K, int init, int n_init);

	double squared_error(const MatrixXd& X, const RowVectorXd* centers, const vector<vector<int>>& clusters);
}

KMeans::KMeans(int n_clusters) : K(0), centers(nullptr)
{
	if (n_clusters <= 0)
		cout << "Error(KMeans::KMeans(int)): Invalid clusters." << endl;
	else
	{
		K = n_clusters;
		centers = new RowVectorXd[K];
	}
}

KMeans::~KMeans() { delete[] centers; }

void KMeans::fit(const MatrixXd& X, int init, int n_init)
{
	using namespace kms;

	if (X.rows() == 0 || X.cols() == 0)
	{
		cout << "Error(mean(const MatrixXd&, RowVectorXd&, int)): Empty dataset." << endl;
		return;
	}

	if (n_init == 1)
		single_fit(X, centers, clusters, K, init);
	else if (n_init > 1)
		ensemble_fit(X, centers, clusters, K, init, n_init);
	else
		cout << "Error(mean(const MatrixXd&, RowVectorXd&, int)): Invalide initialization." << endl;
}

void kms::single_fit(const MatrixXd& X, RowVectorXd*& centers, vector<vector<int>>& clusters,
	int K, int init, int seed)
{
	if (init != 0 && init != 1)
	{
		cout << "Error(kms::single_fit(const MatrixXd&, RowVectorXd*&, int, int)): " <<
			"Invalid initialization." << endl;
		return;
	}
	else if (init == 0)
		rand_init_center(X, centers, K, seed);
	else
		kmpp_init_center(X, centers, K, seed);

	while (1)
	{
		vector<vector<int>> temp_clustsers = to_clusters(X, centers, K);

		if (temp_clustsers == clusters)
			break;

		update_centers(X, centers, temp_clustsers);
		clusters = temp_clustsers;
	}
}

void kms::rand_init_center(const MatrixXd& X, RowVectorXd*& centers, int K, int seed)
{
	srand((unsigned)time(NULL) + seed);

	vector<int> unique;
	for (int i = 0; i < K; i++)
	{
		int idx = unique_random(unique, (int)X.rows());
		centers[i] = X.row(idx);
		unique.push_back(idx);
	}
}

int kms::unique_random(const vector<int>& unique, int range)
{
	bool isOverlap;
	int num;
	do
	{
		num = rand() % range;
		isOverlap = false;
		for (int i = 0; i < unique.size(); i++)
			if (unique[i] == num)
			{
				isOverlap = true;
				break;
			}
	} while (isOverlap);
	return num;
}

void kms::kmpp_init_center(const MatrixXd& X, RowVectorXd*& centers, int K, int seed)
{
	srand((unsigned)time(NULL) + seed);

	centers[0] = X.row(rand() % X.rows());

	for (int i = 1; i < K; i++)
	{
		vector<double> norms;
		for (int j = 0; j < X.rows(); j++)
		{
			int idx = nearest_center(X.row(j), centers, i);
			norms.push_back(euclidean_norm(X.row(j), centers[idx]));
		}
		__int64 max = std::distance(norms.begin(), std::max_element(norms.begin(), norms.end()));
		centers[i] = X.row(max);
	}
}

int kms::nearest_center(const RowVectorXd& point, const RowVectorXd* centers, int K)
{
	int min_idx = 0;
	double min = euclidean_norm(point, centers[0]);
	for (int i = 1; i < K; i++)
	{
		double norm = euclidean_norm(point, centers[i]);
		if (min > norm)
		{
			min = norm;
			min_idx = i;
		}
	}
	return min_idx;
}

double kms::euclidean_norm(const RowVectorXd& p1, const RowVectorXd& p2)
{
	if (p1.size() != p2.size())
	{
		cout << "Error(KMeans::euclidean_norm(cosnt RowVectorXd&, const RowVectorXd&)): " <<
			"Vectors are not compatible." << endl;
		return 0.0;
	}

	double sum = 0;
	for (int i = 0; i < p1.size(); i++)
		sum += std::pow((p1[i] - p2[i]), 2);

	return std::sqrt(sum);
}

vector<vector<int>> kms::to_clusters(const MatrixXd& X, const RowVectorXd* centers, int K)
{
	vector<vector<int>> clusters(K, vector<int>());
	for (int i = 0; i < X.rows(); i++)
	{
		int idx = nearest_center(X.row(i), centers, K);
		clusters[idx].push_back(i);
	}

	if (clusters_are_empty(clusters))
	{
		cout << "Error(to_clusters(const MatrixXd&, const RowVectorXd*, vector<vector<int>>&, int)): " <<
			"Cannot make " << K << " number of clusters." << endl;

		exit(1);
	}

	return clusters;
}

bool kms::clusters_are_empty(const vector<vector<int>>& clusters)
{
	for (int i = 0; i < clusters.size(); i++)
		if (clusters[i].size() == 0)
			return true;
	return false;
}

void kms::update_centers(const MatrixXd& X, RowVectorXd*& centers, const vector<vector<int>>& clusters)
{
	for (int i = 0; i < clusters.size(); i++)
	{
		RowVectorXd sum;
		for (int j = 0; j < clusters[i].size(); j++)
		{
			if (j == 0)
				sum = X.row(clusters[i][j]);
			else
				sum += X.row(clusters[i][j]);
		}
		centers[i] = sum / (double)clusters[i].size();
	}
}

void kms::ensemble_fit(const MatrixXd& X, RowVectorXd*& centers, vector<vector<int>>& clusters,
	int K, int init, int n_init)
{
	vector<RowVectorXd*> container;
	vector<double> errors;

	for (int i = 0; i < n_init; i++)
	{
		RowVectorXd* temp_centers = new RowVectorXd[K];
		single_fit(X, temp_centers, clusters, K, init, i + 1);
		errors.push_back(squared_error(X, temp_centers, clusters));
		container.push_back(temp_centers);
	}

	__int64 min = std::distance(errors.begin(), std::min_element(errors.begin(), errors.end()));

	for (int i = 0; i < K; i++)
		centers[i] = container[min][i];

	for (auto*& temp_centers : container)
		delete[] temp_centers;
}

double kms::squared_error(const MatrixXd& X, const RowVectorXd* centers, const vector<vector<int>>& clusters)
{
	double error = 0;
	for (int i = 0; i < clusters.size(); i++)
		for (const int& idx : clusters[i])
			error += std::pow(euclidean_norm(X.row(idx), centers[i]), 2);
	return error;
}