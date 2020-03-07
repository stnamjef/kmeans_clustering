#pragma once
#include <iostream>
#include <vector>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;

// function declaration

template<class T>
double evaluate_model(const T& model, const MatrixXd& features, const VectorXd& labels);

double silhouette_score(const MatrixXd& X, const RowVectorXd* centers, const vector<vector<int>>& clusters, int K);

double mean_distance(const RowVectorXd& point, const MatrixXd& X, const vector<int>& clusters);

double euclidean_norm(const RowVectorXd& p1, const RowVectorXd& p2);

// function implementation

template<class T>
double evaluate_model(const T& model, const MatrixXd& features, const VectorXd& labels)
{
	return silhouette_score(features, model.centers, model.clusters, model.K);
}

double silhouette_score(const MatrixXd& X, const RowVectorXd* centers, const vector<vector<int>>& clusters, int K)
{
	double s = 0;
	for (int i = 0; i < clusters.size(); i++)
		for (const auto& idx : clusters[i])
		{
			double a = -1, b = -1;
			for (int j = 0; j < K; j++)
			{
				double temp = mean_distance(X.row(idx), X, clusters[j]);

				if (i == j)
					a = temp;
				else if ((b == -1) || (b > temp))
					b = temp;
			}
			s += (b - a) / std::max(a, b);
		}
	return s / X.rows();
}

double mean_distance(const RowVectorXd& point, const MatrixXd& X, const vector<int>& clusters)
{
	double sum = 0;
	for (const auto& idx : clusters)
		sum += euclidean_norm(point, X.row(idx));
	return sum / clusters.size();
}

double euclidean_norm(const RowVectorXd& p1, const RowVectorXd& p2)
{
	if (p1.size() != p2.size())
	{
		cout << "Error(euclidean_norm(cosnt RowVectorXd&, const RowVectorXd&)): " <<
			"Vectors are not compatible." << endl;
		return 0.0;
	}

	double sum = 0;
	for (int i = 0; i < p1.size(); i++)
		sum += std::pow((p1[i] - p2[i]), 2);

	return std::sqrt(sum);
}