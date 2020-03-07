#include <iostream>
#include <Eigen/dense>
#include "file_manage.h"
#include "model_selection.h"
#include "KMeans.h"
using namespace std;
using namespace Eigen;

int main()
{
	MatrixXd features;
	VectorXd labels;

	read_csv("data/iris.csv", features, labels, 150, 5);
	//read_csv("data/seeds.csv", features, labels, 210, 8);

	KMeans km(3);
	km.fit(features, 1, 5);

	double score = evaluate_model(km, features, labels);
	cout << "Silhouette score : " << score << endl;

	return 0;
}