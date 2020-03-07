# K means clustering

### This code is written to deeply understand the kmeans algorithm. 

## 1. Variable description

- K: the number of clusters
- groups: A 2d vector containing the original dataset's index of each group's point.
- centers: A row vector containing the center of each cluster.

## 2. Function description

- fit(const MatrixXd& df, int init, int n_init):

	This function takes three arguments and df is a dataset used to form clusters. init is an option to initialize centers. If it is zero, the function randomly select 'K' number of rows from dataset and assign them as initail centers. If it is one, the function initialize centers using kmeans++ algorithm. This algorithm selects the furthest point from the existing centers. In gerneral, this is known to prevent a convergence to the local minimum, guaranteeing slightly better performance. 
			
	The thrid argument represents the number of initializations. If it is one, the default value, the function forms one cluster set using only one center set. In other words, it starts at only one point(an initial set of centers) and end when it reaches the minimum(the optimal centers). If it is greater than one, the funtion forms multiple cluseter sets using mutiple center set, and selects the best one among them. 

- predict(const MatrixXd& df):

	This function simply assigns each point(each row of datset) to the closest cluster. It also calculate the sillhouette coefficient to evaluate the clustring result. The coefficient has a value between -1 to 1. The closer to 1, the better the clusters.

## 3. Usage

- There are two datasets to test this class.

~~~cpp

	MatrixXd df;
	VectorXd clsVec;

	// iris dataset
	read_csv("iris.csv", df, clsVec, 150, 5);

	// seeds dataset
	//read_csv("seeds.csv", df, clsVec, 210, 8);

	KMeans km(3);

	// random init, 1 cluster set
	km.fit(df, 0);

	// random init, 5 sets of clusters
	//km.fit(df, 0, 5);

	// kmeans++ init, 1 cluster set
	//km.fit(df, 1);

	// kmeans++ init, 5 sets of clusters
	//km.fit(df, 1, 5);

	VectorXd labels = km.predict(df);
~~~