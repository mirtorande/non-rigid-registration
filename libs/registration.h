#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <igl/AABB.h>
#include <igl/svd3x3.h>
#include <igl/point_mesh_squared_distance.h>
#include <igl/barycenter.h>
#include <igl/random_dir.h>
//#include <igl/avg_edge_length.h>


typedef Eigen::MatrixXd Matd;
typedef Eigen::MatrixXi Mati;
typedef Eigen::Matrix3d Mat3d;
typedef Eigen::VectorXi Veci;
typedef Eigen::Vector3d Vec3d;
typedef Eigen::RowVectorXi RVeci;
typedef Eigen::RowVectorXd RVecd;
typedef Eigen::RowVector3d RVec3d;


// Funzione per trovare i vicini di ciascun vertice nella mesh
Mati find_vertex_neighbors(const Mati& FB, int num_vertices) {
	// Creazione della matrice dei vicini dei vertici
	Mati vertex_neighbors = Mati::Zero(num_vertices, 10); // Si suppone che ogni vertice abbia al massimo 10 vicini

	// Contatore per tener traccia del numero di vicini di ciascun vertice
	Veci num_neighbors = Veci::Zero(num_vertices);

	// Per ogni faccia nella matrice delle facce
	for (int i = 0; i < FB.rows(); ++i) {
		int v0 = FB(i, 0);
		int v1 = FB(i, 1);
		int v2 = FB(i, 2);

		// Aggiungi gli archi tra i vertici della faccia
		if (num_neighbors[v0] < vertex_neighbors.cols()) {
			vertex_neighbors(v0, num_neighbors[v0]++) = v1;
			vertex_neighbors(v0, num_neighbors[v0]++) = v2;
		}
		if (num_neighbors[v1] < vertex_neighbors.cols()) {
			vertex_neighbors(v1, num_neighbors[v1]++) = v0;
			vertex_neighbors(v1, num_neighbors[v1]++) = v2;
		}
		if (num_neighbors[v2] < vertex_neighbors.cols()) {
			vertex_neighbors(v2, num_neighbors[v2]++) = v0;
			vertex_neighbors(v2, num_neighbors[v2]++) = v1;
		}
	}

	return vertex_neighbors;
}

void computeAdjacentFaceAreas(const Matd& V, const Mati& F, Eigen::VectorXd& vertexAreas)
{
	// Inizializza l'array delle aree dei vertici a zero
	vertexAreas.setZero(V.rows());

	// Per ogni faccia
	for (int i = 0; i < F.rows(); ++i)
	{
		// Calcola il baricentro della faccia
		RVec3d barycenter;
		igl::barycenter(V, F.row(i), barycenter);

		// Per ogni vertice nella faccia
		for (int j = 0; j < F.cols(); ++j)
		{
			int vertexIndex = F(i, j);
			// Calcola l'area del triangolo formato dalla faccia e dal vertice
			RVec3d v0 = V.row(F(i, (j + 1) % F.cols()));
			RVec3d v1 = V.row(F(i, (j + 2) % F.cols()));
			double area = 0.5 * (v0 - barycenter).cross(v1 - barycenter).norm();

			// Aggiungi l'area alla somma delle aree dei vertici
			vertexAreas(vertexIndex) += area;
		}
	}
}

void select_n_random_points(int n, const Matd& V, Matd& VA)
{
	// Seed the random number generator
	srand(static_cast<unsigned>(time(0)));

	// Get the total number of rows in VX
	const int totalRows = V.rows();

	// Ensure numPoints does not exceed the total number of rows
	if (n > totalRows) {
		n = totalRows;
	}

	// Randomly select numPoints row indices
	Veci selectedRows(n); // Initialize and resize selectedRows
	for (int i = 0; i < n; ++i) {
		selectedRows(i) = rand() % totalRows;
	}

	// Create VA with selected rows
	VA.resize(n, V.cols());
	for (int i = 0; i < n; ++i) {
		VA.row(i) = V.row(selectedRows(i));
	}
}

void initialize_icp_correspondences(
	const Matd& VX,
	const Matd& VY, const Mati& FY, const igl::AABB<Matd, 3> Ytree,
	Matd& VA, Matd& VB)
{
	int numPoints = 100;
	const int totalRows = VX.rows();

	if (numPoints > totalRows) {
		numPoints = totalRows;
	}

	select_n_random_points(numPoints, VX, VA);

	Matd squared_distances;
	Mati closest_indexes;
	Matd closest_points;
	Ytree.squared_distance(VY, FY, VA, squared_distances, closest_indexes, VB);
	// Calculate standard deviation of squared distances
	double squared_distances_sum = squared_distances.sum();
	double mean = squared_distances_sum / squared_distances.size();

	// Remove outliers (points with squared distance greater than 2 standard deviations)
	Matd newVA;
	Matd newVB;
	int newVA_size = 0;
	for (int i = 0; i < squared_distances.size(); i++) {
		if (squared_distances(i) < mean) {
			newVA_size++;
		}
	}
	newVA.resize(newVA_size, VA.cols());
	newVB.resize(newVA_size, VB.cols());
	int j = 0;
	for (int i = 0; i < squared_distances.size(); i++) {
		if (squared_distances(i) < mean) {
			newVA.row(j) = VA.row(i);
			newVB.row(j) = VB.row(i);
			j++;
		}
	}
	VA = newVA;
	VB = newVB;
}

void rigid_shape_matching(Matd VA, Matd VB, Mat3d& R, RVec3d& t)
{
	//assert(VA.rows() == VB.rows());
	// Ricavo traslazione
	// Calcolo baricentro
	const RVecd summerVector = RVecd::Constant(VA.rows(), 1.0f / VA.rows());
	RVecd bariA = summerVector * VA;
	RVecd bariB = summerVector * VB;
	// Rotazione: Matrice di somme di esterni tra punti iniziali e punti finali
	Mat3d outerProduct = (VA.rowwise() - bariA).eval().transpose() * (VB.rowwise() - bariB).eval();

	Mat3d U, D;
	Vec3d S;
	igl::svd3x3(outerProduct, U, S, D);
	R = U * D.transpose();
	//cout << "U*S*D\n" << U * S.asDiagonal() * D.transpose() << endl;

	t = bariB - bariA * R;
}

void ransac3(const Matd& VX, const Matd& VY, const Mati& FY, const igl::AABB<Matd, 3> Ytree, Mat3d& R, RVec3d& t)
{
	int max_iterations = 100;
	int accept_threshold = 100;
	double delta = 0.01f;
	int largest_consensus = 0;
	Mat3d best_R;
	RVec3d best_t;

	for (int i = 0; i < max_iterations; i++)
	{
		// Select triplets
		Matd tri_x, tri_y;
		select_n_random_points(3, VX, tri_x);
		select_n_random_points(3, VY, tri_y);

		// Try matching
		rigid_shape_matching(tri_x, tri_y, R, t);

		// Apply transformation
		Matd ransacdVX = VX * R;
		ransacdVX = (ransacdVX.rowwise() + t).eval();

		// Calculate number of valid points

		Matd squared_distances;
		Mati closest_indexes;
		Matd closest_points;
		Ytree.squared_distance(VY, FY, ransacdVX, squared_distances, closest_indexes, closest_points);

		//igl::point_mesh_squared_distance(ransacdVX, VY, FY, squared_distances, closest_indexes, closest_points);

		int valid_candidates = (squared_distances.array() < delta).count();

		// If over threshold, accept
		if (valid_candidates > accept_threshold)
			return;

		// if best result, save
		if (valid_candidates > largest_consensus)
		{
			best_R = R;
			best_t = t;
		}
	}
	R = best_R;
	t = best_t;
}

bool contains_row(const Mati& matrix, const RVeci& row)
{
	for (int i = 0; i < matrix.rows(); ++i)
	{
		if (matrix.row(i) == row)
		{
			return true;
		}
	}
	return false;
}

const auto apply_random_rotation = [](Matd& VA)
	{
		const Mat3d R = Eigen::AngleAxisd(
			2. * igl::PI * (double)rand() / RAND_MAX * 0.3, igl::random_dir()).matrix();
		const RVec3d cen =
			0.5 * (VA.colwise().maxCoeff() + VA.colwise().minCoeff());
		VA = ((VA * R).rowwise() + (cen - cen * R)).eval();
	};

const auto apply_random_translation = [](Matd& VA)
	{
		double translation_limit = 5.0;
		const RVec3d t = RVec3d::Random(VA.cols()) / 10.0f;
		VA = (VA.rowwise() + t).eval();
	};

const auto icp_iteration = [](Matd& VA, const Matd& VB, const Mati& FB, const igl::AABB<Matd, 3> treeB)
	{
		// Mesh surface based ICP step
		Matd cloudA;
		Matd cloudB;
		initialize_icp_correspondences(VA, VB, FB, treeB, cloudA, cloudB);
		// Remove outliers

		Mat3d R;
		RVec3d t;
		rigid_shape_matching(cloudA, cloudB, R, t);
		VA = VA * R;
		VA = (VA.rowwise() + t).eval();
	};

const auto ransac_iteration = [](Matd& VA, const Matd& VB, const Mati& FB, const igl::AABB<Matd, 3> treeB)
	{
		// Ransac step
		Mat3d R;
		RVec3d t;
		ransac3(VA, VB, FB, treeB, R, t);
		VA = VA * R;
		VA = (VA.rowwise() + t).eval();
	};
