#include <igl/readOBJ.h>
#include <igl/opengl/glfw/Viewer.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <igl/random_dir.h>
#include <igl/avg_edge_length.h>
#include <igl/svd3x3.h>
#include <igl/AABB.h>
#include <igl/per_face_normals.h>
#include <igl/iterative_closest_point.h>
#include <igl/point_mesh_squared_distance.h>

using namespace Eigen;

const std::string MODEL_FILE_PATH = "../resources/";

void select_n_random_points(int n, const MatrixXd& V, MatrixXd& VA);

void initialize_icp_correspondences(
	const MatrixXd& VX,
	const MatrixXd& VY, const MatrixXi& FY, const igl::AABB<MatrixXd, 3> Ytree,
	MatrixXd& VA, MatrixXd& VB);

void rigid_shape_matching(
	MatrixXd VA,	// Mesh A
	MatrixXd VB,	// Mesh B
	Matrix3d& R, RowVector3d& t	// Result
);

void ransac3(const MatrixXd& VX, const MatrixXd& VY, const MatrixXi& FY, const igl::AABB<MatrixXd, 3> Ytree, Matrix3d& R, RowVector3d& t);


int main(int argc, char* argv[])
{
	MatrixXd OVX, VA, VB;
	MatrixXi FA, FB;
	igl::readOBJ(MODEL_FILE_PATH + "rabbit.obj", VA, FA);
	igl::readOBJ(MODEL_FILE_PATH + "rabbit-shard2.obj", VB, FB);

	// Initialize viewer
	igl::opengl::glfw::Viewer v;

	igl::AABB<MatrixXd, 3> treeB;
	treeB.init(VB, FB);
	MatrixXd NY;
	igl::per_face_normals(VB, FB, NY);

	const auto apply_random_rotation = [&]()
		{
			const Matrix3d R = AngleAxisd(
				2. * igl::PI * (double)rand() / RAND_MAX * 0.3, igl::random_dir()).matrix();
			const RowVector3d cen =
				0.5 * (VA.colwise().maxCoeff() + VA.colwise().minCoeff());
			VA = ((VA * R).rowwise() + (cen - cen * R)).eval();
		};

	const auto apply_random_translation = [&]()
		{
			double translation_limit = 5.0;
			const RowVector3d t = RowVector3d::Random(VA.cols()) / 10.0f;
			VA = (VA.rowwise() + t).eval();
		};

	const auto single_iteration = [&]()
		{
			////////////////////////////////////////////////////////////////////////
			// Perform single iteration of ICP method
			////////////////////////////////////////////////////////////////////////
			Matrix3d R;
			RowVector3d t;
			//igl::iterative_closest_point(VX, FX, VY, FY, Ytree, NY, 1000, 1, R, t);
			rigid_shape_matching(VA, VB, R, t);
			VA = VA * R;
			VA = (VA.rowwise() + t).eval();
			v.data().set_mesh(VA, FA);
			v.data().compute_normals();
		};

	v.callback_pre_draw = [&](igl::opengl::glfw::Viewer&)->bool
		{
			if (v.core().is_animating)
			{
				single_iteration();
			}
			return false;
		};
	v.callback_key_pressed =
		[&](igl::opengl::glfw::Viewer&, unsigned char key, int)->bool
		{
			switch (key)
			{
			case ' ':
			{
				v.core().is_animating = false;
				single_iteration();
				return true;
			}
			case 'R':
			case 'r':
			{
				// Random rigid transformation
				apply_random_rotation();
				v.data().set_mesh(VA, FA);
				v.data().compute_normals();
				return true;
				break;
			}
			case 'T':
			case 't':
			{
				// Random rigid transformation
				apply_random_translation();
				v.data().set_mesh(VA, FA);
				v.data().compute_normals();
				return true;
				break;
			}
			case 'Y':
			case 'y':
			{
				// Mesh surface based ICP step
				MatrixXd cloudA;
				MatrixXd cloudB;
				initialize_icp_correspondences(VA, VB, FB, treeB, cloudA, cloudB);
				// Remove outliers

				Matrix3d R;
				RowVector3d t;
				rigid_shape_matching(cloudA, cloudB, R, t);
				VA = VA * R;
				VA = (VA.rowwise() + t).eval();
				v.data().set_mesh(VA, FA);
				v.data().compute_normals();
				return true;
				break;
			}
			case 'G':
			case 'g':
			{
				// Ransac step
				Matrix3d R;
				RowVector3d t;
				ransac3(VA, VB, FB, treeB, R, t);
				VA = VA * R;
				VA = (VA.rowwise() + t).eval();
				v.data().set_mesh(VA, FA);
				v.data().compute_normals();
				return true;
				break;
			}
			}
		};

	v.data().set_mesh(VB, FB);
	v.data().set_colors(RowVector3d(1, 0, 1));
	v.data().show_lines = false;
	v.append_mesh();
	v.data().set_mesh(VA, FA);
	v.data().show_lines = false;
	v.launch();
}

void select_n_random_points(int n, const MatrixXd& V, MatrixXd& VA)
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
	VectorXi selectedRows(n); // Initialize and resize selectedRows
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
	const MatrixXd& VX,
	const MatrixXd& VY, const MatrixXi& FY, const igl::AABB<MatrixXd, 3> Ytree,
	MatrixXd& VA, MatrixXd& VB)
{
	int numPoints = 100;
	const int totalRows = VX.rows();

	if (numPoints > totalRows) {
		numPoints = totalRows;
	}

	select_n_random_points(numPoints, VX, VA);

	MatrixXd squared_distances;
	MatrixXi closest_indexes;
	MatrixXd closest_points;
	Ytree.squared_distance(VY, FY, VA, squared_distances, closest_indexes, VB);
	// Calculate standard deviation of squared distances
	double squared_distances_sum = squared_distances.sum();
	double mean = squared_distances_sum / squared_distances.size();

	// Remove outliers (points with squared distance greater than 2 standard deviations)
	MatrixXd newVA;
	MatrixXd newVB;
	int newVA_size = 0;
	for (int i = 0; i < squared_distances.size(); i++) {
		if (squared_distances(i) < mean ) {
			newVA_size++;
		}
	}
	newVA.resize(newVA_size, VA.cols());
	newVB.resize(newVA_size, VB.cols());
	int j = 0;
	for (int i = 0; i < squared_distances.size(); i++) {
		if (squared_distances(i) < mean ) {
			newVA.row(j) = VA.row(i);
			newVB.row(j) = VB.row(i);
			j++;
		}
	}
	VA = newVA;
	VB = newVB;
}

void rigid_shape_matching(MatrixXd VA, MatrixXd VB, Matrix3d& R, RowVector3d& t)
{
	//assert(VA.rows() == VB.rows());
	// Ricavo traslazione
	// Calcolo baricentro
	const RowVectorXd summerVector = RowVectorXd::Constant(VA.rows(), 1.0f / VA.rows());
	RowVectorXd bariA = summerVector * VA;
	RowVectorXd bariB = summerVector * VB;
	// Rotazione: Matrice di somme di esterni tra punti iniziali e punti finali
	Matrix3d outerProduct = (VA.rowwise() - bariA).eval().transpose() * (VB.rowwise() - bariB).eval();

	Matrix3d U, D;
	Vector3d S;
	igl::svd3x3(outerProduct, U, S, D);
	R = U * D.transpose();
	//cout << "U*S*D\n" << U * S.asDiagonal() * D.transpose() << endl;

	t = bariB - bariA * R;
}

void ransac3(const MatrixXd& VX, const MatrixXd& VY, const MatrixXi& FY, const igl::AABB<MatrixXd, 3> Ytree, Matrix3d& R, RowVector3d& t)
{
	int max_iterations = 100;
	int accept_threshold = 100;
	double delta = 0.01f;
	int largest_consensus = 0;
	Matrix3d best_R;
	RowVector3d best_t;

	for (int i = 0; i < max_iterations; i++)
	{
		// Select triplets
		MatrixXd tri_x, tri_y;
		select_n_random_points(3, VX, tri_x);
		select_n_random_points(3, VY, tri_y);

		// Try matching
		rigid_shape_matching(tri_x, tri_y, R, t);

		// Apply transformation
		MatrixXd ransacdVX = VX * R;
		ransacdVX = (ransacdVX.rowwise() + t).eval();

		// Calculate number of valid points

		MatrixXd squared_distances;
		MatrixXi closest_indexes;
		MatrixXd closest_points;
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