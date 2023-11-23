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

using namespace std;
using namespace Eigen;

const string MODEL_FILE_PATH = "../resources/";

RowVector3d closest_point_on_triangle(RowVector3d point, RowVector3d a, RowVector3d b, RowVector3d c);

RowVector3d find_closest_point_on_mesh(RowVector3d point, const MatrixXd& V, const MatrixXi& F);

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
	MatrixXd OVX, VX, VY;
	MatrixXi FX, FY;
	// Load a mesh in OFF format
	igl::readOBJ(MODEL_FILE_PATH + "rabbit.obj", VX, FX);
	igl::readOBJ(MODEL_FILE_PATH + "rabbit.obj", VY, FY);

	// Initialize viewer
	igl::opengl::glfw::Viewer v;

	const double bbd = (VY.colwise().maxCoeff() - VY.colwise().minCoeff()).norm();
	{
		// sprinkle a noise so that we can see z-fighting when the match is perfect.
		const double h = igl::avg_edge_length(VY, FY);
		OVX = VY + 1e-2 * h * MatrixXd::Random(VY.rows(), VY.cols());
	}

	VX = OVX;

	igl::AABB<MatrixXd, 3> Ytree;
	Ytree.init(VY, FY);
	MatrixXd NY;
	igl::per_face_normals(VY, FY, NY);

	const auto apply_random_rotation = [&]()
		{
			const Matrix3d R = AngleAxisd(
				2. * igl::PI * (double)rand() / RAND_MAX * 0.3, igl::random_dir()).matrix();
			const RowVector3d cen =
				0.5 * (VY.colwise().maxCoeff() + VY.colwise().minCoeff());
			VX = ((OVX * R).rowwise() + (cen - cen * R)).eval();
		};

	const auto apply_random_translation = [&]()
		{
			double translation_limit = 5.0;
			const RowVector3d t = RowVector3d::Random(VY.cols()) / 10.0f;
			VX = (VX.rowwise() + t).eval();
		};

	const auto single_iteration = [&]()
		{
			////////////////////////////////////////////////////////////////////////
			// Perform single iteration of ICP method
			////////////////////////////////////////////////////////////////////////
			Matrix3d R;
			RowVector3d t;
			//igl::iterative_closest_point(VX, FX, VY, FY, Ytree, NY, 1000, 1, R, t);
			rigid_shape_matching(VX, VY, R, t);
			VX = VX * R;
			VX = (VX.rowwise() + t).eval();
			v.data().set_mesh(VX, FX);
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
				v.data().set_mesh(VX, FX);
				v.data().compute_normals();
				return true;
				break;
			}
			case 'T':
			case 't':
			{
				// Random rigid transformation
				apply_random_translation();
				v.data().set_mesh(VX, FX);
				v.data().compute_normals();
				return true;
				break;
			}
			case 'Y':
			case 'y':
			{
				// Mesh surface based ICP step
				MatrixXd VA;
				MatrixXd VB;
				initialize_icp_correspondences(VX, VY, FY, Ytree, VA, VB);
				Matrix3d R;
				RowVector3d t;
				rigid_shape_matching(VA, VB, R, t);
				VX = VX * R;
				VX = (VX.rowwise() + t).eval();
				v.data().set_mesh(VX, FX);
				v.data().compute_normals();
				return true;
				break;
			}
			case 'G':
			case 'g':
			{
				// Ransac step
				MatrixXd VA;
				MatrixXd VB;
				Matrix3d R;
				RowVector3d t;
				ransac3(VX, VY, FY, Ytree, R, t);
				VX = VX * R;
				VX = (VX.rowwise() + t).eval();
				v.data().set_mesh(VX, FX);
				v.data().compute_normals();
				return true;
				break;
			}
			case 'A':
			case 'a':
			{
				RowVector3d point = RowVector3d::Zero(3);
				RowVector3d closestPoint = find_closest_point_on_mesh(point, VX, FX);
				cout << closestPoint << endl;
				return true;
				break;
			}
			return false;
			};
		};

	v.data().set_mesh(VY, FY);
	v.data().set_colors(RowVector3d(1, 0, 1));
	v.data().show_lines = false;
	v.append_mesh();
	v.data().set_mesh(VX, FX);
	v.data().show_lines = false;
	v.launch();
}

RowVector3d closest_point_on_triangle(RowVector3d point, RowVector3d a, RowVector3d b, RowVector3d c) {
	// Compute the edges of the triangle
	RowVector3d e0 = b - a;
	RowVector3d e1 = c - a;

	// Calculate the normal vector of the triangle
	RowVector3d n = e0.cross(e1);
	n.normalize(); // Normalize it once

	// Calculate the vector from 'a' to the query point
	RowVector3d ap = point - a;

	// Calculate the distance along the triangle normal
	double t = ap.dot(n);

	// Calculate the projected point on the plane
	RowVector3d pointOnPlane = point - t * n;

	// Check if the point is inside the triangle
	RowVector3d v0 = b - a;
	RowVector3d v1 = c - a;
	RowVector3d v2 = pointOnPlane - a;

	double dot00 = v0.dot(v0);
	double dot01 = v0.dot(v1);
	double dot02 = v0.dot(v2);
	double dot11 = v1.dot(v1);
	double dot12 = v1.dot(v2);

	double denom = dot00 * dot11 - dot01 * dot01;
	if (denom == 0.0) {
		// Triangle degenerates into a line
		return a; // Return 'a' as the closest point
	}

	double u = (dot11 * dot02 - dot01 * dot12) / denom;
	double v = (dot00 * dot12 - dot01 * dot02) / denom;

	if (u >= 0.0 && v >= 0.0 && u + v <= 1.0) {
		return pointOnPlane; // The point is inside the triangle
	}

	// Find the closest point on each edge of the triangle
	double minDistance = (pointOnPlane - a).norm();
	RowVector3d closest = a;

	double distance = (pointOnPlane - b).norm();
	if (distance < minDistance) {
		minDistance = distance;
		closest = b;
	}

	distance = (pointOnPlane - c).norm();
	if (distance < minDistance) {
		closest = c;
	}

	return closest;
}


RowVector3d find_closest_point_on_mesh(RowVector3d point, const MatrixXd& V, const MatrixXi& F)
{
	double minDistance = 10e12;
	RowVector3d closestPoint;

	for (int triangleIndex = 0; triangleIndex < F.rows(); ++triangleIndex) {
		// Extract the vertices of the triangle
		RowVector3d v0 = V.row(F(triangleIndex, 0));
		RowVector3d v1 = V.row(F(triangleIndex, 1));
		RowVector3d v2 = V.row(F(triangleIndex, 2));

		// Perform your operation on the vertices of the triangle
		RowVector3d closestPointIteration = closest_point_on_triangle(point, v0, v1, v2);
		double distance = (closestPointIteration - point).squaredNorm();

		if (distance < minDistance) {
			minDistance = distance;
			closestPoint = closestPointIteration;
		}
	}
	return closestPoint;
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
	/*
	// Create VB with corresponding values
	VB.resize(numPoints, VX.cols());
	for (int i = 0; i < numPoints; ++i) {
		RowVector3d point(VA.row(i));
		RowVector3d closestPoint = find_closest_point_on_mesh(point, VY, FY);
		VB.row(i) = closestPoint;
	}*/
}

void rigid_shape_matching(MatrixXd VA, MatrixXd VB, Matrix3d& R, RowVector3d& t)
{
	assert(VA.rows() == VB.rows());
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

/*RowVector3d closest_point_on_triangle(RowVector3d point, RowVector3d a, RowVector3d b, RowVector3d c)
{
	RowVector3d e0 = b - a;
	RowVector3d e1 = c - a;
	RowVector3d n = e0.cross(e1).normalized();
	RowVector3d o = a;

	Matrix4d triangleSpace;

	// Find the matrix to convert coordinates to triangle space
	triangleSpace.col(0) << e0.transpose(), 0;
	triangleSpace.col(1) << e1.transpose(), 0;
	triangleSpace.col(2) << n.transpose(), 0;
	triangleSpace.col(3) << o.transpose(), 1;

	Matrix4d toTriangleSpace = triangleSpace.inverse();

	// Convert coordinates
	Vector4d pointHomo;
	pointHomo << point.transpose(), 1.0f;

	Vector4d pointOnPlaneHomo = toTriangleSpace * pointHomo;
	RowVector3d pointOnPlane = pointOnPlaneHomo.transpose().head(3);

	// Check if the point is projected inside the triangle
	double sum = pointOnPlane.x() + pointOnPlane.y();
	if (sum >= 0.0f && sum <= 1.0f) {
		return pointOnPlane;
	}

	// Find the closest point on each edge of the triangle
	RowVector3d closest = a;
	double minDistance = (pointOnPlane - a).norm();

	double distance;
	RowVector3d closestOnEdge;

	distance = (pointOnPlane - b).norm();
	if (distance < minDistance) {
		minDistance = distance;
		closest = b;
	}

	distance = (pointOnPlane - c).norm();
	if (distance < minDistance) {
		minDistance = distance;
		closest = c;
	}

	// The 'closest' variable now contains the closest point on the triangle
	return closest;
}*/

/*
Vector3d triangle_normal(Vector3d a, Vector3d b, Vector3d c)
{
	return ((c - a).cross(b - a)).normalized();
}

Vector3d project_point_on_plane(Vector3d point, Vector3d planeNorm, Vector3d planePoint)
{
	double k = ((point - planePoint).dot(planeNorm) / planeNorm.dot(planeNorm));
	return point + k * planeNorm;
}

bool isPointInTriangle(
	const RowVector3d& point,
	const RowVector3d& vertex0,
	const RowVector3d& vertex1,
	const RowVector3d& vertex2)
{

	// Calculate the vectors from the point to each vertex of the triangle
	RowVector3d edge0 = vertex1 - vertex0;
	RowVector3d edge1 = vertex2 - vertex0;
	RowVector3d edge2 = point - vertex0;

	// Calculate the dot products of these vectors
	double dot00 = edge0.dot(edge0);
	double dot01 = edge0.dot(edge1);
	double dot02 = edge0.dot(edge2);
	double dot11 = edge1.dot(edge1);
	double dot12 = edge1.dot(edge2);

	// Calculate barycentric coordinates
	double invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01);
	double u = (dot11 * dot02 - dot01 * dot12) * invDenom;
	double v = (dot00 * dot12 - dot01 * dot02) * invDenom;

	// Check if the point is inside the triangle
	return (u >= 0.0) && (v >= 0.0) && (u + v <= 1.0);
}*/
/*
RowVector3d find_closest_triangle_to_point(const RowVector3d& point, const MatrixXd& V, const MatrixXi& F, const igl::AABB<MatrixXd, 3>& tree)
{
	igl::AABB<MatrixXd, 3> tree;
	tree.init(V, F);


	double minDistance = 1e12;  // Initialize to a large value
	int closestTriangleIndex = -1;

	// Query the tree for candidate triangles
	VectorXi candidate_triangles;
	tree.squared_distance(V, F, point, candidate_triangles);

	for (int i = 0; i < candidate_triangles.size(); ++i) {
		int triangleIndex = candidate_triangles(i);
		RowVector3d v0 = V.row(F(triangleIndex, 0));
		RowVector3d v1 = V.row(F(triangleIndex, 1));
		RowVector3d v2 = V.row(F(triangleIndex, 2));

		// Check distance between point and the current triangle
		double distance = squared_distance_point_to_triangle(point, v0, v1, v2);

		if (distance < minDistance) {
			minDistance = distance;
			closestTriangleIndex = triangleIndex;
		}
	}

	// You now have the index of the closest triangle or can return more information about it.
	return closestTriangleIndex;
}*/