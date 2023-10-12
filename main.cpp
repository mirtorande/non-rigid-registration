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

using namespace std;
using namespace Eigen;

const string MODEL_FILE_PATH = "../../resources/";

Vector3d triangle_normal(Vector3d a, Vector3d b, Vector3d c);

Vector3d project_point_on_plane(Vector3d point, Vector3d planeNorm, Vector3d planePoint);

bool isPointInTriangle(
	const Eigen::Vector2d& point,
	const Eigen::Vector2d& vertex0,
	const Eigen::Vector2d& vertex1,
	const Eigen::Vector2d& vertex2);

RowVector3d closest_point_to_triangle(RowVector3d point, RowVector3d a, RowVector3d b, RowVector3d c);

void applyOperationToTriangle(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, int triangleIndex);

void rigid_shape_matching(
	MatrixXd VA,	// Mesh A
	MatrixXd VB,	// Mesh B
	Matrix3d& R, RowVector3d& t	// Result
);


int main(int argc, char *argv[])
{
	Eigen::MatrixXd OVX, VX, VY;
	Eigen::MatrixXi FX, FY;
	// Load a mesh in OFF format
	igl::readOBJ(MODEL_FILE_PATH + "golflin.obj", VX, FX);
	igl::readOBJ(MODEL_FILE_PATH + "golflin.obj", VY, FY);
	igl::opengl::glfw::Viewer v;

	const double bbd = (VY.colwise().maxCoeff() - VY.colwise().minCoeff()).norm();
	{
		// sprinkle a noise so that we can see z-fighting when the match is perfect.
		const double h = igl::avg_edge_length(VY, FY);
		OVX = VY + 1e-2 * h * Eigen::MatrixXd::Random(VY.rows(), VY.cols());
	}

	VX = OVX;

	igl::AABB<Eigen::MatrixXd, 3> Ytree;
	Ytree.init(VY, FY);
	Eigen::MatrixXd NY;
	igl::per_face_normals(VY, FY, NY);

	const auto apply_random_rotation = [&]()
	{
		const Eigen::Matrix3d R = Eigen::AngleAxisd(
			2. * igl::PI * (double)rand() / RAND_MAX * 0.3, igl::random_dir()).matrix();
		const Eigen::RowVector3d cen =
			0.5 * (VY.colwise().maxCoeff() + VY.colwise().minCoeff());
		VX = ((OVX * R).rowwise() + (cen - cen * R)).eval();
	};

	const auto apply_random_translation = [&]()
	{
		const Eigen::RowVector3d t = Eigen::RowVector3d::Random(VY.cols());
		VX = (VX.rowwise() + t).eval();
	};

	const auto single_iteration = [&]()
	{
		////////////////////////////////////////////////////////////////////////
		// Perform single iteration of ICP method
		////////////////////////////////////////////////////////////////////////
		Eigen::Matrix3d R;
		Eigen::RowVector3d t;
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
			case 'A':
			case 'a':
			{
				for (int triangleIndex = 0; triangleIndex < FX.rows(); ++triangleIndex) {
					applyOperationToTriangle(VX, FX, triangleIndex);
				}
				break;
			}
			return false;
			};
		};

	v.data().set_mesh(VY, FY);
	v.data().set_colors(Eigen::RowVector3d(1, 0, 1));
	v.data().show_lines = false;
	v.append_mesh();
	v.data().set_mesh(VX, FX);
	v.data().show_lines = false;
	v.launch();
}

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
}

RowVector3d closest_point_to_triangle(RowVector3d point, RowVector3d a, RowVector3d b, RowVector3d c)
{
	RowVector3d planeNormal = triangle_normal(a, b, c);
	RowVector3d projectedPoint = project_point_on_plane(point, planeNormal, a);
	// Case 1: point inside triangle
	if (isPointInTriangle(projectedPoint, a, b, c))
	{
		cout << "In" << a << b << c << endl;
		return point;
	}
	// Case 2: point outside triangle
	// Find nearest point on segment
}

void applyOperationToTriangle(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, int triangleIndex)
{
	// Extract the vertices of the triangle
	RowVector3d v0 = V.row(F(triangleIndex, 0));
	RowVector3d v1 = V.row(F(triangleIndex, 1));
	RowVector3d v2 = V.row(F(triangleIndex, 2));

	// Perform your operation on the vertices of the triangle
	/* Example: Calculate and print the area of the triangle
	Eigen::RowVector3d edge1 = v1 - v0;
	Eigen::RowVector3d edge2 = v2 - v0;
	double area = 0.5 * (edge1.cross(edge2)).norm();
	std::cout << "Triangle " << triangleIndex << " Area: " << area << std::endl;*/
	RowVector3d point = RowVector3d::Zero(3);
	closest_point_to_triangle(point, v0, v1, v2);
}


void rigid_shape_matching(MatrixXd VA, MatrixXd VB, Matrix3d& R, RowVector3d& t)
{
	assert(VA.rows() == VB.rows());
	// Ricavo traslazione
	// Calcolo baricentro
	const Eigen::RowVectorXd summerVector = Eigen::RowVectorXd::Constant(VA.rows(), 1.0f / VA.rows());
	RowVectorXd bariA = summerVector * VA;
	RowVectorXd bariB = summerVector * VB;
	cout << "Translation\n" << t << endl;
	// Rotazione: Matrice di somme di esterni tra punti iniziali e punti finali
	Eigen::Matrix3d outerProduct = (VA.rowwise() - bariA).eval().transpose() * (VB.rowwise() - bariB).eval();
	cout << outerProduct << endl;

	Matrix3d U, D;
	Vector3d S;
	igl::svd3x3(outerProduct, U, S, D);
	R = U * D.transpose();
	cout << "U*S*D\n" << U * S.asDiagonal() * D.transpose() << endl;
	cout << "R\n" << R << endl;

	t = bariB - bariA * R;
}