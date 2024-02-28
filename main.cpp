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
#include <igl/barycenter.h>
// include imgui libraries
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include <algorithm>

using namespace Eigen;

const std::string MODEL_FILE_PATH = "../resources/";

struct Edge {
	int a;
	int b;

	Edge(int _a, int _b): a(std::min(_a, _b)), b(std::max(_a, _b)) {};
	bool is_loop() const { return a == b; };
	bool operator < (const Edge& o) const { return a < o.a || (a == o.a && b < o.b); };
	Edge operator = (const Edge& o) { a = o.a; b = o.b; return *this; };
};

struct Graph {
	int num_nodes = 0;
	std::vector<Edge> edges;
	//std::vector<float> area;
	std::vector<int> parents;

	Graph coarsen(); // Non-const because it populates parents vector
};

void shuffle_edges(std::vector<Edge>& edges) {
	for (int i = 0; i < edges.size() * 10; i++) {
		int a = rand() % edges.size();
		int b = rand() % edges.size();
		std::swap(edges[a], edges[b]);
	}
}

Graph Graph::coarsen() {
	Graph coarserGraph;

	//suffle_edges();

	parents.resize(num_nodes, -1);

	for (Edge e : edges)
		if (parents[e.a] == -1 && parents[e.b] == -1)
			parents[e.a] = parents[e.b] = coarserGraph.num_nodes++;

	for (int& i : parents) if (i == -1) i = coarserGraph.num_nodes++;

	std::set<Edge> uniqueEdges;
	for (Edge e : edges) {
		Edge ep(parents[e.a], parents[e.b]);
		if (!ep.is_loop()) uniqueEdges.insert(ep);
	}
	coarserGraph.edges = std::vector<Edge>(uniqueEdges.begin(), uniqueEdges.end());
	shuffle_edges(coarserGraph.edges);
	return coarserGraph;
}

void mesh_to_graph(const MatrixXd& V, const MatrixXi& F, Graph& graph) {
	graph.num_nodes = V.rows();

	std::set<Edge> uniqueEdges;
	for (int i = 0; i < F.rows(); i++) {
		uniqueEdges.insert(Edge(F(i, 0), F(i, 1)));
		uniqueEdges.insert(Edge(F(i, 1), F(i, 2)));
		uniqueEdges.insert(Edge(F(i, 2), F(i, 0)));
	}
	graph.edges = std::vector<Edge>(uniqueEdges.begin(), uniqueEdges.end());
}

struct GraphLOD {
	std::vector<Graph> lod;

	GraphLOD(const MatrixXd& V, const MatrixXi& F) { from_mesh(V, F); };

	void from_mesh(const MatrixXd& V, const MatrixXi& F) {
		lod.resize(1);
		mesh_to_graph(V, F, lod[0]);
		while (lod.back().num_nodes > 1) {
			lod.push_back(lod.back().coarsen());
			// print number of nodes for each level
			std::cout << lod.back().num_nodes << std::endl;
		}
	}

	int ith_parent(int node, int i);
};

int GraphLOD::ith_parent(int node, int i) {
	/*
		0: self
		1: parent
		2: grand-parent
	*/
	int ancestor = node;
	for (int d = 0; d < i; d++) ancestor = lod[d].parents[ancestor];
	return ancestor;
}

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

// Funzione per trovare i vicini di ciascun vertice nella mesh
MatrixXi find_vertex_neighbors(const MatrixXi& FB, int num_vertices) {
	// Creazione della matrice dei vicini dei vertici
	MatrixXi vertex_neighbors = MatrixXi::Zero(num_vertices, 10); // Si suppone che ogni vertice abbia al massimo 10 vicini

	// Contatore per tener traccia del numero di vicini di ciascun vertice
	VectorXi num_neighbors = VectorXi::Zero(num_vertices);

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

void computeAdjacentFaceAreas(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::VectorXd& vertexAreas)
{
	// Inizializza l'array delle aree dei vertici a zero
	vertexAreas.setZero(V.rows());

	// Per ogni faccia
	for (int i = 0; i < F.rows(); ++i)
	{
		// Calcola il baricentro della faccia
		Eigen::RowVector3d barycenter;
		igl::barycenter(V, F.row(i), barycenter);

		// Per ogni vertice nella faccia
		for (int j = 0; j < F.cols(); ++j)
		{
			int vertexIndex = F(i, j);
			// Calcola l'area del triangolo formato dalla faccia e dal vertice
			Eigen::RowVector3d v0 = V.row(F(i, (j + 1) % F.cols()));
			Eigen::RowVector3d v1 = V.row(F(i, (j + 2) % F.cols()));
			double area = 0.5 * (v0 - barycenter).cross(v1 - barycenter).norm();

			// Aggiungi l'area alla somma delle aree dei vertici
			vertexAreas(vertexIndex) += area;
		}
	}
}

double rand_0_to_1() {
	return (double)rand() / RAND_MAX;
}

Vector3d ith_arbitrary_color(int i) {
	srand(i);

	return Vector3d(rand_0_to_1(), rand_0_to_1(), rand_0_to_1());
}


int main(int argc, char* argv[])
{
	srand(time(nullptr));

	MatrixXd VA, VB;
	MatrixXi FA, FB;
	igl::readOBJ(MODEL_FILE_PATH + "rabbit.obj", VA, FA);
	igl::readOBJ(MODEL_FILE_PATH + "rabbit.obj", VB, FB);

	// Initialize viewer
	igl::opengl::glfw::Viewer v;

	igl::AABB<MatrixXd, 3> treeB;
	treeB.init(VB, FB);
	MatrixXd NB;
	igl::per_face_normals(VB, FB, NB);


	// Create SubdivisionGraphLevel
	GraphLOD m_res(VB, FB);

	Eigen::MatrixXd colors(VB.rows(), 3);

	int lodMaxDepth = m_res.lod.size() - 1;
	int lodDepth = 1;

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

	v.callback_init = [&](igl::opengl::glfw::Viewer& viewer)->bool
		{
			// Initialize ImGui
			ImGui::CreateContext();
			ImGui_ImplGlfw_InitForOpenGL(viewer.window, true);
			ImGui_ImplOpenGL3_Init("#version 150");

			return false;
		};
	v.callback_pre_draw = [&](igl::opengl::glfw::Viewer& v)->bool
		{
			// Start ImGui frame
			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();

			// Create ImGui window
			ImGui::Begin("My ImGui Window");

			ImGui::SliderInt("LOD Level", &lodDepth, 1, lodMaxDepth);

			ImGui::Text("Slider Value: %d", lodDepth);
			ImGui::End();

			for (int i = 0; i < VB.rows(); i++) {
				colors.row(i) = ith_arbitrary_color(m_res.ith_parent(i, lodDepth));
			}

			v.data().set_colors(colors);

			return false;
		};

	v.callback_post_draw = [&](igl::opengl::glfw::Viewer&)->bool
		{
			// Render ImGui
			ImGui::Render();
			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

			return false;
		};
	v.callback_key_pressed =
		[&](igl::opengl::glfw::Viewer&, unsigned char key, int)->bool
		{
			switch (key)
			{
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
	v.data().show_lines = false;
	v.launch();

	// Cleanup
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
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

bool contains_row(const Eigen::MatrixXi& matrix, const Eigen::RowVectorXi& row)
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