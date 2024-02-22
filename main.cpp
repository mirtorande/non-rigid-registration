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

using namespace Eigen;

const std::string MODEL_FILE_PATH = "../resources/";

struct SubdivisionGraphLevel {
	int num_nodes;
	MatrixXi edges;
	VectorXd area;
	VectorXi parents;
};

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

void create_multiresolution_hierarchy(MatrixXd& vertices, MatrixXi& faces, MatrixXd& normals, MatrixXd& colors);

void mesh_to_graph(const MatrixXd& V, const MatrixXi& F, SubdivisionGraphLevel& graph);

void create_coarser_subdivision_level(SubdivisionGraphLevel& prevSubLvl, SubdivisionGraphLevel& thisSubLvl);

void color_by_parent(const std::vector<SubdivisionGraphLevel>& levels, Eigen::MatrixXd& colors);

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


int main(int argc, char* argv[])
{
	MatrixXd OVX, VA, VB;
	MatrixXi FA, FB;
	igl::readOBJ(MODEL_FILE_PATH + "rabbit.obj", VA, FA);
	igl::readOBJ(MODEL_FILE_PATH + "rabbit.obj", VB, FB);

	// Initialize viewer
	igl::opengl::glfw::Viewer v;

	igl::AABB<MatrixXd, 3> treeB;
	treeB.init(VB, FB);
	MatrixXd NB;
	igl::per_face_normals(VB, FB, NB);

	Eigen::MatrixXd colors(VB.rows(), 3);
	for (int i = 0; i < colors.rows(); ++i) {
		colors.row(i) = Eigen::RowVector3d::Random().cwiseAbs();
	}

	// Create SubdivisionGraphLevel
	SubdivisionGraphLevel lod0;
	mesh_to_graph(VB, FB, lod0);

	// Create coarser subdivision level
	SubdivisionGraphLevel lod1;
	create_coarser_subdivision_level(lod0, lod1);

	// Create coarser subdivision level
	SubdivisionGraphLevel lod2;
	create_coarser_subdivision_level(lod1, lod2);

	// Make a vector of SubdivisionGraphLevel to store the levels
	std::vector<SubdivisionGraphLevel> levels;
	levels.emplace_back(lod0);
	levels.emplace_back(lod1);
	levels.emplace_back(lod2);

	// Color the mesh by parent
	color_by_parent(levels, colors);

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
	v.data().set_colors(colors);
	//v.data().set_colors(RowVector3d(1, 0, 1));
	v.data().show_lines = false;
	//v.append_mesh();
	//v.data().set_mesh(VA, FA);
	//v.data().show_lines = false;
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

void mesh_to_graph(const MatrixXd& V, const MatrixXi& F, SubdivisionGraphLevel& graph) {
	// Inizializza la struttura dati del grafo
	graph.num_nodes = V.rows();
	std::cout << "Nodi contati:" << std::endl << graph.num_nodes << std::endl;

	igl::edges(F, graph.edges);

	// Calcola l'area delle facce adiacenti a ciascun vertice e salva la somma in graph.area
	graph.area = VectorXd::Zero(graph.num_nodes);
	for (int i = 0; i < F.rows(); i++) {
		for (int j = 0; j < 3; j++) {
			graph.area(F(i, j)) += 1.0 / 3.0;
		}
	}

	computeAdjacentFaceAreas(V, F, graph.area);

	// Stampiamo le aree
	std::cout << "Aree calcolate:" << std::endl << graph.area << std::endl;

	graph.parents = VectorXi::Zero(graph.num_nodes);

	// Stampiamo gli edge
	//std::cout << "Edge estratti:" << std::endl << graph.edges << std::endl;
	//graph.parents = VectorXi::Zero(num_vertices);
}

bool containsRow(const Eigen::MatrixXi& matrix, const Eigen::RowVectorXi& row)
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

void create_coarser_subdivision_level(SubdivisionGraphLevel& prevSubLvl, SubdivisionGraphLevel& thisSubLvl) {
	/// 1) Shuffle the edges randomly
	MatrixXi edgesToProcess = prevSubLvl.edges;

	srand(time(nullptr));
	std::random_shuffle(edgesToProcess.data(), edgesToProcess.data() + edgesToProcess.size());

	/// 2) Go tough the edges to process, if the nodes of edge do not have a parent, assign them a new parent
	thisSubLvl.num_nodes = 0;
	thisSubLvl.area = Eigen::VectorXd::Zero(0);

	// Cycle through the edges
	for (int i = 0; i < edgesToProcess.rows(); i++) {
		int node1 = edgesToProcess(i, 0);
		int node2 = edgesToProcess(i, 1);

		// Print the nodes of the edge
		//std::cout << "[" << i << "] Edge: " << node1 << " " << node2 << std::endl;
		// If the nodes do not have a parent, assign them a new parent
		if (prevSubLvl.parents(node1) == 0 && prevSubLvl.parents(node2) == 0) {
				prevSubLvl.parents(node1) = thisSubLvl.num_nodes + 1;
			prevSubLvl.parents(node2) = thisSubLvl.num_nodes + 1;
			thisSubLvl.num_nodes++;
		}

	}
	// If any node remains without a parent, assign it a new parent
	for (int i = 0; i < prevSubLvl.num_nodes; i++) {
		if (prevSubLvl.parents(i) == 0) {
			prevSubLvl.parents(i) = thisSubLvl.num_nodes + 1;
			thisSubLvl.num_nodes++;
		}
	}

	// Print the maxium value of the parents
	std::cout << "Max parent value: " << prevSubLvl.parents.maxCoeff() << std::endl;

	std::cout << "Node calcolati:" << std::endl << thisSubLvl.num_nodes << std::endl;



	// TODO : Is this the best way to initialize the parents?
	thisSubLvl.parents = VectorXi::Zero(thisSubLvl.num_nodes+1);

	/// 3) Go through the edges process again, if the nodes of the edge have different parents add an edge to the new graph, watch out for duplicates and symmetrical edges
	thisSubLvl.edges = Eigen::MatrixXi::Zero(0, 2);

	for (int i = 0; i < edgesToProcess.rows(); i++) {
		int node1 = edgesToProcess(i, 0);
		int node2 = edgesToProcess(i, 1);
		
		// Order nodes in increasing order
		if (node1 > node2) {
			int temp = node1;
			node1 = node2;
			node2 = temp;
		}

		// If the nodes have different parents, add an edge to the new graph
		if (prevSubLvl.parents(node1) != prevSubLvl.parents(node2)) {
			// Check if the edge is already present (the matrix has two columns and each edge is stored in a row)
			Eigen::MatrixXi new_row(1, 2);
			new_row << prevSubLvl.parents(node1), prevSubLvl.parents(node2);

			if (!containsRow(thisSubLvl.edges, new_row)) {
				// Add the edge to the new graph
				thisSubLvl.edges.conservativeResize(thisSubLvl.edges.rows() + 1, Eigen::NoChange);
				thisSubLvl.edges.row(thisSubLvl.edges.rows() - 1) = new_row;
			}

		}
	}
	//std::cout << "Edge calcolati:" << std::endl << thisSubLvl.edges << std::endl;
	for (int i = 0; i < edgesToProcess.rows(); i++) {
		int node1 = edgesToProcess(i, 0);
		int node2 = edgesToProcess(i, 1);
		
		// Order nodes in increasing order
		if (node1 > node2) {
			int temp = node1;
			node1 = node2;
			node2 = temp;
		}

		// If the nodes have different parents, add an edge to the new graph
		if (prevSubLvl.parents(node1) != prevSubLvl.parents(node2)) {
			// Check if the edge is already present (the matrix has two columns and each edge is stored in a row)
			Eigen::MatrixXi new_row(1, 2);
			new_row << prevSubLvl.parents(node1), prevSubLvl.parents(node2);

			if (!containsRow(thisSubLvl.edges, new_row)) {
				// Add the edge to the new graph
				thisSubLvl.edges.conservativeResize(thisSubLvl.edges.rows() + 1, Eigen::NoChange);
				thisSubLvl.edges.row(thisSubLvl.edges.rows() - 1) = new_row;
			}

		}
	}
	//std::cout << "Edge calcolati:" << std::endl << thisSubLvl.edges << std::endl;
	//print the maxium value of the first and second column of the edges
	std::cout << "Max value of the first column of the edges: " << thisSubLvl.edges.col(0).maxCoeff() << std::endl;
	std::cout << "Max value of the second column of the edges: " << thisSubLvl.edges.col(1).maxCoeff() << std::endl;
}

// Funzione che prende in input una serie di SubdivisionGraphLevel e una matrice di colori e restituisce una matrice di colori dove i vertici con gli stessi parent hanno lo stesso colore
void color_by_parent(const std::vector<SubdivisionGraphLevel>& levels, Eigen::MatrixXd& colors) {

	// Calcola i parent di ciascun vertice al livello più alto
	VectorXi parents = VectorXi::Zero(levels[0].num_nodes);
	for (int i = 0; i < levels.size() ; i++) {
		for (int j = 0; j < levels[i].num_nodes; j++) {
			parents(j) = levels[i].parents(j);
		}
	}

	// Assegna un colore univoco a ciascun parent
	std::map<int, Eigen::RowVectorXd> parent_colors;
	for (int i = 0; i < parents.size(); i++) {
		if (parent_colors.find(parents(i)) == parent_colors.end()) {
			parent_colors[parents(i)] = Eigen::RowVectorXd::Random(3);
		}
	}

	// Assegna i colori ai vertici
	for (int i = 0; i < parents.size(); i++) {
		colors.row(i) = parent_colors[parents(i)];
	}
}