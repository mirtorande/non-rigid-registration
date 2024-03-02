#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <map>
#include <vector>
#include <iostream>

typedef Eigen::MatrixXd Mat;
typedef Eigen::MatrixXi Mati;
typedef Eigen::Vector3d Vec3;

struct Edge {
	int a;
	int b;

	Edge(int _a, int _b) : a(std::min(_a, _b)), b(std::max(_a, _b)) {};
	bool is_loop() const { return a == b; };
	bool operator < (const Edge& o) const { return a < o.a || (a == o.a && b < o.b); };
	Edge operator = (const Edge& o) { a = o.a; b = o.b; return *this; };
	bool operator == (const Edge& o) const { return a == o.a && b == o.b; };
};


struct Graph {
	int num_nodes = 0;
	std::map<Edge, float> edges; // Edge and confine length
	std::vector<float> area;
	std::vector<int> parents;

	Graph coarsen(); // Non-const because it populates parents vector
};

template <class T>
void shuffle(std::vector<T>& v) {
	for (int i = 0; i < v.size() * 10; i++) {
		int a = rand() % v.size();
		int b = rand() % v.size();
		std::swap(v[a], v[b]);
	}
}

Graph Graph::coarsen() {
	Graph coarserGraph;

	// Sort edges giving priority to the ones connecting nodes with the smallest area on the edge with the longest length
	struct EdgeCollapse {
		Edge edge;
		float priority;

		bool operator < (const EdgeCollapse& o) const { return priority < o.priority; };
	};

	std::vector<EdgeCollapse> pEdges;

	for (auto& e : edges) {
		float priority = (area[e.first.a] + area[e.first.b]) / (e.second * e.second);
		pEdges.push_back({ e.first, priority });
	}

	//shuffle(pEdges);
	std::sort(pEdges.begin(), pEdges.end());

	parents.resize(num_nodes, -1);

	for (EdgeCollapse pe : pEdges) {
		Edge e = pe.edge;
		if (parents[e.a] == -1 && parents[e.b] == -1)
			parents[e.a] = parents[e.b] = coarserGraph.num_nodes++;
	}

	for (int& i : parents) if (i == -1) i = coarserGraph.num_nodes++;


	for (const auto& pair : edges) {
		Edge e(parents[pair.first.a], parents[pair.first.b]);
		if (e.is_loop()) continue;

		coarserGraph.edges[e] = coarserGraph.edges[e] + pair.second;
	}

	coarserGraph.area.resize(coarserGraph.num_nodes, 0);
	for (int i = 0; i < num_nodes; i++) {
		coarserGraph.area[parents[i]] += area[i];
	}

	return coarserGraph;
}

/* NEEDS TO BE FIXED

void mesh_ed_to_graph(const MatrixXd& V, const MatrixXi& F, Graph& graph) {
	graph.num_nodes = V.rows();

	std::set<Edge> uniqueEdges;
	for (int i = 0; i < F.rows(); i++) {
		uniqueEdges.insert(Edge(F(i, 0), F(i, 1)));
		uniqueEdges.insert(Edge(F(i, 1), F(i, 2)));
		uniqueEdges.insert(Edge(F(i, 2), F(i, 0)));
	}
	graph.edges = std::vector<Edge>(uniqueEdges.begin(), uniqueEdges.end());
}*/



void mesh_to_graph(const Mat& V, const Mati& F, Graph& graph) {
	graph.num_nodes = F.rows();

	std::map<Edge, int> edgeToFace;

	for (int i = 0; i < F.rows(); i++) {
		for (int j = 0; j < 3; ++j) {
			Edge e(F(i, j), F(i, (j + 1) % 3)); // Cycling through the vertices of the triangle
			if (edgeToFace.find(e) == edgeToFace.end()) edgeToFace[e] = i;
			else {
				Edge ep(edgeToFace[e], i);
				Vec3 v0 = V.row(F(i, j));
				Vec3 v1 = V.row(F(i, (j + 1) % 3));
				float len = (v1 - v0).norm();
				graph.edges[ep] = len;
			}
		}
	}

	// Compute face areas
	graph.area.resize(F.rows());
	for (int i = 0; i < F.rows(); i++) {
		Vec3 v0 = V.row(F(i, 0));
		Vec3 v1 = V.row(F(i, 1));
		Vec3 v2 = V.row(F(i, 2));
		Vec3 n = (v1 - v0).cross(v2 - v0);
		graph.area[i] = n.norm() / 2;
	}

	// Print the number of edges, the edges with their length and the area of each face
	std::cout << "Number of edges: " << graph.edges.size() << std::endl;
	std::cout << "Number of edges: " << graph.num_nodes << std::endl;
}

struct GraphLOD {
	std::vector<Graph> lod;

	GraphLOD(const Mat& V, const Mati& F) { from_mesh(V, F); };

	void from_mesh(const Mat& V, const Mati& F) {
		lod.resize(1);
		mesh_to_graph(V, F, lod[0]);
		while (lod.back().num_nodes > 1) {
			lod.push_back(lod.back().coarsen());
			// print number of nodes for each level
			//std::cout << lod.back().num_nodes << std::endl;
		}
	}

	int ith_parent(int node, int i) {
		/*
			0: self
			1: parent
			2: grand-parent
		*/
		int ancestor = node;
		for (int d = 0; d < i; d++) ancestor = lod[d].parents[ancestor];
		return ancestor;
	}

	void printf() const {
		for (int i = 0; i < lod.size(); i++) {
			std::cout << "Level " << i << " has " << lod[i].num_nodes << " nodes" << std::endl;
		}
	}
};

double rand_0_to_1() {
	return (double)rand() / RAND_MAX;
}

Vec3 ith_arbitrary_color(int i) {
	srand(i);

	return Vec3(rand_0_to_1(), rand_0_to_1(), rand_0_to_1());
}