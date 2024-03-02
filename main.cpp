#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/opengl/glfw/Viewer.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
//#include <igl/AABB.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <algorithm>
#include "graph.h"
#include "registration.h"


const std::string MODEL_FILE_PATH = "../resources/";

typedef igl::opengl::glfw::Viewer Viewer;


int main(int argc, char* argv[])
{
	srand(time(nullptr));

	Matd VA, VB;
	Mati FA, FB;
	igl::readOBJ(MODEL_FILE_PATH + "rabbit.obj", VA, FA);
	igl::readOBJ(MODEL_FILE_PATH + "rabbit.obj", VB, FB);

	Viewer v;
	v.data().set_face_based(true);

	igl::AABB<Eigen::MatrixXd, 3> treeB;
	treeB.init(VB, FB);


	// Create SubdivisionGraphLevel
	GraphLOD m_res(VB, FB);

	m_res.printf();


	int lodMaxDepth = m_res.lod.size() - 1;
	int lodDepth = 1;


	Matd faceColors(FB.rows(), 3);

	for (int i = 0; i < FB.rows(); i++) {
		faceColors.row(i) = ith_arbitrary_color(m_res.ith_parent(i, lodDepth));
	}
	v.data().set_colors(faceColors);


	v.callback_init = [&](Viewer& viewer)->bool
		{
			// Initialize ImGui
			ImGui::CreateContext();
			ImGui_ImplGlfw_InitForOpenGL(viewer.window, true);
			ImGui_ImplOpenGL3_Init("#version 150");

			return false;
		};
	v.callback_pre_draw = [&](Viewer& v)->bool
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

			for (int i = 0; i < FB.rows(); i++) {
				faceColors.row(i) = ith_arbitrary_color(m_res.ith_parent(i, lodDepth));
			}
			
			//v.data().F_material_diffuse = faceColors;
			v.data().set_colors(faceColors);
			return false;
		};

	v.callback_post_draw = [&](Viewer&)->bool
		{
			// Render ImGui
			ImGui::Render();
			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

			return false;
		};
	v.callback_key_pressed =
		[&](Viewer&, unsigned char key, int)->bool
		{
			switch (key)
			{
			case 'R':
			case 'r':
				apply_random_rotation(VA);
				break;
			case 'T':
			case 't':
				apply_random_translation(VA);
				break;
			case 'Y':
			case 'y':
				icp_iteration(VA, VB, FB, treeB);
				break;
			case 'G':
			case 'g':
				ransac_iteration(VA, VB, FB, treeB);
				break;
			default:
				return false;
			}

			v.data().set_mesh(VA, FA);
			v.data().compute_normals();
			return true;
		};

	v.data().set_mesh(VB, FB);
	v.append_mesh();
	v.data().set_mesh(VA, FA);
	Eigen::MatrixXd F_color2(1, 3);
    F_color2 << 1.0, 0.0, 0.0; 
	v.data().set_colors(F_color2);
	v.core().lighting_factor = 0;
	v.launch();

	// Cleanup
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}