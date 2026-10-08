// In-process pybind11/torch binding for CGAL's Alpha_wrap_3, mirroring the pattern already
// used in this project by CuMesh/third_party/xatlas/binding.cpp and
// third_party/QuadriFlow/bindings/quadriflow_binding.cpp: takes/returns torch::Tensor directly,
// GIL released around the actual computation. See BUILD_NOTES.md for how this is built and
// quadriflow_postprocess_plan.md's Phase 6 section for why Alpha Wrap exists in this pipeline.
//
// Even simpler than QuadriFlow's case: alpha_wrap_3 is a single free function operating purely
// on in-memory CGAL containers (std::vector<Point_3>, std::vector<std::vector<size_t>>,
// CGAL::Surface_mesh) -- src/alpha_wrap.cpp (the CLI binary, unchanged, still used standalone)
// only touches disk at its very start (read_polygon_soup) and end (write_polygon_mesh); nothing
// in between needs modification to be called in-process, just different I/O at the boundary.
#include <torch/extension.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/alpha_wrap_3.h>

namespace alpha_wrap_ext {

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_3 Point_3;
typedef CGAL::Surface_mesh<Point_3> Mesh;

void check_tensor(const torch::Tensor& tensor, const std::string& name, torch::ScalarType type) {
    TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == type, name, " has incorrect data type");
}

std::tuple<torch::Tensor, torch::Tensor> wrap(
    const torch::Tensor& vertices,
    const torch::Tensor& faces,
    double alpha,
    double offset
) {
    check_tensor(vertices, "vertices", torch::kFloat32);
    check_tensor(faces, "faces", torch::kInt32);
    TORCH_CHECK(vertices.dim() == 2 && vertices.size(1) == 3, "vertices must be [N, 3]");
    TORCH_CHECK(faces.dim() == 2 && faces.size(1) == 3, "faces must be [M, 3] (triangles)");

    // Build the input triangle soup -- faces MUST be a BackInsertionSequence
    // (std::vector<std::vector<size_t>>, not a fixed-size std::array) per
    // read_polygon_soup's documented concept; alpha_wrap_3 shares that same requirement.
    // See BUILD_NOTES.md's "Known pitfall" section -- this bit the CLI binary once already,
    // don't simplify this back to a fixed-size container.
    std::vector<Point_3> points;
    std::vector<std::vector<std::size_t>> soup_faces;
    {
        auto vacc = vertices.accessor<float, 2>();
        points.reserve(vertices.size(0));
        for (int64_t i = 0; i < vertices.size(0); ++i)
            points.emplace_back(vacc[i][0], vacc[i][1], vacc[i][2]);

        auto facc = faces.accessor<int32_t, 2>();
        soup_faces.reserve(faces.size(0));
        for (int64_t i = 0; i < faces.size(0); ++i)
            soup_faces.push_back({(std::size_t)facc[i][0], (std::size_t)facc[i][1], (std::size_t)facc[i][2]});
    }

    Mesh wrap_mesh;
    {
        py::gil_scoped_release release_gil;
        CGAL::alpha_wrap_3(points, soup_faces, alpha, offset, wrap_mesh);
    }

    auto out_v = torch::empty({(long)wrap_mesh.number_of_vertices(), 3}, torch::kFloat32);
    auto out_v_acc = out_v.accessor<float, 2>();
    for (auto v : wrap_mesh.vertices()) {
        const Point_3& p = wrap_mesh.point(v);
        auto i = v.idx();
        out_v_acc[i][0] = (float)CGAL::to_double(p.x());
        out_v_acc[i][1] = (float)CGAL::to_double(p.y());
        out_v_acc[i][2] = (float)CGAL::to_double(p.z());
    }

    auto out_f = torch::empty({(long)wrap_mesh.number_of_faces(), 3}, torch::kInt32);
    auto out_f_acc = out_f.accessor<int32_t, 2>();
    for (auto f : wrap_mesh.faces()) {
        int j = 0;
        for (auto v : CGAL::vertices_around_face(wrap_mesh.halfedge(f), wrap_mesh)) {
            TORCH_CHECK(j < 3, "alpha_wrap_3 output face is not a triangle -- unexpected");
            out_f_acc[f.idx()][j] = (int32_t)v.idx();
            ++j;
        }
    }

    return std::make_tuple(out_v, out_f);
}

} // namespace alpha_wrap_ext


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "In-process CGAL Alpha_wrap_3 bindings (see third_party/AlphaWrap/BUILD_NOTES.md)";
    m.def("wrap", &alpha_wrap_ext::wrap,
          py::arg("vertices"), py::arg("faces"), py::arg("alpha"), py::arg("offset"),
          "Reconstruct a watertight, 2-manifold, self-intersection-free surface strictly "
          "containing the input triangle soup. vertices: [N,3] float32, faces: [M,3] int32 "
          "(triangles, need not be manifold/watertight). Returns (out_vertices [K,3] float32, "
          "out_faces [L,3] int32).");
}
