// In-process pybind11/torch binding for QuadriFlow, mirroring the pattern already
// established in this project by CuMesh/third_party/xatlas/binding.cpp: a wrapper class
// owning the library's internal state, taking/returning torch::Tensor directly, GIL released
// around the actual solve. See quadriflow_postprocess_plan.md's "in-process QuadriFlow
// bindings" section for why this exists (avoids a temp-file round trip per call) and
// third_party/QuadriFlow/BUILD_NOTES.md for how this extension is built.
//
// QuadriFlow's own pipeline (confirmed by reading main.cpp/parametrizer-mesh.cpp) only
// touches disk at two points -- Load() (reads an .obj) and OutputMesh() (writes one). Everything
// in between is pure in-memory Eigen computation already. parametrizer.hpp/parametrizer-mesh.cpp
// gained two new methods for this binding: LoadFromArrays() and ExtractMesh(), in-memory
// counterparts to those two disk-touching calls -- the CLI binary (main.cpp) is untouched and
// still uses the original file-based Load()/OutputMesh().
#include <torch/extension.h>
#include "parametrizer.hpp"
#include "optimizer.hpp"

#include <optional>

namespace quadriflow_ext {

void check_tensor(const torch::Tensor& tensor, const std::string& name, torch::ScalarType type) {
    TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == type, name, " has incorrect data type");
}

// QuadriFlow's V/F/rho are Eigen types in a 3xN / 3xM *column-major* layout (columns are
// vertices/faces) -- confirmed by reading loader.cpp's own file parser, which builds them the
// same way. Torch tensors here are the more conventional Nx3/Mx3 row-major layout, so converting
// is a transpose + dtype cast (float32 <-> double), not a reinterpret -- done with a plain copy
// loop; mesh sizes here (thousands to tens of thousands of vertices) make this a trivial cost
// next to the actual quadrangulation solve.
Eigen::MatrixXd tensor_to_matrixXd_transposed(const torch::Tensor& t) {
    auto acc = t.accessor<float, 2>();
    Eigen::MatrixXd m(t.size(1), t.size(0));
    for (int64_t i = 0; i < t.size(0); ++i)
        for (int64_t j = 0; j < t.size(1); ++j)
            m(j, i) = static_cast<double>(acc[i][j]);
    return m;
}

Eigen::MatrixXi tensor_to_matrixXi_transposed(const torch::Tensor& t) {
    auto acc = t.accessor<int32_t, 2>();
    Eigen::MatrixXi m(t.size(1), t.size(0));
    for (int64_t i = 0; i < t.size(0); ++i)
        for (int64_t j = 0; j < t.size(1); ++j)
            m(j, i) = acc[i][j];
    return m;
}

torch::Tensor matrixXd_to_tensor_transposed(const Eigen::MatrixXd& m) {
    auto out = torch::empty({m.cols(), m.rows()}, torch::kFloat32);
    auto acc = out.accessor<float, 2>();
    for (int64_t i = 0; i < m.cols(); ++i)
        for (int64_t j = 0; j < m.rows(); ++j)
            acc[i][j] = static_cast<float>(m(j, i));
    return out;
}

torch::Tensor matrixXi_to_tensor_transposed(const Eigen::MatrixXi& m) {
    auto out = torch::empty({m.cols(), m.rows()}, torch::kInt32);
    auto acc = out.accessor<int32_t, 2>();
    for (int64_t i = 0; i < m.cols(); ++i)
        for (int64_t j = 0; j < m.rows(); ++j)
            acc[i][j] = m(j, i);
    return out;
}

// Runs QuadriFlow's exact pipeline (same sequence as main.cpp, minus its file I/O and
// argv-parsing) on in-memory arrays. One call = one fresh qflow::Parametrizer instance --
// confirmed by reading the vendored source that Parametrizer is only ever instantiated as a
// plain local/file-scope object (main.cpp's `Parametrizer field;`) with no hidden global state
// referenced elsewhere, so this is safe to call repeatedly / is not a singleton.
std::tuple<torch::Tensor, torch::Tensor> remesh(
    const torch::Tensor& vertices,
    const torch::Tensor& faces,
    int target_faces,
    std::optional<torch::Tensor> rho,
    bool adaptive,
    bool sharp,
    bool preserve_boundary,
    bool minimum_cost_flow,
    int seed
) {
    check_tensor(vertices, "vertices", torch::kFloat32);
    check_tensor(faces, "faces", torch::kInt32);
    TORCH_CHECK(vertices.dim() == 2 && vertices.size(1) == 3, "vertices must be [N, 3]");
    TORCH_CHECK(faces.dim() == 2 && faces.size(1) == 3, "faces must be [M, 3] (triangles)");

    qflow::Parametrizer field;
    field.flag_preserve_sharp = sharp ? 1 : 0;
    field.flag_preserve_boundary = preserve_boundary ? 1 : 0;
    field.flag_adaptive_scale = (adaptive || rho.has_value()) ? 1 : 0;
    field.flag_minimum_cost_flow = minimum_cost_flow ? 1 : 0;
    field.hierarchy.rng_seed = seed;

    Eigen::VectorXd rho_eigen;
    const Eigen::VectorXd* rho_ptr = nullptr;
    if (rho.has_value()) {
        check_tensor(*rho, "rho", torch::kFloat32);
        TORCH_CHECK(rho->dim() == 1 && rho->size(0) == vertices.size(0),
                    "rho must be a 1D tensor with one entry per vertex");
        auto acc = rho->accessor<float, 1>();
        rho_eigen.resize(rho->size(0));
        for (int64_t i = 0; i < rho->size(0); ++i) rho_eigen[i] = static_cast<double>(acc[i]);
        rho_ptr = &rho_eigen;
    }

    {
        py::gil_scoped_release release_gil;

        field.LoadFromArrays(tensor_to_matrixXd_transposed(vertices),
                              tensor_to_matrixXi_transposed(faces));
        field.Initialize(target_faces, rho_ptr);

        if (field.flag_preserve_boundary) {
            qflow::Hierarchy& mRes = field.hierarchy;
            mRes.clearConstraints();
            for (uint32_t i = 0; i < 3 * mRes.mF.cols(); ++i) {
                if (mRes.mE2E[i] == -1) {
                    uint32_t i0 = mRes.mF(i % 3, i / 3);
                    uint32_t i1 = mRes.mF((i + 1) % 3, i / 3);
                    Eigen::Vector3d p0 = mRes.mV[0].col(i0), p1 = mRes.mV[0].col(i1);
                    Eigen::Vector3d edge = p1 - p0;
                    if (edge.squaredNorm() > 0) {
                        edge.normalize();
                        mRes.mCO[0].col(i0) = p0;
                        mRes.mCO[0].col(i1) = p1;
                        mRes.mCQ[0].col(i0) = mRes.mCQ[0].col(i1) = edge;
                        mRes.mCQw[0][i0] = mRes.mCQw[0][i1] = mRes.mCOw[0][i0] = mRes.mCOw[0][i1] = 1.0;
                    }
                }
            }
            mRes.propagateConstraints();
        }

        qflow::Optimizer::optimize_orientations(field.hierarchy);
        field.ComputeOrientationSingularities();

        if (field.flag_adaptive_scale) {
            field.EstimateSlope();
        }
        qflow::Optimizer::optimize_scale(field.hierarchy, field.rho, field.flag_adaptive_scale);
        field.flag_adaptive_scale = 1;
        qflow::Optimizer::optimize_positions(field.hierarchy, field.flag_adaptive_scale);

        field.ComputePositionSingularities();
        field.ComputeIndexMap();
    }

    Eigen::MatrixXd V_out;
    Eigen::MatrixXi F_out;
    field.ExtractMesh(V_out, F_out);

    return std::make_tuple(matrixXd_to_tensor_transposed(V_out), matrixXi_to_tensor_transposed(F_out));
}

} // namespace quadriflow_ext


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "In-process QuadriFlow bindings (see third_party/QuadriFlow/BUILD_NOTES.md)";
    m.def("remesh", &quadriflow_ext::remesh,
          py::arg("vertices"), py::arg("faces"), py::arg("target_faces"),
          py::arg("rho") = std::nullopt, py::arg("adaptive") = false, py::arg("sharp") = false,
          py::arg("preserve_boundary") = false, py::arg("minimum_cost_flow") = false,
          py::arg("seed") = 0,
          "Quadrangulate a triangle mesh in-process. vertices: [N,3] float32, faces: [M,3] "
          "int32 (triangles). Returns (out_vertices [K,3] float32, out_faces [L,4] int32) -- "
          "genuine quads, always 4 indices per face. rho: optional [N] float32 per-vertex "
          "curvature radius (same units as vertices) for adaptive quad density; supplying it "
          "implies adaptive=True.");
}
