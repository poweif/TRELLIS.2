// Minimal standalone CLI wrapping CGAL's Alpha_wrap_3, mirroring third_party/QuadriFlow's
// vendoring pattern: a small self-contained binary called via subprocess, not a Python binding.
//
// Usage: alpha_wrap <input.obj> <alpha> <offset> <output.obj>
//
// alpha_wrap_3 takes a triangle SOUP (no manifoldness/watertightness required at all -- this is
// exactly why it's the right tool here, unlike fill_holes-style patching which needs a
// well-formed boundary polygon to begin with) and produces a guaranteed watertight, 2-manifold,
// self-intersection-free surface that strictly contains the input.
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/alpha_wrap_3.h>
#include <CGAL/IO/polygon_soup_io.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>

#include <array>
#include <iostream>
#include <string>
#include <vector>

// alpha_wrap_3's internal triangulation needs robust (filtered exact) predicates --
// Simple_cartesian<double> (tried first) has none, and crashed inside
// insert_bbox_corners() on even a trivial 320-face icosphere (confirmed via gdb
// backtrace) rather than raising a clean error, presumably from a failed/undefined
// geometric predicate on a degenerate configuration. EPICK is CGAL's own standard
// kernel for this class of algorithm.
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_3 Point_3;
typedef CGAL::Surface_mesh<Point_3> Mesh;

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <input.obj> <alpha> <offset> <output.obj>\n";
        return 1;
    }
    const std::string input_filename = argv[1];
    const double alpha = std::stod(argv[2]);
    const double offset = std::stod(argv[3]);
    const std::string output_filename = argv[4];

    std::cerr << "step: reading " << input_filename << std::endl;
    std::vector<Point_3> points;
    // Must be resizable (BackInsertionSequence) per read_polygon_soup's documented
    // PolygonRange concept -- a fixed-size std::array<size_t,3> (tried first) doesn't
    // satisfy that, and silently compiling/running with the wrong container shape is
    // the likely cause of a segfault inside alpha_wrap_3's own triangulation setup that
    // reproduced identically across two different CGAL major versions (5.6.1 and 6.2).
    std::vector<std::vector<std::size_t>> faces;
    if (!CGAL::IO::read_polygon_soup(input_filename, points, faces)) {
        std::cerr << "Failed to read " << input_filename << "\n";
        return 1;
    }
    std::cerr << "Read soup: " << points.size() << " points, " << faces.size() << " faces" << std::endl;
    std::cerr << "alpha=" << alpha << " offset=" << offset << std::endl;

    std::cerr << "step: calling alpha_wrap_3" << std::endl;
    Mesh wrap;
    CGAL::alpha_wrap_3(points, faces, alpha, offset, wrap);
    std::cerr << "step: alpha_wrap_3 returned" << std::endl;

    std::cerr << "Wrap result: " << num_vertices(wrap) << " vertices, "
              << num_faces(wrap) << " faces" << std::endl;

    std::cerr << "step: writing " << output_filename << std::endl;
    if (!CGAL::IO::write_polygon_mesh(output_filename, wrap)) {
        std::cerr << "Failed to write " << output_filename << "\n";
        return 1;
    }
    std::cerr << "step: done" << std::endl;
    return 0;
}
