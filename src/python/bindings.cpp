#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <cuSZp_utility.h>
#include <cuSZp_entry_f32.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;


py::buffer compress(py::array_t<float> original, float tol) {

    unsigned char *compressed_data = nullptr;
    size_t compressed_size = 0;

    // Convert the NumPy array to a C++ vector of float
    std::vector<float> original_data(original.data(), original.data() + original.size());

    compressed_data = new unsigned char[original.size()];

    // pointer_to_original_data(float*), pointer_to_compressed_data(unsigned char*), size_of_original_data(int/size_t), pointer_to_compressed_size, tolerance
    SZp_compress_hostptr_f32(original_data.data(), compressed_data, original.size(), &compressed_size, tol);
    
    return py::array_t<unsigned char>({static_cast<pybind11::ssize_t>(compressed_size)},
                                      reinterpret_cast<unsigned char *>(compressed_data)); // メモリリークする

    // return py::array_t<unsigned char>({compressed_size}, {1},
    //                                   static_cast<unsigned char *>(compressed_data),
    //                                   py::capsule(compressed_data, [](void *ptr) { delete ptr; }));
}

// py::array_t<double> decompress(py::buffer compressed, mgard_x::Config config)
// {
//         // cuSZp decompression.
//     SZp_decompress_hostptr_f32(decData, cmpBytes, nbEle, cmpSize, errorBound);
//     py::buffer_info info = compressed.request();

//     if (info.format != py::format_descriptor<unsigned char>::format()) {
//         throw std::invalid_argument("Input must be a byte array");
//     }
//     if (info.shape.size() != 1) {
//         throw std::invalid_argument("Input must be a 1D array");
//     }

//     void *decompressed_data = nullptr;
//     std::vector<mgard_x::SIZE> shape;
//     mgard_x::data_type dtype;

//     mgard_x::compress_status_type status =
//         decompress(info.ptr, info.size, decompressed_data, shape, dtype, config, false);

//     switch (status) {
//     case mgard_x::compress_status_type::Success:
//         break;
//     case mgard_x::compress_status_type::Failure:
//         throw std::runtime_error("Compression failure");
//         break;
//     case mgard_x::compress_status_type::NotSupportHigherNumberOfDimensionsFailure:
//         throw std::invalid_argument("Not supported higher number of dimensions");
//         break;
//     case mgard_x::compress_status_type::NotSupportDataTypeFailure:
//         throw std::invalid_argument("Not supported data type");
//         break;
//     case mgard_x::compress_status_type::BackendNotAvailableFailure:
//         throw std::invalid_argument("Backend not available");
//         break;
//     }

//     return py::array_t<double>(shape, reinterpret_cast<double *>(decompressed_data),
//                                py::capsule(decompressed_data, [](void *ptr) { delete ptr; }));
// }

PYBIND11_MODULE(_cuSZp, m)
{
    m.doc() = "cuSZp Python bindings";

    m.def("compress", &compress, "Compress a multi-dimensional array", py::arg("original"),
          py::arg("tol")
          );
    // m.def("decompress", &decompress, "Decompress a multi-dimensional array", py::arg("compressed"),
    //       py::arg("config") = mgard_x::Config());
}
