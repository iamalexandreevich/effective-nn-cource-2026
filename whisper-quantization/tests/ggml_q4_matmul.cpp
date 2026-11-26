#include "ggml.h"
#include "ggml-cpu.h"

#include <fstream>
#include <iostream>
#include <vector>

bool read_tensor(const char * path, ggml_tensor * tensor) {
    std::ifstream stream(path, std::ios::binary);
    return bool(stream.read(static_cast<char *>(ggml_get_data(tensor)), ggml_nbytes(tensor)))
        && stream.peek() == std::ifstream::traits_type::eof();
}

int main(int argc, char ** argv) {
    if (argc != 8) {
        std::cerr << "Usage: ggml_q4_matmul width rows batch weight.bin input.f32 decoded.f32 output.f32\n";
        return 1;
    }
    const int width = std::stoi(argv[1]);
    const int rows = std::stoi(argv[2]);
    const int batch = std::stoi(argv[3]);
    if (width <= 0 || width % ggml_blck_size(GGML_TYPE_Q4_0) || rows <= 0 || batch <= 0) {
        return 1;
    }
    const size_t memory = ggml_row_size(GGML_TYPE_Q4_0, width) * rows
        + sizeof(float) * size_t(width + rows) * batch
        + ggml_graph_overhead() + 4 * ggml_tensor_overhead() + (1 << 20);
    ggml_context * context = ggml_init({memory, nullptr, false});
    auto * weights = ggml_new_tensor_2d(context, GGML_TYPE_Q4_0, width, rows);
    auto * input = ggml_new_tensor_2d(context, GGML_TYPE_F32, width, batch);
    if (!read_tensor(argv[4], weights) || !read_tensor(argv[5], input)) {
        std::cerr << "Failed to read the exact tensor payloads\n";
        ggml_free(context);
        return 2;
    }
    std::vector<float> decoded(size_t(width) * rows);
    ggml_get_type_traits(GGML_TYPE_Q4_0)->to_float(ggml_get_data(weights), decoded.data(), int64_t(width) * rows);
    auto * output = ggml_mul_mat(context, weights, input);
    auto * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);
    if (ggml_graph_compute_with_ctx(context, graph, 1) != GGML_STATUS_SUCCESS) {
        std::cerr << "GGML CPU matmul failed\n";
        ggml_free(context);
        return 3;
    }
    std::ofstream decoded_file(argv[6], std::ios::binary);
    std::ofstream output_file(argv[7], std::ios::binary);
    decoded_file.write(reinterpret_cast<const char *>(decoded.data()), decoded.size() * sizeof(float));
    output_file.write(static_cast<const char *>(ggml_get_data(output)), ggml_nbytes(output));
    decoded_file.close();
    output_file.close();
    ggml_free(context);
    return decoded_file && output_file ? 0 : 4;
}
