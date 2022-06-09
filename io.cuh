#ifndef ETA_IO_CUH
#define ETA_IO_CUH

#include "define.cuh"

namespace eta {

void join_path_(std::string& dst) { return; }

template<typename... Args>
void join_path_(std::string& dst, const std::string& a, Args&&... args) {
    dst.push_back('/');
    dst.append(a);
    while (dst.back() == '/') { dst.pop_back(); }
    join_path_(dst, std::forward<Args>(args)...);
}

template<typename... Args>
std::string join_path(const std::string& a, Args&&... args) {
    std::string r{ a };
    while (r.back() == '/') { r.pop_back(); }
    join_path_(r, std::forward<Args>(args)...);
    return r;
}

template<typename T> std::vector<T> read_vector(std::istream& is) {
    size_t size;
    is >> size;

    std::vector<T> r;
    r.resize(size);

    for (size_t i{ 0 }; i < size; ++i) { is >> r[i]; }

    return r;
}

template<typename T> std::vector<T> read_vector(const std::string& path) {
    std::ifstream ifs{ path };
    if (!ifs) { ETA_throw("file error\n"); }
    return read_vector<T>(ifs);
}

} // namespac eta

#endif
