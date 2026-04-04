#ifndef ICSSL_XDMA_H_
#define ICSSL_XDMA_H_

// icssl_xdma.h
// Minimal XDMA helpers: open H2C/C2H, exact pread/pwrite, and AXI-Lite 32-bit reg access.

#include <cstdint>
#include <cstddef>
#include <cerrno>
#include <cstdio>      // std::perror
#include <stdexcept>   // std::runtime_error
#include <fcntl.h>     // open, O_*
#include <unistd.h>    // pread, pwrite, close
#include <sys/types.h> // off_t

namespace icssl {

static inline int open_xdma_wr(const char* path) {
    int fd = ::open(path, O_WRONLY | O_SYNC);
    if (fd < 0) {
        std::perror(path);
        throw std::runtime_error("open failed");
    }
    return fd;
}

static inline int open_xdma_rd(const char* path) {
    int fd = ::open(path, O_RDONLY | O_SYNC);
    if (fd < 0) {
        std::perror(path);
        throw std::runtime_error("open failed");
    }
    return fd;
}

// Optional: for AXI-Lite (/dev/xdma0_user) you typically want O_RDWR.
static inline int open_xdma_rw(const char* path) {
    int fd = ::open(path, O_RDWR | O_SYNC);
    if (fd < 0) {
        std::perror(path);
        throw std::runtime_error("open failed");
    }
    return fd;
}

static inline void pwrite_exact(int fd, const void* buf, std::size_t len, off_t off) {
    const auto* p = static_cast<const std::uint8_t*>(buf);
    std::size_t done = 0;
    while (done < len) {
        ssize_t w = ::pwrite(fd, p + done, len - done, off + static_cast<off_t>(done));
        if (w < 0) {
            std::perror("pwrite");
            throw std::runtime_error("pwrite failed");
        }
        if (w == 0) {
            throw std::runtime_error("pwrite wrote 0 bytes");
        }
        done += static_cast<std::size_t>(w);
    }
}

static inline void pread_exact(int fd, void* buf, std::size_t len, off_t off) {
    auto* p = static_cast<std::uint8_t*>(buf);
    std::size_t done = 0;
    while (done < len) {
        ssize_t r = ::pread(fd, p + done, len - done, off + static_cast<off_t>(done));
        if (r < 0) {
            std::perror("pread");
            throw std::runtime_error("pread failed");
        }
        if (r == 0) {
            throw std::runtime_error("pread read 0 bytes");
        }
        done += static_cast<std::size_t>(r);
    }
}

// AXI-Lite 32-bit register write/read via /dev/xdma0_user (or similar).
// reg_off must be 4-byte aligned (0x0, 0x4, 0x8, ...).
static inline void axil_write32(int fd_user, off_t reg_off, std::uint32_t v) {
    pwrite_exact(fd_user, &v, sizeof(v), reg_off);
}

static inline std::uint32_t axil_read32(int fd_user, off_t reg_off) {
    std::uint32_t v = 0;
    pread_exact(fd_user, &v, sizeof(v), reg_off);
    return v;
}

} // namespace icssl

#endif // ICSSL_XDMA_H_