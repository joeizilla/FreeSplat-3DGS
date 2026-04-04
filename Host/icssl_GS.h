#include <array>
#include <sstream>
#include <cctype>
#include <string>
#include <fstream>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include <SDL2/SDL.h>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <unistd.h>
#include <vector>
#include <chrono>
#include "icssl_xdma.h"

using point1d_t  = uint16_t;
using point2d_t  = std::array<uint16_t, 2>;
using point3d_t  = std::array<uint16_t, 3>;
using point4d_t  = std::array<uint16_t, 4>;
using matrix3x3_t = std::array<uint16_t, 9>;

struct RGB_u8 {
    uint8_t r, g, b;
};

static inline RGB_u8 decode_pixel_u32rgb(const uint8_t x[4])
{
    // format:
    // x[0] = B
    // x[1] = G
    // x[2] = R
    // x[3] = 0 / reserved
    RGB_u8 out;
    out.b = x[0];
    out.g = x[1];
    out.r = x[2];
    return out;
}

enum class fpga_output_format_t {
    FP16_RGBA_DIV = 0,   // 舊格式：8 bytes / pixel，需 decode_pixel()
    PACKED_BGR888 = 1    // 新格式：4 bytes / pixel, [B,G,R,0]
};

struct fpga_sdl_viewer_t {
    SDL_Window* win = nullptr;
    SDL_Renderer* ren = nullptr;
    SDL_Texture* tex = nullptr;

    int width_org = 300;
    int height_org = 170;
    int scale = 3;
    bool apply_gamma = true;

    fpga_output_format_t out_fmt = fpga_output_format_t::FP16_RGBA_DIV;

    std::vector<uint8_t> rgba;
};

typedef struct {
    matrix3x3_t R;
    point3d_t   t;
    point4d_t   P;
    point1d_t   W;
    point1d_t   H;
    point2d_t   focal;
    point3d_t   cam_center;
} campose_t;

uint16_t make_half(uint8_t hi, uint8_t lo)
{
    return ((uint16_t)hi << 8) | lo;
}

float half_to_float(uint16_t h)
{
    uint16_t h_exp = (h & 0x7C00u);
    uint16_t h_sig = (h & 0x03FFu);
    uint32_t f_sgn = ((uint32_t)h & 0x8000u) << 16;

    uint32_t f_exp, f_sig;

    if (h_exp == 0x0000u) {
        if (h_sig == 0) {
            uint32_t f = f_sgn;
            float out;
            std::memcpy(&out, &f, 4);
            return out;
        }

        int shift = 0;
        while ((h_sig & 0x0400u) == 0) {
            h_sig <<= 1;
            shift++;
        }
        h_sig &= 0x03FFu;
        f_exp = (127 - 15 - shift) << 23;
        f_sig = ((uint32_t)h_sig) << 13;
    }
    else if (h_exp == 0x7C00u) {
        f_exp = 0xFFu << 23;
        f_sig = ((uint32_t)h_sig) << 13;
    }
    else {
        int exp = (h_exp >> 10) - 15 + 127;
        f_exp = ((uint32_t)exp) << 23;
        f_sig = ((uint32_t)h_sig) << 13;
    }

    uint32_t f = f_sgn | f_exp | f_sig;
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

struct RGB {
    float r, g, b;
};

struct fpga_context_t {
    int fd_h2c  = -1;
    int fd_c2h  = -1;
    int fd_user = -1;

    uint64_t addr_2dgs_input  = 0;
    uint64_t addr_3dgs_input  = 0;
    uint64_t addr_image_bank  = 0;

    uint32_t num_tiles   = 836;
    uint32_t tile_offset = 0;

    uint32_t data_count  = 0;
    int mode             = -1;

    fpga_output_format_t out_fmt = fpga_output_format_t::FP16_RGBA_DIV;

    bool initialized     = false;
};

RGB decode_pixel(const uint8_t x[8])
{
    uint16_t r_half = make_half(x[7], x[6]);
    uint16_t g_half = make_half(x[5], x[4]);
    uint16_t b_half = make_half(x[3], x[2]);
    uint16_t a_half = make_half(x[1], x[0]);

    float r = half_to_float(r_half);
    float g = half_to_float(g_half);
    float b = half_to_float(b_half);
    float a = half_to_float(a_half);

    const float eps = 0.02f;
    float denom = a + eps;

    RGB out;
    if (a <= 0.0f || !std::isfinite(denom)) {
        out.r = out.g = out.b = 0.0f;
    } else {
        out.r = r / denom;
        out.g = g / denom;
        out.b = b / denom;
    }
    return out;
}

static uint8_t f01_to_u8(float v) {
    v = std::clamp(v, 0.0f, 1.0f);
    int iv = (int)std::lround(v * 255.0f);
    return (uint8_t)std::clamp(iv, 0, 255);
}

static float linear_to_srgb(float x) {
    x = std::clamp(x, 0.0f, 1.0f);
    if (x <= 0.0031308f) return 12.92f * x;
    return 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
}

void show_fpga_image_sdl_tiled(
    const uint8_t* img_buf,
    int width_org = 300,
    int height_org = 170,
    bool apply_gamma = true,
    int scale = 3
) {
    if (!img_buf) return;

    constexpr int BLK = 8;
    constexpr int BYTES_PER_PIXEL = 8;
    constexpr int W_PAD = 304;
    constexpr int H_PAD = 176;
    constexpr int BX_CNT = W_PAD / BLK;
    constexpr int BY_CNT = H_PAD / BLK;
    constexpr int PIXELS_PER_BLOCK = 64;
    constexpr int BYTES_PER_BLOCK = PIXELS_PER_BLOCK * BYTES_PER_PIXEL;

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::fprintf(stderr, "SDL_Init error: %s\n", SDL_GetError());
        return;
    }

    const int winW = width_org * scale;
    const int winH = height_org * scale;

    SDL_Window* win = SDL_CreateWindow(
        "FPGA Tiled FP16 RGB Viewer",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        winW, winH, 0
    );
    SDL_Renderer* ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture* tex = SDL_CreateTexture(
        ren,
        SDL_PIXELFORMAT_RGBA32,
        SDL_TEXTUREACCESS_STREAMING,
        width_org, height_org
    );

    std::vector<uint8_t> rgba((size_t)width_org * (size_t)height_org * 4, 0);

    auto update = [&]() {
        for (int by = 0; by < BY_CNT; ++by) {
            for (int bx = 0; bx < BX_CNT; ++bx) {
                uint64_t block_idx = (uint64_t)by * BX_CNT + bx;
                const uint8_t* block_ptr = img_buf + block_idx * BYTES_PER_BLOCK;

                for (int iy = 0; iy < BLK; ++iy) {
                    for (int ix = 0; ix < BLK; ++ix) {
                        int x = bx * BLK + ix;
                        int y = by * BLK + iy;
                        if (x >= width_org || y >= height_org) continue;

                        int bank = iy * BLK + ix;
                        const uint8_t* px = block_ptr + bank * BYTES_PER_PIXEL;

                        RGB c = decode_pixel(px);

                        float rr = c.r;
                        float gg = c.g;
                        float bb = c.b;

                        if (apply_gamma) {
                            rr = linear_to_srgb(rr);
                            gg = linear_to_srgb(gg);
                            bb = linear_to_srgb(bb);
                        }

                        size_t idx = ((size_t)y * (size_t)width_org + (size_t)x) * 4;
                        rgba[idx + 0] = f01_to_u8(rr);
                        rgba[idx + 1] = f01_to_u8(gg);
                        rgba[idx + 2] = f01_to_u8(bb);
                        rgba[idx + 3] = 255;
                    }
                }
            }
        }

        SDL_UpdateTexture(tex, nullptr, rgba.data(), width_org * 4);
    };

    update();

    bool running = true;
    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) running = false;
            if (e.type == SDL_KEYDOWN) {
                if (e.key.keysym.sym == SDLK_g) {
                    apply_gamma = !apply_gamma;
                    update();
                }
            }
        }

        SDL_RenderClear(ren);
        SDL_Rect dst{0, 0, winW, winH};
        SDL_RenderCopy(ren, tex, nullptr, &dst);
        SDL_RenderPresent(ren);
        SDL_Delay(16);
    }

    SDL_DestroyTexture(tex);
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
}

static uint8_t hex_char_to_val(char c)
{
    if (c >= '0' && c <= '9') return static_cast<uint8_t>(c - '0');
    if (c >= 'a' && c <= 'f') return static_cast<uint8_t>(c - 'a' + 10);
    if (c >= 'A' && c <= 'F') return static_cast<uint8_t>(c - 'A' + 10);
    throw std::runtime_error(std::string("Invalid hex character: ") + c);
}

static std::string clean_hex_string(const std::string& line)
{
    std::string hex_str;
    for (char c : line) {
        if (c == '_' || std::isspace(static_cast<unsigned char>(c))) {
            continue;
        }
        hex_str.push_back(c);
    }
    return hex_str;
}

bool load_hex_lines_to_buf_mode1(	// Load 3D Gaussians data to input buffer
    const std::string& filename,
    std::vector<uint8_t>& in_buf,
    uint32_t &count)
{
    constexpr size_t HEX_CHARS_PER_LINE   = 108; // 432 bits
    constexpr size_t VALID_BYTES_PER_LINE = 54;
    constexpr size_t STRIDE_BYTES         = 64;

    std::ifstream fin(filename + "/3dgs_input.txt");
    if (!fin) {
        std::cerr << "Cannot open file: " << filename << "\n";
        return false;
    }

	std::ifstream fin_count(filename + "/3dgs_count.txt");
    if (!fin_count)
        return false;
    fin_count >> count;

    // 配置 buffer，每行 64 bytes
    in_buf.assign(static_cast<size_t>(count) * STRIDE_BYTES, 0);

    std::string line;

    for (uint32_t line_idx = 0; line_idx < count; ++line_idx) {
        if (!std::getline(fin, line)) {
            std::cerr << "File has fewer than " << count
                      << " lines. Stopped at line " << line_idx << "\n";
            return false;
        }

        std::string hex_str = clean_hex_string(line);

        if (hex_str.size() != HEX_CHARS_PER_LINE) {
            std::cerr << "Line " << line_idx
                      << " length error. Expect " << HEX_CHARS_PER_LINE
                      << " hex chars, got " << hex_str.size() << "\n";
            return false;
        }

        size_t base_addr = static_cast<size_t>(line_idx) * STRIDE_BYTES;

        // 這一行 64 bytes 先清成 0
        std::memset(in_buf.data() + base_addr, 0, STRIDE_BYTES);

        // 最右邊 byte -> in_buf[base_addr + 0]
        // 最左邊 byte -> in_buf[base_addr + 53]
        for (size_t byte_idx = 0; byte_idx < VALID_BYTES_PER_LINE; ++byte_idx) {
            size_t src_pos = hex_str.size() - 2 * (byte_idx + 1);

            uint8_t hi = hex_char_to_val(hex_str[src_pos]);
            uint8_t lo = hex_char_to_val(hex_str[src_pos + 1]);

            in_buf[base_addr + byte_idx] =
                static_cast<uint8_t>((hi << 4) | lo);
        }
    }

    return true;
}

// read all of the per-gaussian data files in a dataset and pack them
// into the provided buffer.  ``count`` is filled from Count.txt and
// the buffer is resized accordingly.  Returns true on success, false
// if any of the files could not be opened.
bool load_tile_params_mode0(const std::string &input_file,
                      std::vector<uint8_t> &in_buf,
                      uint32_t &count)
{
    std::ifstream fin(input_file + "/Count.txt");
    if (!fin)
        return false;
    fin >> count;

    std::ifstream fin_mean2D(input_file + "/Mean2D.txt");
    if (!fin_mean2D)
        return false;

    std::ifstream fin_Cov2D(input_file + "/Cov2D_inv.txt");
    if (!fin_Cov2D)
        return false;

    std::ifstream fin_Opacity(input_file + "/Opacity.txt");
    if (!fin_Opacity)
        return false;

    std::ifstream fin_Color(input_file + "/Color.txt");
    if (!fin_Color)
        return false;

    std::ifstream fin_Radii(input_file + "/Radii.txt");
    if (!fin_Radii)
        return false;

    in_buf.assign(count * 32, 0);

    for (uint32_t i = 0; i < count; ++i) {
        uint32_t tmp1, tmp2, tmp3;
        fin_mean2D >> tmp1 >> tmp2;

        in_buf[32 * i + 21] = tmp1 >> 8;
        in_buf[32 * i + 20] = tmp1;
        in_buf[32 * i + 19] = tmp2 >> 8;
        in_buf[32 * i + 18] = tmp2;

        fin_Cov2D >> std::hex >> tmp1 >> tmp2 >> tmp3;
        in_buf[32 * i + 17] = tmp1 >> 8;
        in_buf[32 * i + 16] = tmp1;
        in_buf[32 * i + 15] = tmp2 >> 8;
        in_buf[32 * i + 14] = tmp2;
        in_buf[32 * i + 13] = tmp3 >> 8;
        in_buf[32 * i + 12] = tmp3;

        fin_Opacity >> std::hex >> tmp1;
        in_buf[32 * i + 11] = tmp1 >> 8;
        in_buf[32 * i + 10] = tmp1;

        fin_Color >> std::hex >> tmp1 >> tmp2 >> tmp3;
        in_buf[32 * i + 9] = tmp1 >> 8;
        in_buf[32 * i + 8] = tmp1;
        in_buf[32 * i + 7] = tmp2 >> 8;
        in_buf[32 * i + 6] = tmp2;
        in_buf[32 * i + 5] = tmp3 >> 8;
        in_buf[32 * i + 4] = tmp3;

        fin_Radii >> std::dec >> tmp1 >> tmp2;
        in_buf[32 * i + 3] = tmp1 >> 8;
        in_buf[32 * i + 2] = tmp1;
        in_buf[32 * i + 1] = tmp2 >> 8;
        in_buf[32 * i + 0] = tmp2;
    }

    return true;
}

std::vector<campose_t> read_campose_vector_from_file(const std::string& filename)
{
    std::ifstream fin(filename);
    if (!fin) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::vector<campose_t> poses;
    std::string line;
    int line_no = 0;

    while (std::getline(fin, line)) {
        line_no++;

        // 去掉空白
        std::string cleaned;
        for (char c : line) {
            if (!std::isspace(static_cast<unsigned char>(c))) {
                cleaned += c;
            }
        }

        if (cleaned.empty()) continue; // skip empty line

        // split by '_'
        std::vector<uint16_t> vals;
        std::stringstream ss(cleaned);
        std::string token;

        while (std::getline(ss, token, '_')) {
            if (token.empty()) continue;

            unsigned int x = 0;
            std::stringstream hs(token);
            hs >> std::hex >> x;

            if (hs.fail()) {
                throw std::runtime_error(
                    "Invalid hex token at line " + std::to_string(line_no) + ": " + token);
            }

            if (x > 0xFFFF) {
                throw std::runtime_error(
                    "Value out of 16-bit range at line " + std::to_string(line_no) + ": " + token);
            }

            vals.push_back(static_cast<uint16_t>(x));
        }

        constexpr int kExpectedFields = 23;
        if (vals.size() != kExpectedFields) {
            throw std::runtime_error(
                "Line " + std::to_string(line_no) +
                ": expected 23 values but got " + std::to_string(vals.size()));
        }
        campose_t pose{};
        int idx = 0;

        for (int i = 0; i < 9; ++i) pose.R[i] = vals[idx++];
        for (int i = 0; i < 3; ++i) pose.t[i] = vals[idx++];
        for (int i = 0; i < 4; ++i) pose.P[i] = vals[idx++];
        pose.W = vals[idx++];
        pose.H = vals[idx++];
        for (int i = 0; i < 2; ++i) pose.focal[i] = vals[idx++];
        for (int i = 0; i < 3; ++i) pose.cam_center[i] = vals[idx++];

        poses.push_back(pose);
    }

    if (poses.empty()) {
        throw std::runtime_error("No valid campose data found in file.");
    }

    return poses;
}

bool init_fpga(const std::string& input_file,
               fpga_context_t& ctx,
               const campose_t& pose,
               uint64_t CONST_2DGS_INPUT,
               uint64_t CONST_3DGS_INPUT,
               uint64_t CONST_IMAGE_BANK,
               int mode,
			   fpga_output_format_t out_fmt = fpga_output_format_t::FP16_RGBA_DIV)
{
    ctx.addr_2dgs_input = CONST_2DGS_INPUT;
    ctx.addr_3dgs_input = CONST_3DGS_INPUT;
    ctx.addr_image_bank = CONST_IMAGE_BANK;
    ctx.mode = mode;
	ctx.out_fmt = out_fmt;
    ctx.num_tiles = 836;
    ctx.tile_offset = ((300 * 2) << 16) + (170 * 3);

    std::vector<uint8_t> in_buf;
    uint32_t count = 0;

    // 1. open devices
    ctx.fd_h2c  = icssl::open_xdma_wr("/dev/xdma0_h2c_0");
    ctx.fd_c2h  = icssl::open_xdma_rd("/dev/xdma0_c2h_0");
    ctx.fd_user = icssl::open_xdma_rw("/dev/xdma0_user");

    if (ctx.fd_h2c < 0 || ctx.fd_c2h < 0 || ctx.fd_user < 0) {
        std::cerr << "Failed to open XDMA devices.\n";
        return false;
    }

    // 2. reset + fixed registers
    icssl::axil_write32(ctx.fd_user, 4 * 63, 1); // reset
    ::usleep(5);

    // 3. load dataset once
    if (mode == 0) {
        if (!load_tile_params_mode0(input_file, in_buf, count)) {
            std::cerr << "load_tile_params_mode0 failed.\n";
            return false;
        }

        icssl::pwrite_exact(ctx.fd_h2c,
                            in_buf.data(),
                            in_buf.size(),
                            ctx.addr_2dgs_input);

        icssl::axil_write32(ctx.fd_user, 4 * 3, count);
        ctx.data_count = count;
    }
    else if (mode == 1) {
        if (!load_hex_lines_to_buf_mode1(input_file, in_buf, count)) {
            std::cerr << "load_hex_lines_to_buf_mode1 failed.\n";
            return false;
        }

        icssl::pwrite_exact(ctx.fd_h2c,
                            in_buf.data(),
                            in_buf.size(),
                            ctx.addr_3dgs_input);

        icssl::axil_write32(ctx.fd_user, 4 * 9, count);
        ctx.data_count = count;

		
        // load P, W, H, focal
		uint32_t tmp_buf = (static_cast<uint32_t>(pose.P[0]) << 16) | static_cast<uint32_t>(pose.P[1]);
		icssl::axil_write32(ctx.fd_user, 4 * 16, tmp_buf);
        		
		tmp_buf = (static_cast<uint32_t>(pose.P[2]) << 16) | static_cast<uint32_t>(pose.P[3]);
		icssl::axil_write32(ctx.fd_user, 4 * 17, tmp_buf);
        		
		tmp_buf = (static_cast<uint32_t>(pose.W) << 16) | static_cast<uint32_t>(pose.H);
		icssl::axil_write32(ctx.fd_user, 4 * 18, tmp_buf);
        
		tmp_buf = (static_cast<uint32_t>(pose.focal[0]) << 16) | static_cast<uint32_t>(pose.focal[1]);
		icssl::axil_write32(ctx.fd_user, 4 * 19, tmp_buf);

		icssl::axil_write32(ctx.fd_user, 4 * 0, 1);
    }
    else {
        std::cerr << "Unsupported mode: " << mode << "\n";
        return false;
    }

    ctx.initialized = true;
    return true;
}


bool run_fpga_frame(const fpga_context_t& ctx,
                    std::vector<uint8_t>& out_buf,
                    const campose_t& pose)
{
    if (!ctx.initialized) {
        std::cerr << "FPGA context is not initialized.\n";
        return false;
    }

    if (ctx.fd_h2c < 0 || ctx.fd_c2h < 0 || ctx.fd_user < 0) {
        std::cerr << "Invalid FPGA device handles.\n";
        return false;
    }

    if (ctx.mode == 0) {
        icssl::axil_write32(ctx.fd_user, 4 * 3, ctx.data_count);
        icssl::axil_write32(ctx.fd_user, 4 * 0, 1);
    }
    else if (ctx.mode == 1) {
        uint32_t tmp_buf;

        tmp_buf = (static_cast<uint32_t>(pose.R[0]) << 16) | static_cast<uint32_t>(pose.R[1]);
        icssl::axil_write32(ctx.fd_user, 4 * 10, tmp_buf);

        tmp_buf = (static_cast<uint32_t>(pose.R[2]) << 16) | static_cast<uint32_t>(pose.R[3]);
        icssl::axil_write32(ctx.fd_user, 4 * 11, tmp_buf);

        tmp_buf = (static_cast<uint32_t>(pose.R[4]) << 16) | static_cast<uint32_t>(pose.R[5]);
        icssl::axil_write32(ctx.fd_user, 4 * 12, tmp_buf);

        tmp_buf = (static_cast<uint32_t>(pose.R[6]) << 16) | static_cast<uint32_t>(pose.R[7]);
        icssl::axil_write32(ctx.fd_user, 4 * 13, tmp_buf);

        tmp_buf = (static_cast<uint32_t>(pose.R[8]) << 16) | static_cast<uint32_t>(pose.t[0]);
        icssl::axil_write32(ctx.fd_user, 4 * 14, tmp_buf);

        tmp_buf = (static_cast<uint32_t>(pose.t[1]) << 16) | static_cast<uint32_t>(pose.t[2]);
        icssl::axil_write32(ctx.fd_user, 4 * 15, tmp_buf);

        tmp_buf = (static_cast<uint32_t>(pose.cam_center[0]) << 16) | static_cast<uint32_t>(pose.cam_center[1]);
        icssl::axil_write32(ctx.fd_user, 4 * 20, tmp_buf);

        tmp_buf = static_cast<uint32_t>(pose.cam_center[2]) << 16;
        icssl::axil_write32(ctx.fd_user, 4 * 21, tmp_buf);

        for(int i=0;i<9;i++) std::cerr << std::setw(4) << std::setfill('0') << std::hex << pose.R[i] << "_";
        for(int i=0;i<3;i++) std::cerr << std::setw(4) << std::setfill('0') << std::hex << pose.t[i] << "_";
        for(int i=0;i<4;i++) std::cerr << std::setw(4) << std::setfill('0') << std::hex << pose.P[i] << "_";
        std::cerr << std::setw(4) << std::setfill('0') << std::hex << pose.W << "_";
        std::cerr << std::setw(4) << std::setfill('0') << std::hex << pose.H << "_";
        for(int i=0;i<2;i++) std::cerr << std::setw(4) << std::setfill('0') << std::hex << pose.focal[i] << "_";
        for(int i=0;i<3;i++) {
		    std::cerr << std::setw(4) << std::setfill('0') << std::hex << pose.cam_center[i];
		    if(i==2) std::cerr << "\n";
		    else std::cerr << "_";
	    }
/*
        for (int i = 0; i < 9; ++i) pose.R[i] = vals[idx++];
        for (int i = 0; i < 3; ++i) pose.t[i] = vals[idx++];
        for (int i = 0; i < 4; ++i) pose.P[i] = vals[idx++];
        pose.W = vals[idx++];
        pose.H = vals[idx++];
        for (int i = 0; i < 2; ++i) pose.focal[i] = vals[idx++];
        for (int i = 0; i < 3; ++i) pose.cam_center[i] = vals[idx++];
*/




    }
    else {
        std::cerr << "Unsupported mode in context.\n";
        return false;
    }

    icssl::axil_write32(ctx.fd_user, 4 * 29, 1);

    uint32_t rb;
    while (true) {
        rb = icssl::axil_read32(ctx.fd_user, 4 * 29);
        if (rb != 1) break;
        ::usleep(100);
    }

    constexpr size_t BLK = 8;
    constexpr size_t W_PAD = 304;
    constexpr size_t H_PAD = 176;
    constexpr size_t BX_CNT = W_PAD / BLK;  // 38
    constexpr size_t BY_CNT = H_PAD / BLK;  // 22
    constexpr size_t BLOCKS_PER_TILE = BX_CNT * BY_CNT; // 836
    constexpr size_t PIXELS_PER_BLOCK = 64;

    const size_t bytes_per_pixel =
        (ctx.out_fmt == fpga_output_format_t::PACKED_BGR888) ? 4 : 8;
    const size_t bytes_per_block = PIXELS_PER_BLOCK * bytes_per_pixel;
    const size_t bytes_per_tile  = BLOCKS_PER_TILE * bytes_per_block;

    constexpr size_t TILE_COUNT_2X2 = 4;
    const size_t out_size = TILE_COUNT_2X2 * bytes_per_tile;

    out_buf.resize(out_size);
    icssl::pread_exact(ctx.fd_c2h,
                       out_buf.data(),
                       out_buf.size(),
                       static_cast<off_t>(ctx.addr_image_bank));

    return true;
}

void close_fpga(fpga_context_t& ctx)
{
    if (ctx.fd_user >= 0) {
        ::close(ctx.fd_user);
        ctx.fd_user = -1;
    }
    if (ctx.fd_c2h >= 0) {
        ::close(ctx.fd_c2h);
        ctx.fd_c2h = -1;
    }
    if (ctx.fd_h2c >= 0) {
        ::close(ctx.fd_h2c);
        ctx.fd_h2c = -1;
    }

    ctx.initialized = false;
}

bool init_fpga_image_sdl_tiled(
    fpga_sdl_viewer_t& viewer,
    int width_org = 300,
    int height_org = 170,
    bool apply_gamma = true,
    int scale = 1,
    fpga_output_format_t out_fmt = fpga_output_format_t::FP16_RGBA_DIV
) {
    viewer.width_org = width_org;
    viewer.height_org = height_org;
    viewer.scale = scale;
    viewer.apply_gamma = apply_gamma;
    viewer.out_fmt = out_fmt;
    viewer.rgba.assign((size_t)width_org * (size_t)height_org * 4, 0);

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::fprintf(stderr, "SDL_Init error: %s\n", SDL_GetError());
        return false;
    }

    const int winW = width_org * scale;
    const int winH = height_org * scale;

    viewer.win = SDL_CreateWindow(
        (out_fmt == fpga_output_format_t::PACKED_BGR888)
            ? "FPGA Tiled Packed RGB Viewer"
            : "FPGA Tiled FP16 RGB Viewer",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        winW, winH, 0
    );
    if (!viewer.win) {
        std::fprintf(stderr, "SDL_CreateWindow error: %s\n", SDL_GetError());
        SDL_Quit();
        return false;
    }

    viewer.ren = SDL_CreateRenderer(viewer.win, -1, SDL_RENDERER_ACCELERATED);
    if (!viewer.ren) {
        std::fprintf(stderr, "SDL_CreateRenderer error: %s\n", SDL_GetError());
        SDL_DestroyWindow(viewer.win);
        viewer.win = nullptr;
        SDL_Quit();
        return false;
    }

    viewer.tex = SDL_CreateTexture(
        viewer.ren,
        SDL_PIXELFORMAT_RGBA32,
        SDL_TEXTUREACCESS_STREAMING,
        width_org, height_org
    );
    if (!viewer.tex) {
        std::fprintf(stderr, "SDL_CreateTexture error: %s\n", SDL_GetError());
        SDL_DestroyRenderer(viewer.ren);
        SDL_DestroyWindow(viewer.win);
        viewer.ren = nullptr;
        viewer.win = nullptr;
        SDL_Quit();
        return false;
    }

    return true;
}

bool update_fpga_image_sdl_tiled(
    fpga_sdl_viewer_t& viewer,
    const uint8_t* img_buf
) {
    if (!img_buf) return false;
    if (!viewer.win || !viewer.ren || !viewer.tex) return false;

    using clock_t = std::chrono::steady_clock;

    auto t0 = clock_t::now();

    constexpr int BLK = 8;
    constexpr int W_PAD = 304;
    constexpr int H_PAD = 176;
    constexpr int BX_CNT = W_PAD / BLK;
    constexpr int BY_CNT = H_PAD / BLK;
    constexpr int PIXELS_PER_BLOCK = 64;

    const int width_org = viewer.width_org;
    const int height_org = viewer.height_org;

    const int bytes_per_pixel =
        (viewer.out_fmt == fpga_output_format_t::PACKED_BGR888) ? 4 : 8;
    const int bytes_per_block = PIXELS_PER_BLOCK * bytes_per_pixel;

    // ----------------------------------------------------------------
    // Stage 1: untile + decode + pack to linear RGBA buffer
    // ----------------------------------------------------------------
    for (int by = 0; by < BY_CNT; ++by) {
        for (int bx = 0; bx < BX_CNT; ++bx) {
            uint64_t block_idx = (uint64_t)by * BX_CNT + bx;
            const uint8_t* block_ptr = img_buf + block_idx * bytes_per_block;

            for (int iy = 0; iy < BLK; ++iy) {
                for (int ix = 0; ix < BLK; ++ix) {
                    int x = bx * BLK + ix;
                    int y = by * BLK + iy;
                    if (x >= width_org || y >= height_org) continue;

                    int bank = iy * BLK + ix;
                    const uint8_t* px = block_ptr + bank * bytes_per_pixel;

                    size_t idx = ((size_t)y * (size_t)width_org + (size_t)x) * 4;

                    if (viewer.out_fmt == fpga_output_format_t::PACKED_BGR888) {
                        // x[0]=B, x[1]=G, x[2]=R, x[3]=unused
                        viewer.rgba[idx + 0] = px[2];  // R
                        viewer.rgba[idx + 1] = px[1];  // G
                        viewer.rgba[idx + 2] = px[0];  // B
                        viewer.rgba[idx + 3] = 255;
                    } else {
                        RGB c = decode_pixel(px);

                        float rr = c.r;
                        float gg = c.g;
                        float bb = c.b;

                        if (viewer.apply_gamma) {
                            rr = linear_to_srgb(rr);
                            gg = linear_to_srgb(gg);
                            bb = linear_to_srgb(bb);
                        }

                        viewer.rgba[idx + 0] = f01_to_u8(rr);
                        viewer.rgba[idx + 1] = f01_to_u8(gg);
                        viewer.rgba[idx + 2] = f01_to_u8(bb);
                        viewer.rgba[idx + 3] = 255;
                    }
                }
            }
        }
    }

    auto t1 = clock_t::now();

    // ----------------------------------------------------------------
    // Stage 2: upload RGBA buffer to SDL texture
    // ----------------------------------------------------------------
    if (SDL_UpdateTexture(viewer.tex, nullptr, viewer.rgba.data(), width_org * 4) != 0) {
        std::fprintf(stderr, "SDL_UpdateTexture error: %s\n", SDL_GetError());
        return false;
    }

    auto t2 = clock_t::now();

    // ----------------------------------------------------------------
    // Stage 3: render commands before present
    // ----------------------------------------------------------------
    if (SDL_RenderClear(viewer.ren) != 0) {
        std::fprintf(stderr, "SDL_RenderClear error: %s\n", SDL_GetError());
        return false;
    }

    SDL_Rect dst{0, 0, width_org * viewer.scale, height_org * viewer.scale};

    if (SDL_RenderCopy(viewer.ren, viewer.tex, nullptr, &dst) != 0) {
        std::fprintf(stderr, "SDL_RenderCopy error: %s\n", SDL_GetError());
        return false;
    }

    auto t3 = clock_t::now();

    // ----------------------------------------------------------------
    // Stage 4: present
    // ----------------------------------------------------------------
    SDL_RenderPresent(viewer.ren);

    auto t4 = clock_t::now();

    double untile_ms  = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double update_ms  = std::chrono::duration<double, std::milli>(t2 - t1).count();
    double render_ms  = std::chrono::duration<double, std::milli>(t3 - t2).count();
    double present_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();
    double total_ms   = std::chrono::duration<double, std::milli>(t4 - t0).count();

    std::fprintf(stderr,
        "[update_fpga_image_sdl_tiled] "
        "untile=%.3f ms, "
        "updateTex=%.3f ms, "
        "renderCopy=%.3f ms, "
        "present=%.3f ms, "
        "total=%.3f ms\n",
        untile_ms, update_ms, render_ms, present_ms, total_ms);

    return true;
}

bool poll_fpga_image_sdl_events(fpga_sdl_viewer_t& viewer)
{
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        if (e.type == SDL_QUIT) return false;

        if (e.type == SDL_KEYDOWN) {
            if (e.key.keysym.sym == SDLK_g &&
                viewer.out_fmt == fpga_output_format_t::FP16_RGBA_DIV) {
                viewer.apply_gamma = !viewer.apply_gamma;
            }
        }
    }
    return true;
}

void destroy_fpga_image_sdl_tiled(fpga_sdl_viewer_t& viewer)
{
    if (viewer.tex) {
        SDL_DestroyTexture(viewer.tex);
        viewer.tex = nullptr;
    }
    if (viewer.ren) {
        SDL_DestroyRenderer(viewer.ren);
        viewer.ren = nullptr;
    }
    if (viewer.win) {
        SDL_DestroyWindow(viewer.win);
        viewer.win = nullptr;
    }
    SDL_Quit();
}

bool init_fpga_image_sdl_tiled_2x2(
    fpga_sdl_viewer_t& viewer,
    int tile_width = 300,
    int tile_height = 170,
    bool apply_gamma = true,
    int scale = 1,
    fpga_output_format_t out_fmt = fpga_output_format_t::PACKED_BGR888
) {
    const int full_width  = tile_width * 2;
    const int full_height = tile_height * 2;

    viewer.width_org = full_width;
    viewer.height_org = full_height;
    viewer.scale = scale;
    viewer.apply_gamma = apply_gamma;
    viewer.out_fmt = out_fmt;
    viewer.rgba.assign((size_t)full_width * (size_t)full_height * 4, 0);

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::fprintf(stderr, "SDL_Init error: %s\n", SDL_GetError());
        return false;
    }

    const int winW = full_width * scale;
    const int winH = full_height * scale;

    viewer.win = SDL_CreateWindow(
        (out_fmt == fpga_output_format_t::PACKED_BGR888)
            ? "FPGA 2x2 Tiled Packed RGB Viewer"
            : "FPGA 2x2 Tiled FP16 RGB Viewer",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        winW, winH, 0
    );
    if (!viewer.win) {
        std::fprintf(stderr, "SDL_CreateWindow error: %s\n", SDL_GetError());
        SDL_Quit();
        return false;
    }

    viewer.ren = SDL_CreateRenderer(viewer.win, -1, SDL_RENDERER_ACCELERATED);
    if (!viewer.ren) {
        std::fprintf(stderr, "SDL_CreateRenderer error: %s\n", SDL_GetError());
        SDL_DestroyWindow(viewer.win);
        viewer.win = nullptr;
        SDL_Quit();
        return false;
    }

    viewer.tex = SDL_CreateTexture(
        viewer.ren,
        SDL_PIXELFORMAT_RGBA32,
        SDL_TEXTUREACCESS_STREAMING,
        full_width, full_height
    );
    if (!viewer.tex) {
        std::fprintf(stderr, "SDL_CreateTexture error: %s\n", SDL_GetError());
        SDL_DestroyRenderer(viewer.ren);
        SDL_DestroyWindow(viewer.win);
        viewer.ren = nullptr;
        viewer.win = nullptr;
        SDL_Quit();
        return false;
    }

    return true;
}

bool update_fpga_image_sdl_tiled_2x2(
    fpga_sdl_viewer_t& viewer,
    const uint8_t* img_buf,
    int tile_width = 300,
    int tile_height = 170
) {
    if (!img_buf) return false;
    if (!viewer.win || !viewer.ren || !viewer.tex) return false;

    using clock_t = std::chrono::steady_clock;
    auto t0 = clock_t::now();

    constexpr int BLK = 8;
    constexpr int W_PAD = 304;
    constexpr int H_PAD = 176;
    constexpr int BX_CNT = W_PAD / BLK;   // 38
    constexpr int BY_CNT = H_PAD / BLK;   // 22
    constexpr int PIXELS_PER_BLOCK = 64;

    constexpr int TILES_X = 2;
    constexpr int TILES_Y = 2;

    const int full_width  = tile_width * TILES_X;
    const int full_height = tile_height * TILES_Y;

    const int bytes_per_pixel =
        (viewer.out_fmt == fpga_output_format_t::PACKED_BGR888) ? 4 : 8;
    const int bytes_per_block = PIXELS_PER_BLOCK * bytes_per_pixel;
    const size_t bytes_per_tile =
        (size_t)BX_CNT * (size_t)BY_CNT * (size_t)bytes_per_block;

    if ((int)viewer.width_org != full_width || (int)viewer.height_org != full_height) {
        std::fprintf(stderr,
            "update_fpga_image_sdl_tiled_2x2 size mismatch: viewer=(%d,%d), expected=(%d,%d)\n",
            viewer.width_org, viewer.height_org, full_width, full_height);
        return false;
    }

    std::fill(viewer.rgba.begin(), viewer.rgba.end(), 0);

    // ------------------------------------------------------------
    // Stage 1: untile + decode + stitch 4 tiles into one RGBA image
    // ------------------------------------------------------------
    for (int tile_y = 0; tile_y < TILES_Y; ++tile_y) {
        for (int tile_x = 0; tile_x < TILES_X; ++tile_x) {
            const int tile_id = tile_y * TILES_X + tile_x;
            const uint8_t* tile_base = img_buf + (size_t)tile_id * bytes_per_tile;

            for (int by = 0; by < BY_CNT; ++by) {
                for (int bx = 0; bx < BX_CNT; ++bx) {
                    uint64_t block_idx = (uint64_t)by * BX_CNT + bx;
                    const uint8_t* block_ptr =
                        tile_base + (size_t)block_idx * bytes_per_block;

                    for (int iy = 0; iy < BLK; ++iy) {
                        for (int ix = 0; ix < BLK; ++ix) {
                            int local_x = bx * BLK + ix;
                            int local_y = by * BLK + iy;

                            if (local_x >= tile_width || local_y >= tile_height) continue;

                            int out_x = tile_x * tile_width + local_x;
                            int out_y = tile_y * tile_height + local_y;

                            int bank = iy * BLK + ix;
                            const uint8_t* px = block_ptr + bank * bytes_per_pixel;

                            size_t idx =
                                ((size_t)out_y * (size_t)full_width + (size_t)out_x) * 4;

                            if (viewer.out_fmt == fpga_output_format_t::PACKED_BGR888) {
                                // px[0]=B, px[1]=G, px[2]=R, px[3]=unused
                                viewer.rgba[idx + 0] = px[2];  // R
                                viewer.rgba[idx + 1] = px[1];  // G
                                viewer.rgba[idx + 2] = px[0];  // B
                                viewer.rgba[idx + 3] = 255;
                            } else {
                                RGB c = decode_pixel(px);

                                float rr = c.r;
                                float gg = c.g;
                                float bb = c.b;

                                if (viewer.apply_gamma) {
                                    rr = linear_to_srgb(rr);
                                    gg = linear_to_srgb(gg);
                                    bb = linear_to_srgb(bb);
                                }

                                viewer.rgba[idx + 0] = f01_to_u8(rr);
                                viewer.rgba[idx + 1] = f01_to_u8(gg);
                                viewer.rgba[idx + 2] = f01_to_u8(bb);
                                viewer.rgba[idx + 3] = 255;
                            }
                        }
                    }
                }
            }
        }
    }

    auto t1 = clock_t::now();

    // ------------------------------------------------------------
    // Stage 2: upload RGBA buffer to SDL texture
    // ------------------------------------------------------------
    if (SDL_UpdateTexture(viewer.tex, nullptr, viewer.rgba.data(), full_width * 4) != 0) {
        std::fprintf(stderr, "SDL_UpdateTexture error: %s\n", SDL_GetError());
        return false;
    }

    auto t2 = clock_t::now();

    // ------------------------------------------------------------
    // Stage 3: render
    // ------------------------------------------------------------
    if (SDL_RenderClear(viewer.ren) != 0) {
        std::fprintf(stderr, "SDL_RenderClear error: %s\n", SDL_GetError());
        return false;
    }

    SDL_Rect dst{0, 0, full_width * viewer.scale, full_height * viewer.scale};

    if (SDL_RenderCopy(viewer.ren, viewer.tex, nullptr, &dst) != 0) {
        std::fprintf(stderr, "SDL_RenderCopy error: %s\n", SDL_GetError());
        return false;
    }

    auto t3 = clock_t::now();

    // ------------------------------------------------------------
    // Stage 4: present
    // ------------------------------------------------------------
    SDL_RenderPresent(viewer.ren);

    auto t4 = clock_t::now();

    double untile_ms  = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double update_ms  = std::chrono::duration<double, std::milli>(t2 - t1).count();
    double render_ms  = std::chrono::duration<double, std::milli>(t3 - t2).count();
    double present_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();
    double total_ms   = std::chrono::duration<double, std::milli>(t4 - t0).count();

    std::fprintf(stderr,
        "[update_fpga_image_sdl_tiled_2x2] "
        "untile=%.3f ms, "
        "updateTex=%.3f ms, "
        "renderCopy=%.3f ms, "
        "present=%.3f ms, "
        "total=%.3f ms\n",
        untile_ms, update_ms, render_ms, present_ms, total_ms);

    return true;
}
