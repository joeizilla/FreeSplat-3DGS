#include "VIntegrate.h"
#include "verilated.h"
#include "verilated_vcd_c.h"

#include <SDL2/SDL.h>
#include <cstdint>
#include <array>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <cstdio>
#include <memory>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <iomanip>
#include <stdint.h>
#include <vector>
#include <bitset>
#include <time.h>

///////////////////////////////		GATTK
#define CONST_2DGS_INPUT		0x000000000ull
#define CONST_3DGS_INPUT		0x040000000ull
#define CONST_IMAGE_BANK		0x080000000ull
///////////////////////////////

//#ifdef
//#else
//#endif

#define DRAM_LATENCY    20
#define REG_LATENCY     5

#define IDLE            0
#define READ_ADDR       1
#define READ_DATA       2
#define WRITE_ADDR      3
#define WRITE_DATA      4
#define RESP            5
#define DELAY           6
#define READ_ADDR_WAIT  7
#define WRITE_ADDR_WAIT 8
#define WRITE_DATA_WAIT 9
#define FINISH          10

#define DRAM_SIZE       0x400000000ull //16GB

std::unique_ptr<uint8_t[]> fpga_dram;

using point1d_t  = uint16_t;
using point2d_t  = std::array<uint16_t, 2>;
using point3d_t  = std::array<uint16_t, 3>;
using point4d_t  = std::array<uint16_t, 4>;
using matrix3x3_t = std::array<uint16_t, 9>;

typedef struct {
    matrix3x3_t R;
    point3d_t   t;
    point4d_t   P;
    point1d_t   W;
    point1d_t   H;
    point2d_t   focal;
    point3d_t   cam_center;
} campose_t;

campose_t read_campose_from_file(const std::string& filename)
{
    std::ifstream fin(filename);
    if (!fin) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    // 讀整個檔案
    std::stringstream buffer;
    buffer << fin.rdbuf();
    std::string s = buffer.str();

    // 去掉空白字元
    std::string cleaned;
    for (char c : s) {
        if (!std::isspace(static_cast<unsigned char>(c))) {
            cleaned += c;
        }
    }

    // 依 '_' 切開並轉成 16-bit 數值
    std::vector<uint16_t> vals;
    std::stringstream ss(cleaned);
    std::string token;

    while (std::getline(ss, token, '_')) {
        if (token.empty()) continue;

        unsigned int x = 0;
        std::stringstream hs(token);
        hs >> std::hex >> x;

        if (hs.fail()) {
            throw std::runtime_error("Invalid hex token: " + token);
        }

        if (x > 0xFFFF) {
            throw std::runtime_error("Value out of 16-bit range: " + token);
        }

        vals.push_back(static_cast<uint16_t>(x));
    }

    constexpr int kExpectedFields = 23;
    if (vals.size() != kExpectedFields) {
        throw std::runtime_error(
            "campose_t requires 23 half values, but got " + std::to_string(vals.size()));
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

    //for (int i = 0; i < 9; ++i) pose.R[i] = vals[idx--];
    //for (int i = 0; i < 3; ++i) pose.t[i] = vals[idx--];
    //for (int i = 0; i < 4; ++i) pose.P[i] = vals[idx--];
    //pose.W = vals[idx--];
    //pose.H = vals[idx--];
    //for (int i = 0; i < 2; ++i) pose.focal[i] = vals[idx--];
    //for (int i = 0; i < 3; ++i) pose.cam_center[i] = vals[idx--];

    return pose;
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

bool load_hex_lines_to_fpga_dram(
    const std::string& filename,
    uint64_t offset,
    uint32_t N)
{
    constexpr size_t HEX_CHARS_PER_LINE   = 108; // 432 bits
    constexpr size_t VALID_BYTES_PER_LINE = 54;
    constexpr size_t STRIDE_BYTES         = 64;

    if (!fpga_dram) {
        std::cerr << "fpga_dram is not allocated.\n";
        return false;
    }

    // offset overflow
    if (offset + static_cast<uint64_t>(N) * STRIDE_BYTES > DRAM_SIZE) {
        std::cerr << "load_hex_lines_to_fpga_dram overflow: "
                  << "offset=0x" << std::hex << offset
                  << ", N=" << std::dec << N
                  << ", stride=" << STRIDE_BYTES
                  << ", DRAM_SIZE=0x" << std::hex << DRAM_SIZE << "\n";
        return false;
    }

    std::ifstream fin(filename);
    if (!fin) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return false;
    }

    std::string line;

    for (uint32_t line_idx = 0; line_idx < N; ++line_idx) {
        if (!std::getline(fin, line)) {
            std::cerr << "File has fewer than " << N
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

        uint64_t base_addr = offset + static_cast<uint64_t>(line_idx) * STRIDE_BYTES;

        // 這一行 64 bytes 先清成 0
        std::memset(fpga_dram.get() + base_addr, 0, STRIDE_BYTES);

        // 最右邊 byte -> fpga_dram[base_addr + 0]
        // 最左邊 byte -> fpga_dram[base_addr + 53]
        for (size_t byte_idx = 0; byte_idx < VALID_BYTES_PER_LINE; ++byte_idx) {
            size_t src_pos = hex_str.size() - 2 * (byte_idx + 1);

            uint8_t hi = hex_char_to_val(hex_str[src_pos]);
            uint8_t lo = hex_char_to_val(hex_str[src_pos + 1]);

            fpga_dram[base_addr + byte_idx] = static_cast<uint8_t>((hi << 4) | lo);
        }
    }

    return true;
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

void WATCH_DOG_PE_cluster(
std::ofstream &os,
VlWide<12> d_sel,
VlWide<32> cws_out,
uint64_t cov,
uint16_t opacity,
uint64_t colors
) {
	for(uint32_t i = 0; i < 12 ; i++) {
		os << std::setw(8) << std::setfill('0') << std::hex << d_sel[11-i];
		
		if(i == 11) os << "\n";
		else os << "_";
	}

	return;
}

void WATCH_DOG_Projection(
std::ofstream &os,
uint32_t valid,
VlWide<5> qsdata_2d
) {
	if(valid) {
		for(int i=0;i<5;i++) {
			os << std::setw(8) << std::setfill('0') << std::hex << qsdata_2d[4-i];
		}
		os << "\n";
	}
	return;
}

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
    float r,g,b;
};

RGB decode_pixel(const uint8_t x[8])
{
    uint16_t r_half = make_half(x[7], x[6]);
    uint16_t g_half = make_half(x[5], x[4]);
    uint16_t b_half = make_half(x[3], x[2]);
    uint16_t d_half = make_half(x[1], x[0]);

    float r = half_to_float(r_half);
    float g = half_to_float(g_half);
    float b = half_to_float(b_half);
    float d = half_to_float(d_half);

	const float eps = 0.02f;

    RGB out;

	float denom = d + eps;

    if (d <= 0.0f || !std::isfinite(denom)) {
        out.r = out.g = out.b = 0.0f;
    } else {
        out.r = r / denom;
        out.g = g / denom;
        out.b = b / denom;
    }

    return out;
}

void show_fpga_image_sdl_tiled(
    const uint8_t* fpga_dram,
    uint64_t base_addr,
    int width_org = 300,
    int height_org = 170,
    bool apply_gamma = true,
    int scale = 3
) {
    if (!fpga_dram) return;

    constexpr int BLK = 8;
    constexpr int BYTES_PER_PIXEL = 8;
    constexpr int W_PAD = 304;
    constexpr int H_PAD = 176;
    constexpr int BX_CNT = W_PAD / BLK;   // 38
    constexpr int BY_CNT = H_PAD / BLK;   // 22
    constexpr int PIXELS_PER_BLOCK = 64;
    constexpr int BYTES_PER_BLOCK = PIXELS_PER_BLOCK * BYTES_PER_PIXEL; // 512

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::fprintf(stderr, "SDL_Init error: %s\n", SDL_GetError());
        return;
    }

    const int winW = width_org * scale;
    const int winH = height_org * scale;

    SDL_Window* win = SDL_CreateWindow(
        "Verilator (Emulation) Tiled FP16 RGB Viewer",
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
                const uint8_t* block_ptr = fpga_dram + base_addr + block_idx * BYTES_PER_BLOCK;

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

                        uint8_t R = f01_to_u8(rr);
                        uint8_t G = f01_to_u8(gg);
                        uint8_t B = f01_to_u8(bb);

                        size_t idx = ((size_t)y * (size_t)width_org + (size_t)x) * 4;
                        rgba[idx + 0] = R;
                        rgba[idx + 1] = G;
                        rgba[idx + 2] = B;
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
                if (e.key.keysym.sym == SDLK_r) {
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

// -------------- don't touch -------------- Edited by Yi-Chung
class w_element {
	public:
		uint32_t	addr;
		uint32_t	data;

		w_element() {
			addr	= 0;
			data	= 0;
		}
};

class r_element {
	public:
		uint32_t	addr;

		r_element() {
			addr	= 0;
		}
};

class dram_ch_r {
	public:
		//AXI-AR channel
		uint64_t    araddr;
		uint8_t     arlen;
		bool        arready;
		bool        arvalid;
    
		//AXI-R channel
		VlWide<16>  rdata;
		bool        rlast;
		bool        rready;
		uint8_t     rresp;
		bool        rvalid;

		//reg current
		uint8_t     state[2];
		uint16_t    counter[2];
		uint16_t    burstcount[2];
		uint8_t     err[2];
		uint64_t    araddr_reg[2];
		
		//random delay 
		uint8_t		random_delay;

		//Constructor: all zeros
		dram_ch_r() {	
			araddr			= 0;
			arlen			= 0;
			arready			= 0;
			arvalid			= 0;
			
			for(int i=0;i<16;i++) rdata[i]	= 0;
			rlast			= 0;
			rready			= 0;
			rresp			= 0;
			rvalid			= 0;
			
			state[0]		= 0;
			counter[0]		= 0;
			burstcount[0]	= 0;
			err[0]			= 0;
			araddr_reg[0]	= 0;
			
			state[1]		= 0;
			counter[1]		= 0;
			burstcount[1]	= 0;
			err[1]			= 0;
			araddr_reg[1]	= 0;

			random_delay	= DRAM_LATENCY;
		}
		
		//evaluate new outputs and reg data
		void eval(){
			switch(state[0]) {
				case IDLE:
					state[1]		= arvalid ? DELAY : state[0];
					araddr_reg[1]	= arvalid ? araddr : araddr_reg[0];
					burstcount[1]	= arvalid ? arlen : burstcount[0];
					err[1]			= arvalid ? (araddr%64 != 0) : err[0];
					random_delay	= arvalid ? DRAM_LATENCY + rand()%20 : random_delay;
					break;
				case DELAY:
					if(counter[0] == random_delay) {
						state[1]	= READ_DATA;
						counter[1]	= 0;
					}
					else {
						state[1]	= state[0];
						counter[1]	= counter[0] + 1;
					}
					break;
				case READ_DATA:
					state[1]		= (rready && (counter[0] == burstcount[0])) ? IDLE : state[0];
					err[1]			= (rready && (counter[0] == burstcount[0])) ? 0 : err[0];
					counter[1]		= (rready && (counter[0] == burstcount[0])) ? 0 : counter[0] + rready;
					break;
			}
			
			arready 		= (state[1] == IDLE);
			rvalid			= (state[1] == READ_DATA);
			rlast			= rvalid && (counter[1] == burstcount[1]);
			rresp			= (rvalid && err[1]!=0) ? 2 : 0;
			
			if(rvalid) {
				uint32_t tmp 	= 0;
				for(int i=0;i<16;i++) {
					tmp			= 0;
					for(int j=0;j<4;j++) {
						tmp		= tmp << 8;
						tmp 	+= fpga_dram[araddr_reg[1] + 64*counter[1] + 63 - i*4 - j];
					}
					rdata[15-i]	= tmp;
				}
			}
			else {
				for(int i=0;i<16;i++) rdata[i]	= 0;
			}
		}
		
		//update reg data
		void update(){
			state[0]		= state[1];
			counter[0]		= counter[1];
			burstcount[0]	= burstcount[1];
			err[0]			= err[1];
			araddr_reg[0]	= araddr_reg[1];
		}
};

//DRAM Write Channel
class dram_ch_w{
	public:
		uint64_t    awaddr;
		uint8_t     awlen;
		bool        awready;
		bool        awvalid;
    
		//AXI-RESP channel
		bool        bready;
		uint8_t     bresp;
		bool        bvalid;
    
		//AXI-W channel
		VlWide<16>  wdata;
		bool        wlast;
		bool        wready;
		bool        wvalid;
		uint64_t    wstrb;

		//reg control
		uint8_t     state[2];
		uint16_t    counter[2];
		uint8_t     err[2];
		uint64_t    awaddr_reg[2];

		//random delay
		uint8_t		random_delay;

		//Constructor: all zeros
		dram_ch_w() {
			awaddr			= 0;
			awlen			= 0;
			awready			= 0;
			awvalid			= 0;
			bready			= 0;
			bresp			= 0;
			bvalid			= 0;
			for(int i=0;i<16;i=i+1) wdata[i]	= 0;
			wlast			= 0;
			wready			= 0;
			wvalid			= 0;
			wstrb			= 0xffffffffffffffff;
			
			state[0]		= 0;
			counter[0]		= 0;
			err[0]			= 0;
			awaddr_reg[0]	= 0;
			
			state[1]		= 0;
			counter[1]		= 0;
			err[1]			= 0;
			awaddr_reg[1]	= 0;

			random_delay	= DRAM_LATENCY;
		}
		
		//evaluate new outputs and reg data
		void eval(){

			switch(state[0]) {
				case IDLE:
					state[1]		= awvalid ? DELAY : state[0];
					awaddr_reg[1]	= awvalid ? awaddr : awaddr_reg[0];
					err[1]			= awvalid ? (awaddr%64 != 0) : err[0];
					random_delay	= awvalid ? DRAM_LATENCY + rand()%20 : random_delay;
					break;
				case DELAY:
					if(counter[0] == random_delay) {
						state[1]	= WRITE_DATA;
						counter[1]	= 0;
					}
					else {
						state[1]	= state[0];
						counter[1]	= counter[0] + 1;
					}
				
					break;
				case WRITE_DATA:
					state[1]		= (wvalid & wlast) ? RESP : state[0];
					counter[1]		= wlast ? 0 : (counter[0] + wvalid);
					break;
				case RESP:
					err[1]			= bready ? 0 : err[0];
					state[1]		= bready ? IDLE : state[0];
					counter[1]		= 0;
					break;
			}
			
			awready         = (state[1] == IDLE);
			wready          = (state[1] == WRITE_DATA);
			bvalid          = (state[1] == RESP);
			bresp           = (state[1] == RESP && err[1] != 0) ? 2 : 0;
		}
		
		//update reg data
		void update() {
			if(state[0] == WRITE_DATA) {
				if(wvalid) {
					//std::cerr << "Here. " << uint64_t(awaddr_reg[0]) << "\t" << uint32_t(counter[0]) << "\n";
					uint32_t tmp;
					uint64_t tmp_mask = wstrb;
					for(int i=0;i<16;i++) {
						tmp = wdata[i];
						for(int j=0;j<4;j++) {
							//std::cerr << std::hex << std::setw(2) << std::setfill('0') << uint32_t(tmp%256) << "\n";
							if(tmp_mask%2==1) fpga_dram[awaddr_reg[0] + 64*counter[0] + i*4 + j] = tmp;
							tmp_mask = tmp_mask >> 1;
							tmp = tmp >> 8;
						}
					}
				}
			}
			state[0]		= state[1];
			counter[0]		= counter[1];
			err[0]			= err[1];
			awaddr_reg[0]	= awaddr_reg[1];
		}
};

//REG Read Channel
class reg_ch_r{
	public:
		uint32_t	reg_addr;
		bool		finish;
		uint32_t	out_data;

		//AXIL-AR channel
		uint32_t	araddr;
		bool		arready;
		bool		arvalid;

		//AXIL-R channel
		uint32_t	rdata;
		bool		rready;
		bool		rvalid;

		//reg control
		uint8_t		state[2];
		uint8_t		counter[2];
		
		//Constructor: all zeros
		reg_ch_r() {
			reg_addr	= 0;
			finish		= 0;
			out_data	= 0;
			
			araddr		= 0;
			arready		= 0;
			arvalid		= 0;
			rdata		= 0;
			rready		= 0;
			rvalid		= 0;
			
			state[0]	= 0;
			counter[0]	= 0;
			
			state[1]	= 0;
			counter[1]	= 0;
		}
		
		//Assign read addr.
		void assign(uint32_t addr) {
			reg_addr	= addr;
		}
		
		//Clear all outputs and reg
		void clear() {
			reg_addr	= 0;
			finish		= 0;
			out_data	= 0;
			
			araddr		= 0;
			arready		= 0;
			arvalid		= 0;
			rdata		= 0;
			rready		= 0;
			rvalid		= 0;
			
			state[0]	= 0;
			counter[0]	= 0;
			
			state[1]	= 0;
			counter[1]	= 0;
		}
		
		//evaluate outputs and reg data
		void eval(){
			switch(state[0]) {
				case IDLE:
					state[1]	= READ_ADDR;
					break;
				case READ_ADDR:
					state[1]	= arready ? READ_ADDR_WAIT : state[0];
					break;
				case READ_ADDR_WAIT:
					state[1]	= counter[0] == 5 ? READ_DATA : state[0];
					counter[1]	= counter[0] == 5 ? 0 : counter[0] + 1;
					break;
				case READ_DATA:
					state[1]	= rvalid ? FINISH : state[0];
					break;
				case FINISH:
					state[1]	= IDLE;
					break;
			}
			
			finish		= (state[0] == FINISH);
			arvalid		= (state[0] == READ_ADDR);
			araddr		= arvalid ? reg_addr : 0;
			rready		= (state[0] == READ_DATA);
			out_data	= (rready && rvalid) ? rdata : out_data;
		}
		
		//update reg data
		void update() {
			state[0]	= state[1];
			counter[0]	= counter[1];
		}
};

//REG Write Channel
class reg_ch_w{
	public:
		//C++ interface
		uint32_t    reg_addr;
		uint32_t    reg_data;
		bool        finish;

		//AXIL-AW channel
		uint32_t    awaddr;
		bool        awready;
		bool        awvalid;
		bool        bready;
		bool        bvalid;

		//AXIL-W channel
		uint32_t    wdata;
		bool        wready;
		bool        wvalid;

		//reg control
		uint8_t     state[2];
		uint8_t     counter[2];

		//Constructor: all zeros
		reg_ch_w() {
			reg_addr		= 0;
			reg_data		= 0;
			finish			= 0;
			
			awaddr			= 0;
			awready			= 0;
			awvalid			= 0;
			bready			= 0;
			bvalid			= 0;
			
			wdata			= 0;
			wready			= 0;
			wvalid			= 0;
			
			state[0]		= 0;
			counter[0]		= 0;
			state[1]		= 0;
			counter[1]		= 0;
		}
		
		//Assign write addr and write data
		void assign(uint32_t addr, uint32_t data) {
			reg_addr		= addr;
			reg_data		= data;
		}
		
		//Clear all outputs and reg
		void clear() {
			reg_addr		= 0;
			reg_data		= 0;
			finish			= 0;
			
			awaddr			= 0;
			awready			= 0;
			awvalid			= 0;
			bready			= 0;
			bvalid			= 0;
			
			wdata			= 0;
			wready			= 0;
			wvalid			= 0;
			
			state[0]		= 0;
			counter[0]		= 0;
			state[1]		= 0;
			counter[1]		= 0;
		}
		
		//evaluate outputs and reg data
		void eval(){
			switch(state[0]) {
				case IDLE:
					state[1]	= WRITE_ADDR;
					break;
				case WRITE_ADDR:
					state[1]		= awready ? WRITE_ADDR_WAIT : state[0];
					break;
				case WRITE_ADDR_WAIT:
					state[1]	= counter[0] == 5 ? WRITE_DATA : state[0];
					counter[1]	= counter[0] == 5 ? 0 : counter[0] + 1;
					break;
				case WRITE_DATA:
					state[1]	= wready ? WRITE_DATA_WAIT : state[0];
					break;
				case WRITE_DATA_WAIT:
					state[1]	= counter[0] == 5 ? RESP : state[0];
					counter[1]	= counter[0] == 5 ? 0 : counter[0] + 1;
					break;
				case RESP:
					state[1]	= bvalid ? FINISH : state[0];
					break;
				case FINISH:
					state[1]	= IDLE;
					break;
			}
			
			finish		= (state[1] == FINISH);
			awvalid		= (state[1] == WRITE_ADDR);
			awaddr		= awvalid ? reg_addr%256 : 0;
			wvalid		= (state[1] == WRITE_DATA);
			wdata		= wvalid ? reg_data : 0;
			bready		= (state[1] == RESP);
		}
		
		//update reg data
		void update() {
			state[0]		= state[1];
			counter[0]		= counter[1];
		}
};

// -------------- don't touch End Line -------------- Edited by Yi-Chung

void INTEGRATE_IP(
	int dump_lv, 
	uint32_t max_cycle, 
	uint32_t dump_cycle, 
	uint32_t count,
	uint32_t & IP_counter, 
	int mode,
	campose_t pose,
	std::ofstream &os,
	std::ofstream &os_projection) 
{
	std::unique_ptr<VerilatedContext> contextp(new VerilatedContext);
	std::unique_ptr<VIntegrate> IP(new VIntegrate); // init Integrate module

	//dram read/write channel
	dram_ch_r r_ch;
	dram_ch_w w_ch[2];

	//reg read/write channel
	reg_ch_r reg_r;
	reg_ch_w reg_w;

	//waveform generation
	Verilated::traceEverOn(true);
	VerilatedVcdC* tfp = new VerilatedVcdC;

	IP->trace(tfp, dump_lv);
	tfp->open("waveform.vcd");

	IP->clk = 1;	IP->rst_n = 1;	IP->eval();
	contextp->timeInc(1); tfp->dump(contextp->time());

	IP->clk = 0;	IP->rst_n = 0;	IP->eval();
	contextp->timeInc(1); tfp->dump(contextp->time());

	IP->clk = 1;	IP->rst_n = 0;	IP->eval();
	contextp->timeInc(1); tfp->dump(contextp->time());

	IP->clk = 0;	IP->rst_n = 1;	IP->eval();
	contextp->timeInc(1); tfp->dump(contextp->time());

	for(int i=0;i<20;i++) {
		IP->clk = 1;	IP->eval();
		contextp->timeInc(1); tfp->dump(contextp->time());

		IP->clk = 0;	IP->eval();
		contextp->timeInc(1); tfp->dump(contextp->time());
	}

	//initialize
	VlWide<16> tmp = {0};
	for(int i=0;i<16;i++) tmp[i] = i;

	IP->axil_araddr      = 0;
    IP->axil_arvalid     = 0;
    IP->axil_rready      = 0;

    IP->axil_awaddr      = 0;
    IP->axil_awvalid     = 0;
    IP->axil_bready      = 0;
    IP->axil_wdata       = 0;
    IP->axil_wvalid      = 0;
    
    IP->axi_r_arready    = 0;
    IP->axi_r_rvalid     = 0;
    IP->axi_r_rdata      = tmp;

    IP->axi_w0_awready    = 0;
    IP->axi_w0_bvalid     = 0;
    IP->axi_w0_wready     = 0;

	IP->axi_w1_awready    = 0;
    IP->axi_w1_bvalid     = 0;
    IP->axi_w1_wready     = 0;
	//xdma reg read/write (put into que)
	std::vector<w_element> w_que_caller;
	std::vector<r_element> r_que_caller;

	w_element w_tmp;
	r_element r_tmp;

	//put parameters into que
	//w_tmp.addr = 2 * 4;	//destination addr in DRAM
	//w_tmp.data = CONST_IMAGE_BANK;
	//w_que_caller.push_back(w_tmp);

	//w_tmp.addr = 4 * 4;	//Total image blocks count (64 pixels per block)
	//w_tmp.data = 836;
	//w_que_caller.push_back(w_tmp);
/*
	uint32_t tt = ((300*1) << 16) + (170*3);

	w_tmp.addr = 6 * 4;	//Offset Assign
	w_tmp.data = tt;
	w_que_caller.push_back(w_tmp);

	tt = ((300*1) << 16) + (170*2);
	w_tmp.addr = 7 * 4;	//Offset Assign
	w_tmp.data = tt;
	w_que_caller.push_back(w_tmp);
*/
	w_tmp.addr = 0 * 4; //activate
	w_tmp.data = 7; //projection + 3DGS + RGB output
	w_que_caller.push_back(w_tmp);

	if(mode == 0) {	//only rasterize
		w_tmp.addr = 3 * 4;	//total input gaussians count
		w_tmp.data = count;
		w_que_caller.push_back(w_tmp);
	}
	else if(mode == 1) {	//projection + rasterize
		//w_tmp.addr = 5 * 4; //3DGS input address
		//w_tmp.data = CONST_3DGS_INPUT;
		//w_que_caller.push_back(w_tmp);
		
		w_tmp.addr = 9 * 4; //3D gaussians data amount
		w_tmp.data = count;
		w_que_caller.push_back(w_tmp);
		
		uint32_t tmp_buf;

		//w_tmp.addr = 6 * 4;	//Offset Assign
		//w_tmp.data = tt;
		//w_que_caller.push_back(w_tmp);

		tmp_buf = (static_cast<uint32_t>(pose.R[0]) << 16) | static_cast<uint32_t>(pose.R[1]);
		w_tmp.addr = 10 * 4; 
		w_tmp.data = tmp_buf;
		w_que_caller.push_back(w_tmp);
		
		tmp_buf = (static_cast<uint32_t>(pose.R[2]) << 16) | static_cast<uint32_t>(pose.R[3]);
		w_tmp.addr = 11 * 4; 
		w_tmp.data = tmp_buf;
		w_que_caller.push_back(w_tmp);
		
		tmp_buf = (static_cast<uint32_t>(pose.R[4]) << 16) | static_cast<uint32_t>(pose.R[5]);
		w_tmp.addr = 12 * 4; 
		w_tmp.data = tmp_buf;
		w_que_caller.push_back(w_tmp);
		
		tmp_buf = (static_cast<uint32_t>(pose.R[6]) << 16) | static_cast<uint32_t>(pose.R[7]);
		w_tmp.addr = 13 * 4; 
		w_tmp.data = tmp_buf;
		w_que_caller.push_back(w_tmp);
		
		tmp_buf = (static_cast<uint32_t>(pose.R[8]) << 16) | static_cast<uint32_t>(pose.t[0]);
		w_tmp.addr = 14 * 4; 
		w_tmp.data = tmp_buf;
		w_que_caller.push_back(w_tmp);
		
		tmp_buf = (static_cast<uint32_t>(pose.t[1]) << 16) | static_cast<uint32_t>(pose.t[2]);
		w_tmp.addr = 15 * 4; 
		w_tmp.data = tmp_buf;
		w_que_caller.push_back(w_tmp);
		
		tmp_buf = (static_cast<uint32_t>(pose.P[0]) << 16) | static_cast<uint32_t>(pose.P[1]);
		w_tmp.addr = 16 * 4; 
		w_tmp.data = tmp_buf;
		w_que_caller.push_back(w_tmp);
		
		tmp_buf = (static_cast<uint32_t>(pose.P[2]) << 16) | static_cast<uint32_t>(pose.P[3]);
		w_tmp.addr = 17 * 4; 
		w_tmp.data = tmp_buf;
		w_que_caller.push_back(w_tmp);
		
		tmp_buf = (static_cast<uint32_t>(pose.W) << 16) | static_cast<uint32_t>(pose.H);
		w_tmp.addr = 18 * 4; 
		w_tmp.data = tmp_buf;
		w_que_caller.push_back(w_tmp);
		
		tmp_buf = (static_cast<uint32_t>(pose.focal[0]) << 16) | static_cast<uint32_t>(pose.focal[1]);
		w_tmp.addr = 19 * 4; 
		w_tmp.data = tmp_buf;
		w_que_caller.push_back(w_tmp);
				
		tmp_buf = (static_cast<uint32_t>(pose.cam_center[0]) << 16) | static_cast<uint32_t>(pose.cam_center[1]);
		w_tmp.addr = 20 * 4; 
		w_tmp.data = tmp_buf;
		w_que_caller.push_back(w_tmp);		
		
		tmp_buf = static_cast<uint32_t>(pose.cam_center[2]) << 16;
		w_tmp.addr = 21 * 4; 
		w_tmp.data = tmp_buf;
		w_que_caller.push_back(w_tmp);
	}

	w_tmp.addr = 29 * 4; //bank ready 1
	w_tmp.data = 1;
	w_que_caller.push_back(w_tmp);

	uint64_t latency = 0;
	bool w_que_valid = false;
	bool r_que_valid = false;

	bool compute_done = false;

	uint32_t pool_period = 10000;

	std::cerr << "w que size: " << w_que_caller.size() << "\n";

	std::ofstream file_out		("./log/data.txt");

	VlWide<12> d_sel_tmp = {0};

	while(latency++ < max_cycle && !compute_done) {
		if(latency%10000 == 0) std::cerr << "Cycle: " << std::dec << latency << "\n";

		if(!w_que_valid && w_que_caller.size() != 0) {
			w_que_valid = true;
			reg_w.assign(w_que_caller[0].addr, w_que_caller[0].data);
			w_que_caller.erase(w_que_caller.begin());
		}

		if(!r_que_valid && r_que_caller.size() != 0) {
			r_que_valid = true;
			reg_r.assign(r_que_caller[0].addr);
			r_que_caller.erase(r_que_caller.begin());
		}

		if(latency % pool_period == 0 && compute_done == false) {
			r_tmp.addr = 4 * 29;	//read busy signal
			r_que_caller.push_back(r_tmp);
		}

		IP->clk = 1;

		r_ch.eval();
		w_ch[0].eval();
		w_ch[1].eval();

		if(w_que_valid) reg_w.eval();
		if(r_que_valid) reg_r.eval();

		IP->eval();

		contextp->timeInc(1);
        
		if(latency < dump_cycle) tfp->dump(contextp->time());

		IP->clk = 0;

		if(w_que_valid) {
			IP->axil_awaddr    = reg_w.awaddr;
			IP->axil_awvalid   = reg_w.awvalid;
			IP->axil_bready    = reg_w.bready;
			IP->axil_wdata     = reg_w.wdata;
			IP->axil_wvalid    = reg_w.wvalid;
		}
		if(r_que_valid) {
			IP->axil_araddr    = reg_r.araddr;
        	IP->axil_arvalid   = reg_r.arvalid;
        	IP->axil_rready    = reg_r.rready;
		}

		IP->axi_r_arready  = r_ch.arready;
        IP->axi_r_rdata    = r_ch.rdata;
        IP->axi_r_rvalid   = r_ch.rvalid;
		IP->axi_r_rlast    = r_ch.rlast;

        IP->axi_w0_awready  = w_ch[0].awready;
        IP->axi_w0_bvalid   = w_ch[0].bvalid;
        IP->axi_w0_wready   = w_ch[0].wready;

		IP->axi_w1_awready  = w_ch[1].awready;
        IP->axi_w1_bvalid   = w_ch[1].bvalid;
        IP->axi_w1_wready   = w_ch[1].wready;

		IP->eval();

		contextp->timeInc(1);
        if(latency < dump_cycle) tfp->dump(contextp->time());

		if(w_que_valid) reg_w.update();
		if(r_que_valid) reg_r.update();
		r_ch.update();
		w_ch[0].update();
		w_ch[1].update();

		if(w_que_valid) {
			reg_w.awready	= IP->axil_awready;
			reg_w.bvalid	= IP->axil_bvalid;
			reg_w.wready	= IP->axil_wready;
		}
		if(r_que_valid) {
			reg_r.arready   = IP->axil_arready;
        	reg_r.rdata     = IP->axil_rdata;
        	reg_r.rvalid    = IP->axil_rvalid;
		}

		r_ch.araddr		= IP->axi_r_araddr;
		r_ch.arlen		= IP->axi_r_arlen;
		r_ch.arvalid	= IP->axi_r_arvalid;
		r_ch.rdata		= IP->axi_r_rdata;
		r_ch.rready		= IP->axi_r_rready;
		
		w_ch[0].awaddr		= IP->axi_w0_awaddr;
		w_ch[0].awlen		= IP->axi_w0_awlen;
		w_ch[0].awvalid		= IP->axi_w0_awvalid;
		w_ch[0].bready		= IP->axi_w0_bready;
		w_ch[0].wdata		= IP->axi_w0_wdata;
		w_ch[0].wlast		= IP->axi_w0_wlast;
		w_ch[0].wvalid		= IP->axi_w0_wvalid;
        w_ch[0].wstrb      	= IP->axi_w0_wstrb;

		w_ch[1].awaddr		= IP->axi_w1_awaddr;
		w_ch[1].awlen		= IP->axi_w1_awlen;
		w_ch[1].awvalid		= IP->axi_w1_awvalid;
		w_ch[1].bready		= IP->axi_w1_bready;
		w_ch[1].wdata		= IP->axi_w1_wdata;
		w_ch[1].wlast		= IP->axi_w1_wlast;
		w_ch[1].wvalid		= IP->axi_w1_wvalid;
        w_ch[1].wstrb      	= IP->axi_w1_wstrb;

		//watch dog filter
		/*
		bool change = 0;
		for(int k = 0; k < 12; k++) {
			if(d_sel_tmp[k] != IP->d_c0_sel_wire[k]) {
				change = 1;
				d_sel_tmp[k] = IP->d_c0_sel_wire[k];
			}
		}

		if(change) {
			WATCH_DOG_PE_cluster(
				os,
				IP->d_c0_sel_wire,
				IP->cws_out_c0_wire,
				IP->cov_r0_wire,
				IP->opacity_r3_wire,
				IP->colors_r5_wire); 
			}
		//watch dog end

		WATCH_DOG_Projection(
			os_projection,
			IP->tile14_valid,
			IP->RAM_W_qsdata_2d);
		*/

		if(w_que_valid) {
			if(reg_w.finish) {
				reg_w.clear();
				w_que_valid = false;
			}
		}

		if(r_que_valid) {
			if(reg_r.finish) {
				reg_r.clear();
				r_que_valid = false;
			}

			if(reg_r.rready && reg_r.rvalid) { //determine busy signal. if 0, then finish.
				if(reg_r.reg_addr == (4*29) && reg_r.rdata == 0) {
					std::cerr << "Calling Finished!\n";
					std::cerr << "Total Latency: " << std::dec << latency << "\n";
					compute_done = true;

					r_tmp.addr = 4 * 2; //read out counter
					r_que_caller.push_back(r_tmp);
				}

				if(reg_r.reg_addr == 4*2) {	// read out counter
					IP_counter = reg_r.rdata;
				}
			}
		}
	}

	file_out.close();

	std::cerr << "Program Finished.\n";

	tfp->close();
	delete tfp;

	return;
}

void show_fpga_image_sdl_tiled_u32rgb_multi_tile(
    const uint8_t* fpga_dram,
    uint64_t base_addr,
    int tile_width = 300,
    int tile_height = 170,
    int num_tiles_x = 2,
    int scale = 1
) {
    if (!fpga_dram) return;

    constexpr int BLK = 8;
    constexpr int BYTES_PER_PIXEL = 4;   // 32-bit RGB
    constexpr int W_PAD = 304;
    constexpr int H_PAD = 176;
    constexpr int BX_CNT = W_PAD / BLK;  // 38
    constexpr int BY_CNT = H_PAD / BLK;  // 22
    constexpr int PIXELS_PER_BLOCK = 64;
    constexpr int BYTES_PER_BLOCK = PIXELS_PER_BLOCK * BYTES_PER_PIXEL; // 256
    constexpr uint64_t BYTES_PER_TILE = (uint64_t)BX_CNT * BY_CNT * BYTES_PER_BLOCK; // 214016

    const int out_width  = tile_width * num_tiles_x;
    const int out_height = tile_height;

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::fprintf(stderr, "SDL_Init error: %s\n", SDL_GetError());
        return;
    }

    const int winW = out_width * scale;
    const int winH = out_height * scale;

    SDL_Window* win = SDL_CreateWindow(
        "Verilator Packed RGB Viewer (Multi Tile)",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        winW, winH, 0
    );

    SDL_Renderer* ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);

    SDL_Texture* tex = SDL_CreateTexture(
        ren,
        SDL_PIXELFORMAT_RGBA32,
        SDL_TEXTUREACCESS_STREAMING,
        out_width, out_height
    );

    std::vector<uint8_t> rgba((size_t)out_width * (size_t)out_height * 4, 0);

    auto update = [&]() {
        std::fill(rgba.begin(), rgba.end(), 0);

        for (int tile_id = 0; tile_id < num_tiles_x; ++tile_id) {
            uint64_t tile_base = base_addr + (uint64_t)tile_id * BYTES_PER_TILE;

            for (int by = 0; by < BY_CNT; ++by) {
                for (int bx = 0; bx < BX_CNT; ++bx) {
                    uint64_t block_idx = (uint64_t)by * BX_CNT + bx;
                    const uint8_t* block_ptr =
                        fpga_dram + tile_base + block_idx * BYTES_PER_BLOCK;

                    for (int iy = 0; iy < BLK; ++iy) {
                        for (int ix = 0; ix < BLK; ++ix) {
                            int local_x = bx * BLK + ix;
                            int local_y = by * BLK + iy;

                            if (local_x >= tile_width || local_y >= tile_height) continue;

                            int out_x = tile_id * tile_width + local_x;
                            int out_y = local_y;

                            int bank = iy * BLK + ix;
                            const uint8_t* px = block_ptr + bank * BYTES_PER_PIXEL;

                            // memory layout:
                            // px[0] = B
                            // px[1] = G
                            // px[2] = R
                            // px[3] = unused
                            uint8_t B = px[0];
                            uint8_t G = px[1];
                            uint8_t R = px[2];

                            size_t idx = ((size_t)out_y * (size_t)out_width + (size_t)out_x) * 4;
                            rgba[idx + 0] = R;
                            rgba[idx + 1] = G;
                            rgba[idx + 2] = B;
                            rgba[idx + 3] = 255;
                        }
                    }
                }
            }
        }

        SDL_UpdateTexture(tex, nullptr, rgba.data(), out_width * 4);
    };

    update();

    bool running = true;
    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) running = false;
            if (e.type == SDL_KEYDOWN) {
                if (e.key.keysym.sym == SDLK_r) {
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


void show_fpga_image_sdl_tiled_u32rgb_2x2tiles(
    const uint8_t* fpga_dram,
    uint64_t base_addr,
    int tile_width = 300,
    int tile_height = 170,
    int scale = 3
) {
    if (!fpga_dram) return;

    constexpr int BLK = 8;
    constexpr int BYTES_PER_PIXEL = 4;   // 32-bit RGB
    constexpr int W_PAD = 304;
    constexpr int H_PAD = 176;
    constexpr int BX_CNT = W_PAD / BLK;  // 38
    constexpr int BY_CNT = H_PAD / BLK;  // 22
    constexpr int PIXELS_PER_BLOCK = 64;
    constexpr int BYTES_PER_BLOCK = PIXELS_PER_BLOCK * BYTES_PER_PIXEL; // 256
    constexpr uint64_t BYTES_PER_TILE =
        (uint64_t)BX_CNT * (uint64_t)BY_CNT * (uint64_t)BYTES_PER_BLOCK; // 214016

    constexpr int TILES_X = 2;
    constexpr int TILES_Y = 2;

    const int out_width  = tile_width * TILES_X;   // 600
    const int out_height = tile_height * TILES_Y;  // 340

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::fprintf(stderr, "SDL_Init error: %s\n", SDL_GetError());
        return;
    }

    const int winW = out_width * scale;
    const int winH = out_height * scale;

    SDL_Window* win = SDL_CreateWindow(
        "Verilator Packed RGB Viewer (2x2 Tiles)",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        winW, winH, 0
    );

    SDL_Renderer* ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);

    SDL_Texture* tex = SDL_CreateTexture(
        ren,
        SDL_PIXELFORMAT_RGBA32,
        SDL_TEXTUREACCESS_STREAMING,
        out_width, out_height
    );

    std::vector<uint8_t> rgba((size_t)out_width * (size_t)out_height * 4, 0);

    auto update = [&]() {
        std::fill(rgba.begin(), rgba.end(), 0);

        for (int tile_y = 0; tile_y < TILES_Y; ++tile_y) {
            for (int tile_x = 0; tile_x < TILES_X; ++tile_x) {
                int tile_id = tile_y * TILES_X + tile_x;

                uint64_t tile_base = base_addr + (uint64_t)tile_id * BYTES_PER_TILE;

                for (int by = 0; by < BY_CNT; ++by) {
                    for (int bx = 0; bx < BX_CNT; ++bx) {
                        uint64_t block_idx = (uint64_t)by * BX_CNT + bx;
                        const uint8_t* block_ptr =
                            fpga_dram + tile_base + block_idx * BYTES_PER_BLOCK;

                        for (int iy = 0; iy < BLK; ++iy) {
                            for (int ix = 0; ix < BLK; ++ix) {
                                int local_x = bx * BLK + ix;
                                int local_y = by * BLK + iy;

                                if (local_x >= tile_width || local_y >= tile_height) continue;

                                int out_x = tile_x * tile_width + local_x;
                                int out_y = tile_y * tile_height + local_y;

                                int bank = iy * BLK + ix;
                                const uint8_t* px = block_ptr + bank * BYTES_PER_PIXEL;

                                // 記憶體格式:
                                // px[0] = B
                                // px[1] = G
                                // px[2] = R
                                // px[3] = unused
                                uint8_t B = px[0];
                                uint8_t G = px[1];
                                uint8_t R = px[2];

                                size_t idx =
                                    ((size_t)out_y * (size_t)out_width + (size_t)out_x) * 4;

                                rgba[idx + 0] = R;
                                rgba[idx + 1] = G;
                                rgba[idx + 2] = B;
                                rgba[idx + 3] = 255;
                            }
                        }
                    }
                }
            }
        }

        SDL_UpdateTexture(tex, nullptr, rgba.data(), out_width * 4);
    };

    update();

    bool running = true;
    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) running = false;

            if (e.type == SDL_KEYDOWN) {
                if (e.key.keysym.sym == SDLK_r) {
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

void show_fpga_image_sdl_tiled_u32rgb(
    const uint8_t* fpga_dram,
    uint64_t base_addr,
    int width_org = 300,
    int height_org = 170,
    int scale = 1
) {
    if (!fpga_dram) return;

    constexpr int BLK = 8;
    constexpr int BYTES_PER_PIXEL = 4;   // 新格式：32-bit / pixel
    constexpr int W_PAD = 304;
    constexpr int H_PAD = 176;
    constexpr int BX_CNT = W_PAD / BLK;  // 38
    constexpr int BY_CNT = H_PAD / BLK;  // 22
    constexpr int PIXELS_PER_BLOCK = 64;
    constexpr int BYTES_PER_BLOCK = PIXELS_PER_BLOCK * BYTES_PER_PIXEL; // 256

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::fprintf(stderr, "SDL_Init error: %s\n", SDL_GetError());
        return;
    }

    const int winW = width_org * scale;
    const int winH = height_org * scale;

    SDL_Window* win = SDL_CreateWindow(
        "Verilator Packed RGB Viewer (32bpp)",
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
                const uint8_t* block_ptr =
                    fpga_dram + base_addr + block_idx * BYTES_PER_BLOCK;

                for (int iy = 0; iy < BLK; ++iy) {
                    for (int ix = 0; ix < BLK; ++ix) {
                        int x = bx * BLK + ix;
                        int y = by * BLK + iy;

                        if (x >= width_org || y >= height_org) continue;

                        int bank = iy * BLK + ix;
                        const uint8_t* px = block_ptr + bank * BYTES_PER_PIXEL;

                        // 你的格式：
                        // px[0] = B
                        // px[1] = G
                        // px[2] = R
                        // px[3] = unused (0)
                        uint8_t B = px[0];
                        uint8_t G = px[1];
                        uint8_t R = px[2];

                        size_t idx = ((size_t)y * (size_t)width_org + (size_t)x) * 4;
                        rgba[idx + 0] = R;
                        rgba[idx + 1] = G;
                        rgba[idx + 2] = B;
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
                if (e.key.keysym.sym == SDLK_r) {
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

int main(int argc, char** argv)
{
	srand(time(NULL));
	int dump_lv = 99;
	uint32_t max_cycle = 100000;
	uint32_t dump_cycle = max_cycle;
	uint32_t count = 0;
	std::string input_file, input_pose, input_count;
	int mode = 0;	//mode 0 = 2DGS, mode 1 = 3DGS

    for(int i=0;i<argc;i++) {
        std::string argv_str(argv[i]);
		if(argv_str == "-in")			input_file 		= argv[i+1];
		if(argv_str == "-in_pose")		input_pose 		= argv[i+1];
		if(argv_str == "-in_count")		input_count		= argv[i+1];
		if(argv_str == "-mode")			mode	 		= std::stoi(argv[i+1]);
		if(argv_str == "-dump_lv") 		dump_lv 		= std::stoi(argv[i+1]);
		if(argv_str == "-max_cycle") 	max_cycle 		= std::stoul(argv[i+1]);
		if(argv_str == "-dump_cycle")	dump_cycle 		= std::stoul(argv[i+1]);
    }

	fpga_dram.reset(new uint8_t[DRAM_SIZE]);
	campose_t pose;

	if(mode == 0) { //mode 0 == 2DGS
		std::ifstream fin("tile23/Count.txt");
		if(!fin) {
			std::cerr << "file err.\n";
			return 1;
		}
		
		fin >> count;

		std::ifstream fin_mean2D("tile23/Mean2D.txt");
		if(!fin_mean2D) {
			std::cerr << "fin_mean2D err.\n";
			return 1;
		}

		std::ifstream fin_Cov2D("tile23/Cov2D_inv.txt");
		if(!fin_Cov2D) {
			std::cerr << "fin_Cov2D err.\n";
			return 1;
		}

		std::ifstream fin_Opacity("tile23/Opacity.txt");
		if(!fin_Opacity) {
			std::cerr << "fin_Opacity err.\n";
			return 1;
		}

		std::ifstream fin_Color("tile23/Color.txt");
		if(!fin_Color) {
			std::cerr << "fin_Color err.\n";
			return 1;
		}

		std::ifstream fin_Radii("tile23/Radii.txt");
		if(!fin_Radii) {
			std::cerr << "fin_Radii err.\n";
			return 1;
		}

		VlWide<8> data = {0};

		//load 2D tile parameters into DRAM
		for(uint32_t i=0;i<count;i++) {
			uint32_t tmp1, tmp2, tmp3;
			fin_mean2D >> std::dec >> tmp1 >> tmp2;

			//tile23
			//tmp1 -= 300*2;
			//tmp2 -= 170*3;

			fpga_dram[32*i + 21] = tmp1 >> 8;
			fpga_dram[32*i + 20] = tmp1;
			fpga_dram[32*i + 19] = tmp2 >> 8;
			fpga_dram[32*i + 18] = tmp2;

			fin_Cov2D >> std::hex >> tmp1 >> tmp2 >> tmp3;
			fpga_dram[32*i + 17] = tmp1 >> 8;
			fpga_dram[32*i + 16] = tmp1;
			fpga_dram[32*i + 15] = tmp2 >> 8;
			fpga_dram[32*i + 14] = tmp2;
			fpga_dram[32*i + 13] = tmp3 >> 8;
			fpga_dram[32*i + 12] = tmp3;

			fin_Opacity >> std::hex >> tmp1;
			fpga_dram[32*i + 11] = tmp1 >> 8;
			fpga_dram[32*i + 10] = tmp1;

			fin_Color >> std::hex >> tmp1 >> tmp2 >> tmp3;
			fpga_dram[32*i + 9] = tmp1 >> 8;
			fpga_dram[32*i + 8] = tmp1;
			fpga_dram[32*i + 7] = tmp2 >> 8;
			fpga_dram[32*i + 6] = tmp2;
			fpga_dram[32*i + 5] = tmp3 >> 8;
			fpga_dram[32*i + 4] = tmp3;

			fin_Radii >> std::dec >> tmp1 >> tmp2;
			fpga_dram[32*i + 3] = tmp1 >> 8;
			fpga_dram[32*i + 2] = tmp1;
			fpga_dram[32*i + 1] = tmp2 >> 8;
			fpga_dram[32*i + 0] = tmp2;
		}

		std::cerr << std::dec << "Count: " << count << "\n";

	}
	else {
		std::ifstream fin(input_count);
		if(!fin) {
			std::cerr << "file err.\n";
			return 1;
		}
		fin >> count;
		
	    if (!load_hex_lines_to_fpga_dram(input_file, CONST_3DGS_INPUT, count)) {
    	    return 1;
    	}
		pose = read_campose_from_file(input_pose);
	}

	uint32_t IP_counter;

	std::ofstream dog_pe("./log/pe_cluster.txt");
	std::ofstream dog_projection("./log/tile_2dgs.txt");

	std::cerr << "Total 3D Gaussians Count: " << count << "\n";

	INTEGRATE_IP(
		dump_lv, 
		max_cycle, 
		dump_cycle, 
		count,
		IP_counter,
		mode, 
		pose,
		dog_pe,
		dog_projection
	);

	dog_pe.close();
	dog_projection.close();

	std::cerr << "IP counter: " << IP_counter << "\n";
	
	/*
	show_fpga_image_sdl_tiled_u32rgb(
    fpga_dram.get(),
    CONST_IMAGE_BANK,
    300,
    170,
    1
	);
	*/
	/*
	show_fpga_image_sdl_tiled_u32rgb(
    fpga_dram.get(),
    CONST_IMAGE_BANK + 214016,
    300,
    170,
    1
	);
	*/

	
	show_fpga_image_sdl_tiled_u32rgb_2x2tiles(
	    fpga_dram.get(),
	    CONST_IMAGE_BANK,
	    300,
	    170,
	    1
	);
	
	return 0;
}
