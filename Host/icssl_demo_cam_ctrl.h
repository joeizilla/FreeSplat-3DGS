#pragma once

#include <SDL2/SDL.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <cstring>
#include <iostream>

namespace demo3dgs {

using Vec3 = std::array<float, 3>;
using Vec4 = std::array<float, 4>;
using Mat3 = std::array<std::array<float, 3>, 3>;
using Mat4 = std::array<std::array<float, 4>, 4>;

struct CameraPoseFP16 {
    std::array<uint16_t, 9> R;
    std::array<uint16_t, 3> T;
    std::array<uint16_t, 3> C;
};

struct TrajRecord {
    Mat4 M_w2c;
    Vec3 cam_center;
};

static inline uint16_t float_to_fp16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));

    uint32_t sign = (x >> 16) & 0x8000;
    int exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;

    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7C00;

    return sign | (exp << 10) | (mant >> 13);
}

static inline float fp16_to_float(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp  = (h & 0x7C00u) >> 10;
    uint32_t frac = (h & 0x03FFu);

    uint32_t f_bits = 0;

    if (exp == 0) {
        if (frac == 0) {
            // zero
            f_bits = sign;
        } else {
            // subnormal
            exp = 1;
            while ((frac & 0x0400u) == 0) {
                frac <<= 1;
                --exp;
            }
            frac &= 0x03FFu;
            const uint32_t exp32 = (exp + (127 - 15)) << 23;
            const uint32_t frac32 = frac << 13;
            f_bits = sign | exp32 | frac32;
        }
    } else if (exp == 0x1Fu) {
        // inf / nan
        f_bits = sign | 0x7F800000u | (frac << 13);
    } else {
        // normal
        const uint32_t exp32 = (exp + (127 - 15)) << 23;
        const uint32_t frac32 = frac << 13;
        f_bits = sign | exp32 | frac32;
    }

    float out;
    std::memcpy(&out, &f_bits, sizeof(out));
    return out;
}

struct CameraPose {
    Mat3 R_w2c;
    Vec3 T_w2c;
    Vec3 camera_center;
};

static inline Vec3 normalize(const Vec3& v) {
    const float n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (n < 1e-12f) return v;
    return {v[0] / n, v[1] / n, v[2] / n};
}

static inline Mat3 transpose3(const Mat3& m) {
    Mat3 t{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            t[i][j] = m[j][i];
        }
    }
    return t;
}

static inline Mat4 identity4() {
    return {{{1, 0, 0, 0},
             {0, 1, 0, 0},
             {0, 0, 1, 0},
             {0, 0, 0, 1}}};
}

static inline Mat3 mat3_mul(const Mat3& a, const Mat3& b) {
    Mat3 c{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            c[i][j] = 0.0f;
            for (int k = 0; k < 3; ++k) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return c;
}

static inline Mat3 axis_angle_to_rot(const Vec3& axis_in, float angle) {
    const Vec3 axis = normalize(axis_in);
    const float x = axis[0], y = axis[1], z = axis[2];
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    const float C = 1.0f - c;
    return {{{c + x * x * C,     x * y * C - z * s, x * z * C + y * s},
             {y * x * C + z * s, c + y * y * C,     y * z * C - x * s},
             {z * x * C - y * s, z * y * C + x * s, c + z * z * C}}};
}


static inline TrajRecord read_traj_line_as_matrix(const std::string& path, int line_idx = 0) {
    std::ifstream fin(path);
    if (!fin) {
        throw std::runtime_error("Failed to open traj file: " + path);
    }

    std::string line;
    int i = 0;
    while (std::getline(fin, line)) {
        if (i == line_idx) {
            std::istringstream iss(line);
            std::string token;
            std::vector<uint16_t> vals_u16;

            while (std::getline(iss, token, '_')) {
                vals_u16.push_back(static_cast<uint16_t>(std::stoul(token, nullptr, 16)));
            }

            if (vals_u16.size() != 23) {
                throw std::runtime_error("traj line does not have 23 hex values");
            }

            Mat4 mat{};
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    mat[r][c] = 0.0f;
                }
            }

            // R
            for (int k = 0; k < 9; ++k) {
                mat[k / 3][k % 3] = fp16_to_float(vals_u16[k]);
            }

            // t
            mat[0][3] = fp16_to_float(vals_u16[9]);
            mat[1][3] = fp16_to_float(vals_u16[10]);
            mat[2][3] = fp16_to_float(vals_u16[11]);

            // homogeneous row
            mat[3][0] = 0.0f;
            mat[3][1] = 0.0f;
            mat[3][2] = 0.0f;
            mat[3][3] = 1.0f;

            // cam_center 在最後 3 個
            Vec3 cam_center = {
                fp16_to_float(vals_u16[20]),
                fp16_to_float(vals_u16[21]),
                fp16_to_float(vals_u16[22])
            };

            return {mat, cam_center};
        }
        ++i;
    }

    throw std::runtime_error("traj file has fewer lines than requested");
}
/*
static inline Mat4 read_traj_line_as_matrix(const std::string& path, int line_idx = 0) {
    std::ifstream fin(path);
    if (!fin) {
        throw std::runtime_error("Failed to open traj file: " + path);
    }

    std::string line;
    int i = 0;
    while (std::getline(fin, line)) {
        if (i == line_idx) {
            std::istringstream iss(line);
            std::vector<float> vals;
            float x;
            while (iss >> x) vals.push_back(x);
            if (vals.size() != 16) {
                throw std::runtime_error("traj line does not have 16 floats");
            }
            Mat4 mat{};
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    mat[r][c] = vals[r * 4 + c];
                }
            }
            return mat;
        }
        ++i;
    }
    throw std::runtime_error("traj file has fewer lines than requested");
}
*/

static inline Mat4 inverse4_rigid(const Mat4& m) {
    Mat3 R{};
    Vec3 t{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) R[i][j] = m[i][j];
        t[i] = m[i][3];
    }
    const Mat3 Rt = transpose3(R);
    const Vec3 tinv = {
        -(Rt[0][0] * t[0] + Rt[0][1] * t[1] + Rt[0][2] * t[2]),
        -(Rt[1][0] * t[0] + Rt[1][1] * t[1] + Rt[1][2] * t[2]),
        -(Rt[2][0] * t[0] + Rt[2][1] * t[1] + Rt[2][2] * t[2])
    };

    Mat4 out = identity4();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) out[i][j] = Rt[i][j];
        out[i][3] = tinv[i];
    }
    return out;
}

static inline std::tuple<Mat3, Vec3, Vec3>
extract_R_T_from_traj(const std::string& path, int line = 0, bool flip_yz = false) {
    TrajRecord rec = read_traj_line_as_matrix(path, line);
    Mat4 M_w2c = rec.M_w2c;
    Vec3 delta = rec.cam_center;

    if (flip_yz) {
        for (int r = 0; r < 3; ++r) {
            M_w2c[r][1] *= -1.0f;
            M_w2c[r][2] *= -1.0f;
        }
        delta[1] *= -1.0f;
        delta[2] *= -1.0f;
    }

    Mat3 R_w2c{};
    Vec3 T_w2c{};

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            R_w2c[r][c] = M_w2c[r][c];
        }
        T_w2c[r] = M_w2c[r][3];
    }

    return {R_w2c, T_w2c, delta};
}

// static inline std::tuple<Mat3, Vec3, Mat3, Vec3>
// extract_R_T_from_traj(const std::string& path, int line = 0, bool flip_yz = false) {
//     Mat4 M_c2w = read_traj_line_as_matrix(path, line);

//     if (flip_yz) {
//         for (int r = 0; r < 3; ++r) {
//             M_c2w[r][1] *= -1.0f;
//             M_c2w[r][2] *= -1.0f;
//         }
//     }

//     Mat3 R_c2w{};
//     Vec3 T_c2w{};
//     for (int r = 0; r < 3; ++r) {
//         for (int c = 0; c < 3; ++c) R_c2w[r][c] = M_c2w[r][c];
//         T_c2w[r] = M_c2w[r][3];
//     }

//     // const Mat4 M_w2c = inverse4_rigid(M_c2w);
//     // Mat3 R_w2c{};
//     // Vec3 T_w2c{};
//     // for (int r = 0; r < 3; ++r) {
//     //     for (int c = 0; c < 3; ++c) R_w2c[r][c] = M_w2c[r][c];
//     //     T_w2c[r] = M_w2c[r][3];
//     // }

//     return {R_c2w, T_c2w};
// }

class CameraController {
public:
    CameraController(const std::string& traj_path,
                    int line = 0,
                    bool flip_yz = false,
                    float move_speed = 1.0f,
                    float mouse_sens = 0.0001f,
                    float roll_speed = 0.5f,
                    float key_rot_speed = 0.5f)
        : move_speed_(move_speed),
        mouse_sens_(mouse_sens),
        roll_speed_(roll_speed),
        key_rot_speed_(key_rot_speed) {
        auto [R_w2c, T_w2c, delta] = extract_R_T_from_traj(traj_path, line, flip_yz);
        R_w2c_ = R_w2c;
        T_w2c_ = T_w2c;
        // delta_ = delta;
    }
    // CameraController(const std::string& traj_path,
    //                  int line = 0,
    //                  bool flip_yz = false,
    //                  float move_speed = 1.5f,
    //                  float mouse_sens = 2.0f,
    //                  float roll_speed = 1.2f,
    //                  float key_rot_speed = 1.6f)
    //     : move_speed_(move_speed),
    //       mouse_sens_(mouse_sens),
    //       roll_speed_(roll_speed),
    //       key_rot_speed_(key_rot_speed) {
    //     auto [R_c2w, T_c2w, R_w2c, T_w2c] = extract_R_T_from_traj(traj_path, line, flip_yz);
    //     (void)R_c2w;
    //     (void)T_c2w;
    //     R_w2c_ = R_w2c;
    //     T_w2c_ = T_w2c;
    // }

    bool process_input(float dt, bool& running) {
        bool pose_changed = false;

        const Uint8* keys = SDL_GetKeyboardState(nullptr);
        const float speed_mul = (keys[SDL_SCANCODE_LSHIFT] || keys[SDL_SCANCODE_RSHIFT]) ? 3.0f : 1.0f;
        const float move = move_speed_ * dt * speed_mul;
        const float rot_roll = roll_speed_ * dt * speed_mul;
        const float rot_key = key_rot_speed_ * dt * speed_mul;
        const float mouse_sens = mouse_sens_ * speed_mul;

        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            } else if (event.type == SDL_KEYDOWN) {
                if (event.key.keysym.sym == SDLK_ESCAPE) {
                    running = false;
                }
            } else if (event.type == SDL_MOUSEBUTTONDOWN && event.button.button == SDL_BUTTON_RIGHT) {
                toggle_mouse_capture();
            } else if (event.type == SDL_MOUSEMOTION && mouse_captured_) {
                const int dx = event.motion.xrel;
                const int dy = event.motion.yrel;
                const float yaw = static_cast<float>(dx) * mouse_sens;
                const float pitch = -static_cast<float>(dy) * mouse_sens;
                rotate_local(yaw, pitch, 0.0f);
                pose_changed = true;
                std::cerr << "mouse_captured_: " << mouse_captured_ << "\n";
            } else if (event.type == SDL_MOUSEWHEEL) {
                if (event.wheel.y < 0) {
                    scale_ *= 1.1f;
                } else if (event.wheel.y > 0) {
                    scale_ /= 1.1f;
                }

                if (scale_ < 0.1f) scale_ = 0.1f;
                if (scale_ > 20.0f) scale_ = 20.0f;

                pose_changed = true;
            }
            
        }

        float dx = 0.0f, dy = 0.0f, dz = 0.0f;
        if (keys[SDL_SCANCODE_Q]) dy -= move;
        if (keys[SDL_SCANCODE_E]) dy += move;
        if (keys[SDL_SCANCODE_A]) dx -= move;
        if (keys[SDL_SCANCODE_D]) dx += move;
        if (keys[SDL_SCANCODE_W]) dz += move;
        if (keys[SDL_SCANCODE_S]) dz -= move;
        if (dx != 0.0f || dy != 0.0f || dz != 0.0f) {
            move_local(dx, dy, dz);
            pose_changed = true;
        }

        float yaw = 0.0f, pitch = 0.0f, roll = 0.0f;
        if (keys[SDL_SCANCODE_RIGHT]) yaw += rot_key;
        if (keys[SDL_SCANCODE_LEFT])  yaw -= rot_key;
        if (keys[SDL_SCANCODE_UP])    pitch += rot_key;
        if (keys[SDL_SCANCODE_DOWN])  pitch -= rot_key;
        if (keys[SDL_SCANCODE_Z])     roll -= rot_roll;
        if (keys[SDL_SCANCODE_C])     roll += rot_roll;
        if (yaw != 0.0f || pitch != 0.0f || roll != 0.0f) {
            rotate_local(yaw, pitch, roll);
            pose_changed = true;
        }

        return pose_changed;
    }

    // CameraPose pose() const {
    //     return {R_w2c_, T_w2c_, camera_center()};
    // }

    CameraPose pose() const {
        Vec3 T_scaled = scaled_T_w2c();
        Mat3 Rt = transpose3(R_w2c_);

        Vec3 C_scaled = {
            -(Rt[0][0] * T_scaled[0] + Rt[0][1] * T_scaled[1] + Rt[0][2] * T_scaled[2]),
            -(Rt[1][0] * T_scaled[0] + Rt[1][1] * T_scaled[1] + Rt[1][2] * T_scaled[2]),
            -(Rt[2][0] * T_scaled[0] + Rt[2][1] * T_scaled[1] + Rt[2][2] * T_scaled[2])
        };

        return {R_w2c_, T_scaled, C_scaled};
    }

    static inline CameraPoseFP16 to_fp16(const CameraPose& p) {
        CameraPoseFP16 out{};

        // R
        for (int i = 0; i < 9; i++) {
            out.R[i] = float_to_fp16(p.R_w2c[i / 3][i % 3]);
        }

        // T
        for (int i = 0; i < 3; i++) {
            out.T[i] = float_to_fp16(p.T_w2c[i]);
        }

        // camera center
        for (int i = 0; i < 3; i++) {
            out.C[i] = float_to_fp16(p.camera_center[i]);
        }

        return out;
    }

    const Mat3& R_w2c() const { return R_w2c_; }
    const Vec3& T_w2c() const { return T_w2c_; }
    bool mouse_captured() const { return mouse_captured_; }

private:
    Mat3 R_c2w() const {
        return transpose3(R_w2c_);
    }

    Vec3 camera_center() const {
        const Mat3 Rt = transpose3(R_w2c_);
        return {
            -(Rt[0][0] * T_w2c_[0] + Rt[0][1] * T_w2c_[1] + Rt[0][2] * T_w2c_[2]),
            -(Rt[1][0] * T_w2c_[0] + Rt[1][1] * T_w2c_[1] + Rt[1][2] * T_w2c_[2]),
            -(Rt[2][0] * T_w2c_[0] + Rt[2][1] * T_w2c_[1] + Rt[2][2] * T_w2c_[2])
        };
    }

    void move_local(float dx, float dy, float dz) {
        const Mat3 Rc2w = R_c2w();
        const Vec3 right   = {Rc2w[0][0], Rc2w[1][0], Rc2w[2][0]};
        const Vec3 up      = {Rc2w[0][1], Rc2w[1][1], Rc2w[2][1]};
        const Vec3 forward = {Rc2w[0][2], Rc2w[1][2], Rc2w[2][2]};

        const Vec3 delta_world = {
            right[0] * dx + up[0] * dy + forward[0] * dz,
            right[1] * dx + up[1] * dy + forward[1] * dz,
            right[2] * dx + up[2] * dy + forward[2] * dz
        };

        const Vec3 C = camera_center();
        const Vec3 C_new = {C[0] + delta_world[0], C[1] + delta_world[1], C[2] + delta_world[2]};
        T_w2c_ = {
            -(R_w2c_[0][0] * C_new[0] + R_w2c_[0][1] * C_new[1] + R_w2c_[0][2] * C_new[2]),
            -(R_w2c_[1][0] * C_new[0] + R_w2c_[1][1] * C_new[1] + R_w2c_[1][2] * C_new[2]),
            -(R_w2c_[2][0] * C_new[0] + R_w2c_[2][1] * C_new[1] + R_w2c_[2][2] * C_new[2])
        };
    }

    void rotate_local(float yaw, float pitch, float roll) {
        const Vec3 C = camera_center();
        Mat3 Rc2w = R_c2w();
        if (std::fabs(yaw) > 0.0f)   Rc2w = mat3_mul(Rc2w, axis_angle_to_rot({0.0f, 1.0f, 0.0f}, yaw));
        if (std::fabs(pitch) > 0.0f) Rc2w = mat3_mul(Rc2w, axis_angle_to_rot({1.0f, 0.0f, 0.0f}, pitch));
        if (std::fabs(roll) > 0.0f)  Rc2w = mat3_mul(Rc2w, axis_angle_to_rot({0.0f, 0.0f, 1.0f}, roll));
        R_w2c_ = transpose3(Rc2w);
        T_w2c_ = {
            -(R_w2c_[0][0] * C[0] + R_w2c_[0][1] * C[1] + R_w2c_[0][2] * C[2]),
            -(R_w2c_[1][0] * C[0] + R_w2c_[1][1] * C[1] + R_w2c_[1][2] * C[2]),
            -(R_w2c_[2][0] * C[0] + R_w2c_[2][1] * C[1] + R_w2c_[2][2] * C[2])
        };
    }

    void toggle_mouse_capture() {
        mouse_captured_ = !mouse_captured_;
        SDL_SetRelativeMouseMode(mouse_captured_ ? SDL_TRUE : SDL_FALSE);
        SDL_ShowCursor(mouse_captured_ ? SDL_DISABLE : SDL_ENABLE);
    }

    Vec3 scaled_T_w2c() const {
        Vec3 Rd = {
            R_w2c_[0][0] * delta_[0] + R_w2c_[0][1] * delta_[1] + R_w2c_[0][2] * delta_[2],
            R_w2c_[1][0] * delta_[0] + R_w2c_[1][1] * delta_[1] + R_w2c_[1][2] * delta_[2],
            R_w2c_[2][0] * delta_[0] + R_w2c_[2][1] * delta_[1] + R_w2c_[2][2] * delta_[2]
        };

        return {
            scale_ * (T_w2c_[0] - Rd[0]),
            scale_ * (T_w2c_[1] - Rd[1]),
            scale_ * (T_w2c_[2] - Rd[2])
        };
    }

private:
    Mat3 R_w2c_{};
    Vec3 T_w2c_{};
    bool mouse_captured_ = false;
    float move_speed_ = 0.7f;
    float mouse_sens_ = 0.00001f;
    float roll_speed_ = 0.6f;
    float key_rot_speed_ = 0.8f;
    float scale_ = 1.0f;
    Vec3 delta_ = {0.0f, 0.0f, 0.0f};
};

} // namespace demo3dgs
