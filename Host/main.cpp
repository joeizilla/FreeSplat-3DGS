#include <SDL2/SDL.h>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include "icssl_GS.h"
#include <chrono>
#include "icssl_demo_cam_ctrl.h"

// Space Definition
#define CONST_2DGS_INPUT       0x000000000ull
#define CONST_3DGS_INPUT       0x040000000ull
#define CONST_IMAGE_BANK       0x080000000ull

int main(int argc, char** argv)
{
    std::string input_file;
    int mode = 1;
    int auto_run = 0;
    int demo_ctrl = 0;

    for (int i = 1; i < argc; ++i) {
        std::string s(argv[i]);
        if (s == "-in" && i + 1 < argc) {
            input_file = argv[++i];
        }
        else if (s == "-mode" && i + 1 < argc) {
            mode = std::stoi(argv[++i]);
        }
        else if (s == "-auto" && i + 1 < argc) {
            auto_run = std::stoi(argv[++i]);
        }
        else if (s == "-demo" && i + 1 < argc) { // For demo control camera with keyboard and mouse
            demo_ctrl = std::stoi(argv[++i]);
        }
    }

    std::string init_pos_file_path = input_file + "/campose_input_all.txt";

    auto poses = read_campose_vector_from_file(init_pos_file_path);
    if (poses.empty()) {
        std::cerr << "No poses loaded.\n";
        return -1;
    }

    fpga_context_t ctx;
    fpga_sdl_viewer_t viewer;
    std::vector<uint8_t> out_buf;

    if (!init_fpga(input_file,
                   ctx,
                   poses[0],
                   CONST_2DGS_INPUT,
                   CONST_3DGS_INPUT,
                   CONST_IMAGE_BANK,
                   mode,
                   fpga_output_format_t::PACKED_BGR888)) {
        std::cerr << "init_fpga failed\n";
        return -1;
    }

    if (!init_fpga_image_sdl_tiled_2x2(viewer, 300, 170, false, 1, fpga_output_format_t::PACKED_BGR888)) {
        std::cerr << "init viewer failed\n";
        close_fpga(ctx);
        return -1;
    }

    // SDL_MaximizeWindow(viewer.window);
    
    // For demo control camera with keyboard and mouse
    if (demo_ctrl==1) std::cout << "Demo control mode enabled. Use keyboard and mouse to control the camera.\n";
    demo3dgs::CameraController demo_cam_ctrl(init_pos_file_path, 0, false);


    bool running = true;
    size_t frame_idx = 0;
    bool need_render = true;

    // For demo control camera with keyboard and mouse
    uint32_t last = SDL_GetTicks();

    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) {
                running = false;
            }
            //else 
            else if (e.type == SDL_KEYDOWN && demo_ctrl == 0) { // Only process keyboard events for frame control when not in demo control mode
                switch (e.key.keysym.sym) {
                case SDLK_SPACE:
                case SDLK_RIGHT:
                    if (frame_idx + 1 < poses.size()) {
                        frame_idx++;
                        need_render = true;
                    }
                    break;

                case SDLK_LEFT:
                    if (frame_idx > 0) {
                        frame_idx--;
                        need_render = true;
                    }
                    break;

                case SDLK_g:
                    if (viewer.out_fmt == fpga_output_format_t::FP16_RGBA_DIV) {
                        viewer.apply_gamma = !viewer.apply_gamma;
                        need_render = true;
                    }
                    break;

                case SDLK_ESCAPE:
                    running = false;
                    break;

                default:
                    break;
                }
            }
        }

        if (demo_ctrl==1) { // For demo control camera with keyboard and mouse
            uint32_t now = SDL_GetTicks();
            float dt = (now - last) / 2000.0f;
            last = now;

            bool changed = demo_cam_ctrl.process_input(dt, running);
            // std::cerr << "changed: " << changed << "\n";
            if (changed) {
                auto pose_float = demo_cam_ctrl.pose();
                // convert to fp16 
                auto pose_fp16 = demo3dgs::CameraController::to_fp16(pose_float);

                // std::cerr << "\n=== Camera Updated ===\n";
                poses[0].R = pose_fp16.R;
                poses[0].t = pose_fp16.T;
                poses[0].cam_center = pose_fp16.C;

                need_render = true;
            }


        } // For demo control camera with keyboard and mouse



        if (need_render) {
            if (auto_run == 0 && demo_ctrl == 0) { 
                std::cerr << "Rendering frame " << frame_idx
                          << " / " << (poses.size() - 1) << "\n";
            }

            auto t0 = std::chrono::steady_clock::now();
            if (!run_fpga_frame(ctx, out_buf, poses[frame_idx])) {
                std::cerr << "run_fpga_frame failed at frame " << frame_idx << "\n";
                break;
            }
            auto t1 = std::chrono::steady_clock::now();

            if (!update_fpga_image_sdl_tiled_2x2(viewer, out_buf.data(), 300, 170)) {
                std::cerr << "update_fpga_image_sdl_tiled failed at frame " << frame_idx << "\n";
                break;
            }
            auto t2 = std::chrono::steady_clock::now();

            double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::cerr << "FPGA time: " << elapsed_ms << " ms\n";

            elapsed_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            std::cerr << "Image time: " << elapsed_ms << " ms\n\n";

            need_render = false;

            if (auto_run == 1) {
                if (frame_idx + 1 < poses.size()) {
                    frame_idx++;
                    need_render = true;
                } else {
                    std::cerr << "Auto run finished.\n";
                    running = false;
                }
            }
        }
    }

    destroy_fpga_image_sdl_tiled(viewer);
    close_fpga(ctx);

    return 0;
}
