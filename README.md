# FreeSplat-3DGS (FPGA-based 3D Gaussian Splatting)

This repository provides a **runnable demonstration** of our FPGA-based 3D Gaussian Splatting system, including:

- Verilator-based simulation (pre-built executable)
- FPGA bitstream for Xilinx Alveo U200
- Host-side control program
- Example dataset and rendering results

> ⚠️ Note: Due to the large size and IP protection considerations,  
> the full RTL source code and Verilator-generated models are not included.

---

# 📌 1. Verilator Simulation

The Verilator simulation environment is located in: verilator/

## 📂 Contents

- `dataset/`  
  Input dataset for rendering

- `main.cpp`  
  Verilator simulation driver (C++)

- `Makefile`  
  Build configuration (reference only)

- `rasterize.sh`  
  Simulation execution script

- `run`  
  Pre-built simulation executable

---

## ⚠️ Important Note

The following components are **not included** in this repository:

- RTL source code
- Verilator-generated models (`obj_dir`)

Reasons:

- Extremely large file size
- Intellectual property protection

Instead, we provide:

- Pre-built executable (`run`)
- Dataset
- Execution script

---

## ▶️ How to Run Simulation

cd verilator

./rasterize.sh

After execution, rendered images will be generated.

<img width="602" height="367" alt="圖片1" src="https://github.com/user-attachments/assets/0188634b-dd80-4bb3-a8a2-0505f0177a46" />



📌 2. FPGA Bitstream (U200)

The FPGA bitstream: rasterizer_2x2tile_60MHz.bit
is generated using Vivado and can be directly programmed onto Xilinx Alveo U200

🧩 Architecture Overview
<img width="4400" height="2365" alt="圖片2" src="https://github.com/user-attachments/assets/18fce136-54bb-4bda-9372-af5ed69f417c" />


▶️ Usage Flow

Program FPGA with the provided bitstream

Run the host application (see next section)



📌 3. Host Program (FPGA Execution)

Located in: Host/

📂 Contents 
- main.cpp: Main control program
- icssl_xdma.h: PCIe XDMA low-level interface
- icssl_GS.h: Rendering and display control
- icssl_demo_cam_ctrl.h: Camera control (keyboard + mouse interaction)

▶️ Execution Scripts
🔹 01_demo_cam_ctrl.sh
  - Interactive mode
  - Control camera using keyboard and mouse.
  - Suitable for real-time exploration

🔹 02_demo_continuous.sh
  - Predefined camera trajectory
  - Automatically renders continuous viewpoints
  - Suitable for demonstration

⚙️ System Requirements
  - Linux (tested on Ubuntu)
  - Xilinx Alveo U200
  - XDMA driver installed
  - Required libraries:
  - SDL2 (for display)
  - Standard C++ runtime
