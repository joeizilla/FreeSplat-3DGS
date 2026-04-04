time ./run \
	-in dataset/3dgs_input.txt \
	-in_pose dataset/campose_input.txt \
	-in_count dataset/3dgs_count.txt \
	-mode 1 \
	-dump_lv 99 \
	-dump_cycle 500000 \
	-max_cycle 500000 \
	-3DGS_count 128570 \
	> exe_hw.log
