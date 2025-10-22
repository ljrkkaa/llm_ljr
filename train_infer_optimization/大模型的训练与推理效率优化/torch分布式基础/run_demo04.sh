torchrun --nproc_per_node=2 \
        --nnodes=1 \
        --master_addr="127.0.0.1"\
        --master_port=1234 \
        multiprocessing_demo4.py
