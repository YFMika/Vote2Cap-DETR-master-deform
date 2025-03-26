python demo.py \
    --use_color \
    --use_normal \
    --dataset scene_scanrefer \
    --vocabulary scanrefer \
    --use_beam_search \
    --detector detector_Vote2Cap_DETRv2 \
    --captioner captioner_dccv2 \
    --checkpoint_dir outputs/vote2cap_detrv2 \
    --batchsize_per_gpu 8 \
    --test_ckpt ./weights/vote2cap-detr++/scanrefer_vote2cap_detrv2_XYZ_RGB_NORMAL.pth \
    --gpu 3