root = "/data/RescueNet"
output_dir = "/working/runs/rescuenet_final_b4_ohem_cosine_V2" # Update name so you know it has cosine!
# TRYING B4 (If it crashes on 'safetensors', we will revert to B2)
model_name = "nvidia/segformer-b4-finetuned-ade-512-512"

epochs = 300
train_split = "train"
val_split = "val"
test_split = "test"

# Safe batch size for B4 + 1024 Res
batch_size = 2 
fp16 = False
ignore_index = 255
num_classes = 11  # Back to 11

lr_scheduler_type = "cosine"
warmup_ratio = 0.1
label_smoothing_factor = 0.1

DATASET_NAME = "rescuenet_segformer"
DATASET_KWARGS = {
    "root": root,
    "image_size": 1024,
    "augment": True,
    "ignore_index": ignore_index,
    "num_classes": num_classes
}