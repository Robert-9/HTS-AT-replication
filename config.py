# Ke Chen
# knutchen@ucsd.edu
# HTS-AT: A HIERARCHICAL TOKEN-SEMANTIC AUDIO TRANSFORMER FOR SOUND CLASSIFICATION AND DETECTION
# The configuration for training the model
import os
from datetime import datetime

job_id = os.environ.get("SLURM_JOB_ID", datetime.now().strftime('%Y%m%d_%H%M%S'))

exp_name = "htsat_esc_50"  # the saved ckpt prefix name of the model

workspace = "/public/home/hpc234612204/article_codes/htsat"  # "/home/kechen/Research/HTSAT"  # 代码所在文件夹
dataset_path = "/public/home/hpc234612204/DataSets/ESC_50"  # "/home/Research/audioset"  # 数据集路径
# desed_folder = "/home/Research/DESED" # the desed file 音频事件检测  不需要

dataset_type = "esc-50"  # "audioset" "esc-50" "scv2"
index_type = "full_train"  # only works for audioset
balanced_data = True  # only works for audioset

# AudioSet & SCV2: "clip_bce" |  ESC-50: "clip_ce"
loss_type = "clip_ce"

# trained from a checkpoint, or evaluate a single model 
resume_checkpoint = None 
# "/home/Research/model_backup/AudioSet/HTSAT_AudioSet_Saved_1.ckpt"


# just for esc dataset, select the fold you need for evaluation and (+1) validation
esc_fold = 0
# 为True时用于快速验证代码能否顺利运行  为False则用于正式实验
debug = False
#  用于固定随机数种子，保证实验可复现性
random_seed = 970131  # 19970318 970131 12412 127777 1009 34047


# 训练参数
batch_size = 16 * 1  # batch size per GPU x GPU number , default is 32 x 4 = 128
learning_rate = 0.001  # 1e-4 also workable
max_epoch = 35
num_workers = 0

lr_scheduler_epoch = [10,20,30]
lr_rate = [0.02, 0.05, 0.1]  # [0.1, 0.01, 0.001]


# 数据预处理 没带来提升 已弃用
# these data preparation optimizations do not bring many improvements, so deprecated
enable_token_label = False # token label
class_map_path = "class_hier_map.npy"
class_filter = None 
retrieval_index = [15382, 9202, 130, 17618, 17157, 17516, 16356, 6165, 13992, 9238, 5550, 5733, 1914, 1600, 3450, 13735, 11108, 3762, 
    9840, 11318, 8131, 4429, 16748, 4992, 16783, 12691, 4945, 8779, 2805, 9418, 2797, 14357, 5603, 212, 3852, 12666, 1338, 10269, 2388, 8260, 4293, 14454, 7677, 11253, 5060, 14938, 8840, 4542, 2627, 16336, 8992, 15496, 11140, 446, 6126, 10691, 8624, 10127, 9068, 16710, 10155, 14358, 7567, 5695, 2354, 8057, 17635, 133, 16183, 14535, 7248, 4560, 14429, 2463, 10773, 113, 2462, 9223, 4929, 14274, 4716, 17307, 4617, 2132, 11083, 1039, 1403, 9621, 13936, 2229, 2875, 17840, 9359, 13311, 9790, 13288, 4750, 17052, 8260, 14900]
token_label_range = [0.2,0.6]
enable_time_shift = False # shift time
enable_label_enhance = False # enhance hierarchical label
enable_repeat_mode = False # repeat the spectrogram / reshape the spectrogram


# for model's design
enable_tscam = True  # enbale the token-semantic layer 对于音频事件检测（如Audioset）推荐True  若仅分类且不需要时间-频率定位信息，可以考虑False。

# for signal processing
sample_rate = 32000  # 16000 for scv2, 32000 for audioset and esc-50
clip_samples = sample_rate * 10  # audio_set 10-sec clip
# STFT相关参数
window_size = 1024
hop_size = 320  # 160 for scv2, 320 for audioset and esc-50
mel_bins = 64
fmin = 50
fmax = 14000
shift_max = int(clip_samples * 0.5)

# for data collection
classes_num = 50  # esc: 50 | audioset: 527 | scv2: 35
patch_size = (25, 4)  # 已弃用
crop_size = None  # int(clip_samples * 0.5) 已弃用

# HTS-AT模型结构超参数
# for htsat hyperparamater
htsat_window_size = 8
htsat_spec_size = 256
htsat_patch_size = 4 
htsat_stride = [4, 4]  # (4, 4)
htsat_num_head = [4,8,16,32]
htsat_dim = 96 
htsat_depth = [2,2,6,2]

swin_pretrain_path = "/public/home/hpc234612204/pretrained_models/swin_small_patch4_window7_224.pth"  # None
# "/home/Research/model_backup/pretrain/swin_tiny_c24_patch4_window8_256.pth"

# 一些废弃优化选项（实验性）
# Some Deprecated Optimization in the model design, check the model code for details
htsat_attn_heatmap = False
htsat_hier_output = False 
htsat_use_max = False


# 模型集成与权重平均设置（测试阶段用）
# for ensemble test
ensemble_checkpoints = []  # 用于指定多个已经训练完成的模型checkpoint路径。进行集成测试时，会同时加载多个模型进行预测，并综合多个模型的预测结果。当此列表为空时，表示不进行集成测试。
ensemble_strides = []  # 表示模型推理阶段时对音频数据进行滑动窗口预测时的步长  若为空，则使用默认stride；若非空，则表示每个模型分别使用不同stride

# 权值平均文件夹
# weight average folder
wa_folder = ""  # os.path.join(workspace, "version_0/checkpoints/")  # "/home/version_0/checkpoints/"
# weight average output filename
wa_model_path = ""  # os.path.join(workspace, "HTSAT_AudioSet_Saved_x.ckpt")

# 多模型(ESM)测试设置
esm_model_pathes = [
    # "/public/home/hpc234612204/article_codes/htsat/model_backup/AudioSet/HTSAT_AudioSet_Saved_1.ckpt",
    # "/public/home/hpc234612204/article_codes/htsat/model_backup/AudioSet/HTSAT_AudioSet_Saved_2.ckpt",
    # "/public/home/hpc234612204/article_codes/htsat/model_backup/AudioSet/HTSAT_AudioSet_Saved_3.ckpt",
    # "/public/home/hpc234612204/article_codes/htsat/model_backup/AudioSet/HTSAT_AudioSet_Saved_4.ckpt",
    # "/public/home/hpc234612204/article_codes/htsat/model_backup/AudioSet/HTSAT_AudioSet_Saved_5.ckpt",
    # "/public/home/hpc234612204/article_codes/htsat/model_backup/AudioSet/HTSAT_AudioSet_Saved_6.ckpt"
    # "/home/Research/model_backup/AudioSet/HTSAT_AudioSet_Saved_1.ckpt",
    # "/home/Research/model_backup/AudioSet/HTSAT_AudioSet_Saved_2.ckpt",
    # "/home/Research/model_backup/AudioSet/HTSAT_AudioSet_Saved_3.ckpt",
    # "/home/Research/model_backup/AudioSet/HTSAT_AudioSet_Saved_4.ckpt",
    # "/home/Research/model_backup/AudioSet/HTSAT_AudioSet_Saved_5.ckpt",
    # "/home/Research/model_backup/AudioSet/HTSAT_AudioSet_Saved_6.ckpt"
]

# 帧级别定位（音频事件检测）配置（可选高级用法）
# for framewise localization
heatmap_dir = os.path.join(workspace, "heatmap_output")
test_file = "htsat-test-ensemble"
fl_local = False  # indicate if we need to use this dataset for the framewise detection
fl_dataset = os.path.join(workspace, "desed/desed_eval.npy")
fl_class_num = [
    # "Speech", "Frying", "Dishes", "Running_water",
    # "Blender", "Electric_shaver_toothbrush", "Alarm_bell_ringing",
    # "Cat", "Dog", "Vacuum_cleaner"
]

# 将527类映射到10类
# map 527 classes into 10 classes
fl_audioset_mapping = [
    [0,1,2,3,4,5,6,7],
    [366, 367, 368],
    [364],
    [288, 289, 290, 291, 292, 293, 294, 295, 296, 297],
    [369],
    [382],
    [310, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402],
    [81, 82, 83, 84, 85],
    [74, 75, 76, 77, 78, 79],
    [377]
]