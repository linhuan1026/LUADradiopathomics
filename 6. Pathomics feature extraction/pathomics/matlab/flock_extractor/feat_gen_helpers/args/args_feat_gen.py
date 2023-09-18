import argparse


parser = argparse.ArgumentParser()
# data
parser.add_argument('--root', type=str, default='/gstore/scratch/goya/tiles/4828/Cellular_Features_4828_m_'
                                                '40_0_tile_256_step_256_mask_filter_based_on_custom_tumor_'
                                                'image_format_png',)
parser.add_argument('--header', type=str, default='goya_1',)
parser.add_argument('--sheet_ext', type=str, default='.csv',)
parser.add_argument('--preload_filelist_dir', type=str, default='./preloaded/')
parser.add_argument('--export_dir', type=str, default='./outputs')
parser.add_argument('--preprocess_n_workers', type=int, default=16)
parser.add_argument('--step_size', type=int, default=256)
parser.add_argument('--tile_size', type=int, default=2048)
parser.add_argument('--num_to_sample', type=int, default=20)
parser.add_argument('--min_patch_num', type=int, default=10)

parser.add_argument('--save_json_debug', type=bool, default=True)
parser.add_argument('--spatial_keys', type=str, nargs='+', default=['x', 'y'])
parser.add_argument('--attr_names', type=str, nargs='+', default=['Area', 'original_firstorder_Mean'])
# parser.add_argument('--valid_ext', type=str, nargs='+', default=['.csv'])
parser.add_argument('--min_cell_num_inclusive', type=int, default=5)

parser.add_argument('--flock_bandwidth', type=float, default=180)
parser.add_argument('--flock_pheno_num', type=int, default=2)
# compatibility of jupyter notebooks
opt, remaining = parser.parse_known_args()

CONST_JSON_KEY = 'result'
CONST_FILELIST_NAME = 'file_lists'
CONST_AGGR_SHEET_NAME = 'aggregated_sheets'


def filelist_name(header: str):
    return f"{CONST_FILELIST_NAME}_{header}.json"


def aggr_sheet_name(header: str):
    return f"{CONST_AGGR_SHEET_NAME}_{header}.json"

