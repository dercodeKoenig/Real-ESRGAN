import argparse
import glob
import os
from PIL import Image

def main(args):
    # For DF2K, we consider the following three scales
    scale_list = [0.75, 0.5, 1 / 3]
    min_size = 512

    path_list = sorted(glob.glob(os.path.join(args.input, '*')))
    for path in path_list:
        print(path)
        try:
            basename = os.path.splitext(os.path.basename(path))[0]
    
            img = Image.open(path)
            width, height = img.size
            for idx, scale in enumerate(scale_list):
                if int(width * scale) < min_size or int(height * scale) < min_size:
                    print("scale", scale, "skipped")
                    continue
                print(f'\t{scale:.2f}')
                rlt = img.resize((int(width * scale), int(height * scale)), resample=Image.LANCZOS)
                rlt.save(os.path.join(args.output, f'{basename}T{idx}.png'))
        except Exception as e:
            print(e)
         

if __name__ == '__main__':
    """Generate multi-scale versions for GT images with LANCZOS resampling.
    It is now used for DF2K dataset (DIV2K + Flickr 2K)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='datasets/DF2K/DF2K_HR', help='Input folder')
    parser.add_argument('--output', type=str, default='datasets/DF2K/DF2K_multiscale', help='Output folder')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    main(args)
