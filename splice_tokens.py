import os
from re import sub

from train import train_model

DATAROOT = "/home/labs/testing/class48/githubs/dl4cv_proj_r/datasets/curr_pair"
STYLE_LIST = ['Cubism', 'Impressionism', 'Pop Art', 'Rococo', 'Ukiyo-e']


def snake_case(s):
    return '_'.join(
        sub('([A-Z][a-z]+)', r' \1',
            sub('([A-Z]+)', r' \1',
                s.replace('-', ' '))).split()).lower()


def main():
    style = os.getenv('STYLE')
    for style_name in STYLE_LIST:
        if style_name.lower().replace(' ', '').replace('-', '').startswith(
                style.lower().replace(' ', '').replace('-', '')):
            style = style_name
            break
    else:
        raise Exception(f'Could not file style. Please specify a style from the list: {", ".join(STYLE_LIST)}')

    train_model(DATAROOT, style, "./datasets/tokens_path", "./datasets/train_img_path", output_file_prefix=f'mona_token_{snake_case(style_name)}')


if __name__ == "__main__":
    main()
