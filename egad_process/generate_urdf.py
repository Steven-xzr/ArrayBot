import os
from tqdm import tqdm


path = 'egad/eval'


def main():
    with open('template.urdf', 'rt') as t:
        template_urdf = t.read()
    root = os.path.join(os.path.expanduser("~"), path)

    for file_name in tqdm(os.listdir(root)):
        if file_name.endswith('_rescaled_4cm.obj'):
            obj_name = file_name.replace('.obj', '')
            with open(os.path.join(root, file_name.replace('.obj', '.urdf')), 'wt') as f:
                urdf = template_urdf.replace('template', obj_name)
                f.write(urdf)


if __name__ == '__main__':
    main()
