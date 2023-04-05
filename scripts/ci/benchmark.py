import os
import argparse
import datetime
import subprocess
import time
import pytz
import hidet

hidet.option.cache_dir(os.path.join(hidet.option.get_cache_dir(), 'benchmark'))
hidet.utils.hidet_clear_op_cache()


parser = argparse.ArgumentParser('Benchmark hidet performance.')
parser.add_argument('--git-commit', default=None, type=str, help='Git commit hash.')
parser.add_argument('--space', default=0, type=int, help='Search space of hidet.')
parser.add_argument('--report', default='./report.txt', type=str, help='Report file path.')


def info(args) -> str:
    envs = [
        '# {}'.format(datetime.datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')),
        '- Hidet version: {}'.format(hidet.__version__),
        '- Git commit: {}'.format(args.git_commit),
        '',
    ]
    return '\n'.join(envs)


def main():
    args = parser.parse_args()
    commands = [
        # f'hidet bench --space {args.space} --dtype float32 --report resnet50_f32.txt --tensor-core resnet --models resnet50',
        # f'hidet bench --space {args.space} --dtype float16 --report resnet50_f16.txt --tensor-core resnet --models resnet50',
        f'hidet bench --space {args.space} --dtype float32 --report bert-seq128-f32.txt --tensor-core nlp --seq-length 128 --models bert-base-uncased',
        # f'hidet bench --space {args.space} --dtype float16 --report bert-seq128-f16.txt --tensor-core nlp --seq-length 128 --models bert-base-uncased',
    ]
    with open(args.report, 'w') as f:
        t1 = time.time()
        f.write(info(args) + '\n')
        for idx, command in enumerate(commands):
            output_file = command.split('--report ')[1].split(' ')[0]
            subprocess.run(command.split(), check=True)
            with open(output_file, 'r') as g:
                if idx == 0:
                    f.write(g.read())
                else:
                    f.write(g.readlines()[-1])
        t2 = time.time()
        f.write('\n')
        f.write('Time: {:.2f} hours\n'.format((t2 - t1) / 60 / 60))


if __name__ == '__main__':
    main()