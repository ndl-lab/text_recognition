# Copyright (c) 2022, National Diet Library, Japan
#
# This software is released under the CC BY 4.0.
# https://creativecommons.org/licenses/by/4.0/


import pathlib
import json
import lmdb

from xmlparser import XMLRawDataset, ListRawDataset

import argparse


class Env:
    def __init__(self, output_path, interval_writeCache=1000):
        self.output_path = output_path
        self.env = lmdb.open(str(output_path), map_size=1099511627776)
        self.cache = dict()
        self.n = 0
        self.interval = interval_writeCache

    def finish_line(self):
        self.n += 1
        if self.n % 1000 == 0:
            self.writeCache()

    def writeCache(self):
        with self.env.begin(write=True) as txn:
            for k, v in self.cache.items():
                txn.put(k, v)
        self.cache = {}
        print(f'Written {self.n} lines @ {self.output_path}')


def createDataset(input_path, output_path, db_type='xml', dry_run=False):
    p = pathlib.Path(output_path)
    p.mkdir(parents=True, exist_ok=True)

    if db_type == 'xml':
        generator = XMLRawDataset.from_list(input_path, image_type=XMLRawDataset.IMAGE_TYPE_ENCODED)
    elif db_type == 'list':
        generator = ListRawDataset(input_path, image_type=XMLRawDataset.IMAGE_TYPE_ENCODED)
    if dry_run:
        return

    # generate database
    env = Env(output_path[0])
    env.cache['dbtype'.encode()] = 'xml'.encode()

    for il, (g, line) in enumerate(generator):
        env.cache[f'{env.n:09d}-direction'.encode()] = line.get('direction').encode()
        env.cache[f'{env.n:09d}-label'.encode()] = line.get('label').encode()
        env.cache[f'{env.n:09d}-cattrs'.encode()] = json.dumps(line.get('cattrs')).encode()
        env.cache[f'{env.n:09d}-image'.encode()] = g
        env.finish_line()

    env.cache['n_line'.encode()] = str(env.n).encode()
    env.writeCache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', nargs='+', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--db_type', default='xml', choices=['xml', 'list'])
    parser.add_argument('--dry-run', action='store_true')
    opt = parser.parse_args()
    createDataset(opt.input_path, opt.output_path, opt.db_type, dry_run=opt.dry_run)
