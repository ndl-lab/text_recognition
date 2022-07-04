# Copyright (c) 2022, National Diet Library, Japan
#
# This software is released under the CC BY 4.0.
# https://creativecommons.org/licenses/by/4.0/


import sys
import pathlib
import itertools
import re

from joblib import Parallel, delayed

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import IterableDataset, ChainDataset

import xml.etree.ElementTree as ET

pt = re.compile(r'(\{(.*)\})?(.*)')


def find_image_root(image_name, leaf_dir, root_dir):
    image_root = leaf_dir.parent
    while image_root >= root_dir:
        image_paths = list(image_root.rglob(image_name))
        if len(image_paths) > 1:
            print(ValueError(f'{image_name} is ambiguous\n{image_paths}'), sys.stderr)
            break
        elif len(image_paths) == 1:
            return pathlib.Path(str(image_paths[0])[:-len(image_name)])
        image_root = image_root.parent
    return None


class XMLRawDataset(IterableDataset):
    IMAGE_TYPE_NONE = 0
    IMAGE_TYPE_BYTE = 1
    IMAGE_TYPE_ENCODED = 2
    IMAGE_TYPE_IMAGE = 3
    IMAGE_TYPE_GRAY_IMAGE = 4

    @staticmethod
    def from_list(input_paths,
                  image_type=IMAGE_TYPE_IMAGE,
                  accept_empty=False, accept_only_en=False,
                  keep_remain=False,
                  n_jobs=4):

        if type(input_paths) == str:
            input_paths = [input_paths]
        else:
            assert type(input_paths) == list
        xml_files = []
        for p in input_paths:
            print(p)
            p = pathlib.Path(p)
            assert p.exists()
            xmls = p.rglob('*.xml')
            xml_files.extend([(f, p) for f in xmls])
        xml_files = sorted(xml_files)
        return ChainDataset(Parallel(n_jobs)([
            delayed(XMLRawDataset)(xml_file, input_path, image_type, accept_empty, accept_only_en, keep_remain)
            for xml_file, input_path in xml_files]))

    def __init__(self, xml_file, input_path,
                 image_type=IMAGE_TYPE_IMAGE,
                 accept_empty=False, accept_only_en=False,
                 keep_remain=False,
                 sensitive=False):
        self.xml_file = xml_file
        self.input_path = input_path
        self.image_type = image_type
        self.accept_empty = accept_empty
        self.accept_only_en = accept_only_en
        self.keep_remain = keep_remain
        self.sensitive = sensitive
        with open(xml_file, 'r') as f:
            try:
                tree = ET.parse(f)
                self.root = tree.getroot()
            except ValueError:
                with open(xml_file, encoding="utf-8") as file:
                    self.root = ET.fromstring(file.read())
            except ET.ParseError as e:
                print(f)
                raise e
        groups = pt.match(self.root.tag).groups()
        if groups[1]:
            self.namespace = f"{{{groups[1]}}}"
        else:
            self.namespace = ''
        if groups[2] != 'OCRDATASET':
            return []
        self.remains = None

    def __iter__(self):
        if self.remains is not None:
            for v in self.remains:
                yield v
            self.remains = None
            return
        if self.keep_remain:
            self.remains = []
        image_root = None

        for xpage in self.root:
            page = xpage.attrib

            image_name = page['IMAGENAME']
            if image_root is None:
                image_root = find_image_root(image_name, self.xml_file, self.input_path)
                if image_root is None:
                    print(FileNotFoundError(f'{image_name} is not Found'), file=sys.stderr)
                    continue
            image_path = image_root / image_name
            if not image_path.exists():
                print(FileNotFoundError(f'{image_name} is not Found'), file=sys.stderr)
                continue
            ext = image_path.suffix

            if self.image_type is not XMLRawDataset.IMAGE_TYPE_NONE:
                try:
                    with open(image_path, 'rb') as f:
                        imageBin = f.read()
                    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
                    img = cv2.imdecode(imageBuf, cv2.IMREAD_ANYCOLOR)
                    if img is None:
                        raise Exception('image cant not decode')
                except Exception as e:
                    print(f'error occured at {image_path}\n{e}')
                    continue

            if self.accept_only_en:
                ilines = itertools.chain.from_iterable([
                    # xpage.iterfind(f'.//{self.namespace}BLOCK[@TYPE="欧文"]'),
                    xpage.iterfind(f'.//{self.namespace}INLINE[@TYPE="欧文"]'),
                    xpage.iterfind(f'.//{self.namespace}INLINE[@TYPE="回転欧文"]'),
                ])
            else:
                data_list = [
                    xpage.iterfind(f'.//{self.namespace}LINE'),
                ]
                if self.accept_empty:
                    data_list.extend([
                        xpage.iterfind(f'.//{self.namespace}BLOCK[@TYPE="回転欧文"]'),
                        xpage.iterfind(f'.//{self.namespace}BLOCK[@TYPE="欧文"]'),
                        xpage.iterfind(f'.//{self.namespace}BLOCK[@TYPE="ノンブル"]'),
                        xpage.iterfind(f'.//{self.namespace}BLOCK[@TYPE="柱"]'),
                    ])
                ilines = itertools.chain.from_iterable(data_list)
            for xline in ilines:
                line = xline.attrib
                line['done'] = True

                if 'ERROR' in line:
                    continue

                line['X'] = int(line['X'])
                line['Y'] = int(line['Y'])
                line['WIDTH'] = int(line['WIDTH'])
                line['HEIGHT'] = int(line['HEIGHT'])
                line['path'] = image_path
                line['tag'] = xline.tag

                if self.accept_empty:
                    line['STRING'] = line.get('STRING', None)
                    if 'DIRECTION' not in line:
                        if line['WIDTH'] > line['HEIGHT']:
                            line['DIRECTION'] = '横'
                        else:
                            line['DIRECTION'] = '縦'
                elif len(line.get('STRING', '')) == 0:
                    continue
                line['label'] = line['STRING']
                line['direction'] = line['DIRECTION']

                chars = list()
                line['cattrs'] = chars

                for char in xline:
                    char = char.attrib
                    char['X'] = int(char['X']) - line['X']
                    char['Y'] = int(char['Y']) - line['Y']
                    char['WIDTH'] = int(char['WIDTH'])
                    char['HEIGHT'] = int(char['HEIGHT'])
                    chars.append(char)

                if self.image_type is not self.IMAGE_TYPE_NONE:
                    y1, y2 = line['Y'], line['Y'] + line['HEIGHT']
                    x1, x2 = line['X'], line['X'] + line['WIDTH']
                    g = img[y1:y2, x1:x2]
                    if self.image_type == self.IMAGE_TYPE_ENCODED:
                        retval, g = cv2.imencode(ext, g)
                        assert retval
                    elif self.image_type == self.IMAGE_TYPE_IMAGE:
                        g = Image.fromarray(g)
                    elif self.image_type == self.IMAGE_TYPE_GRAY_IMAGE:
                        g = Image.fromarray(g).convert('L')
                else:
                    g = None

                if not self.sensitive and line['label']:
                    line['label'] = line['label'].lower()

                yield g, line
            if self.keep_remain:
                for xline in xpage.iterfind("./*"):
                    line = xline.attrib
                    if 'done' in line:
                        continue
                    line['path'] = image_path
                    line['tag'] = xline.tag
                    self.remains.append(line)


class SyntheticDataset(IterableDataset):
    def __init__(self, chars, fontpath):
        from PIL import ImageFont, ImageDraw
        import functools
        self.chars = chars
        dtmp = ImageDraw.Draw(Image.new('L', (400, 200)))
        self._font = ImageFont.truetype(fontpath, 32)
        self._textsize = functools.partial(dtmp.multiline_textsize, font=self._font)

    def __iter__(self):
        import random
        from more_itertools import chunked
        from PIL import ImageDraw
        chars = []
        chars.extend(self.chars)
        random.shuffle(chars)

        for s in chunked(chars, 40):
            s = ''.join(s)
            w, h = self._textsize(s)
            g = Image.new('RGB', (w, h), (255, 255, 255))
            d = ImageDraw.Draw(g)
            d.text((0, 0), s, font=self._font, fill=(0, 0, 0))
            line = {
                'path': None,
                'direction': '横',
                'label': s,
                'WIDTH': w,
                'HEIGHT': h,
                'STRING': s,
                'cattrs': None,
            }
            yield g.convert('L'), line


class ListRawDataset(IterableDataset):
    IMAGE_TYPE_BYTE = 0
    IMAGE_TYPE_ENCODED = 1
    IMAGE_TYPE_IMAGE = 2
    IMAGE_TYPE_GRAY_IMAGE = 3

    def __init__(self, input_paths, read_image=True, image_type=IMAGE_TYPE_IMAGE, n_jobs=4):
        self.read_image = read_image
        self.image_type = image_type

        if type(input_paths) == str:
            input_paths = [input_paths]
        else:
            assert type(input_paths) == list

        self.images = list()
        self.labels = list()

        for input_path in input_paths:
            p = pathlib.Path(input_path)
            assert p.exists()
            with open(p, 'r') as f:
                lines = f.readlines()

            for line in lines:
                image_path, label = line.strip('\n').split()
                image_path = p.parent / image_path
                self.images.append(image_path)
                self.labels.append(label)

    def __iter__(self):
        for image_path, label in zip(self.images, self.labels):
            ext = image_path.suffix

            try:
                with open(image_path, 'rb') as f:
                    imageBin = f.read()
                imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
                img = cv2.imdecode(imageBuf, cv2.IMREAD_ANYCOLOR)
                h, w, c = img.shape
                if img is None:
                    raise Exception('image cant not decode')
            except Exception as e:
                print(f'error occured at {image_path}\n{e}')
                continue

            if self.read_image:
                g = img
                if self.image_type == self.IMAGE_TYPE_ENCODED:
                    retval, g = cv2.imencode(ext, g)
                    assert retval
                elif self.image_type == self.IMAGE_TYPE_IMAGE:
                    g = Image.fromarray(g)
                elif self.image_type == self.IMAGE_TYPE_GRAY_IMAGE:
                    g = Image.fromarray(g).convert('L')
            else:
                g = None
            line = {
                'path': image_path,
                'direction': '横',
                'label': label,
                'WIDTH': w,
                'HEIGHT': h,
                'STRING': label,
            }
            yield g, line


class XMLRawDatasetWithCli(IterableDataset):
    def __init__(self, img_data, xml_data,
                 accept_empty=False, accept_only_en=False,
                 sensitive=False, yield_block_pillar=True, yield_block_page_num=True,yield_block_rubi=True):

        self.xml_data = xml_data
        self.orig_img = img_data
        self.accept_empty = accept_empty
        self.accept_only_en = accept_only_en
        self.sensitive = sensitive
        self.root = xml_data.getroot()
        self.remains = None
        self.yield_block_pillar = yield_block_pillar
        self.yield_block_page_num = yield_block_page_num
        self.yield_block_rubi=yield_block_rubi
        groups = pt.match(self.root.tag).groups()
        if groups[1]:
            self.namespace = f"{{{groups[1]}}}"
        else:
            self.namespace = ''
        if groups[2] != 'OCRDATASET':
            return []

    def __iter__(self):
        from PIL import Image
        root = self.xml_data.getroot()
        for xml_line in root.find('PAGE'):
            is_block_page_num = (xml_line.tag == 'BLOCK') and (xml_line.attrib['TYPE'] == 'ノンブル')
            is_block_pillar = (xml_line.tag == 'BLOCK') and (xml_line.attrib['TYPE'] == '柱')
            is_block_rubi = (xml_line.tag == 'BLOCK') and (xml_line.attrib['TYPE'] == 'ルビ')
            do_yield = (xml_line.tag == 'LINE') or (is_block_page_num and self.yield_block_page_num) or (is_block_pillar and self.yield_block_pillar) or (is_block_rubi and self.yield_block_rubi)
            if not do_yield:
                print("This {0} elemetn will be skipped.".format(xml_line.tag))
                print(xml_line.attrib)
                continue
            xml_attrib = xml_line.attrib

            w = xml_attrib['WIDTH']
            h = xml_attrib['HEIGHT']
            y1, y2 = int(xml_attrib['Y']), int(xml_attrib['Y']) + int(xml_attrib['HEIGHT'])
            x1, x2 = int(xml_attrib['X']), int(xml_attrib['X']) + int(xml_attrib['WIDTH'])
            g = Image.fromarray(self.orig_img[y1:y2, x1:x2]).convert('L')

            direction = '横'
            if 'DIRECTION' not in xml_attrib:
                if int(w) > int(h):
                    direction = '横'
                else:
                    direction = '縦'
            else:
                direction = xml_attrib['DIRECTION']

            line = {
                'path': None,
                'direction': direction,
                'label': None,
                'WIDTH': w,
                'HEIGHT': h,
                'STRING': None,
                'cattrs': None,
                'X' : xml_attrib['X'],
                'Y' : xml_attrib['Y'],
                'TYPE' :xml_attrib['TYPE'],
                'tag' : xml_line.tag,
                'STRING': None,
                'done': True
            }

            yield g, line
