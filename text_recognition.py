# Copyright (c) 2022, National Diet Library, Japan
#
# This software is released under the CC BY 4.0.
# https://creativecommons.org/licenses/by/4.0/


import argparse
import functools
import difflib
import collections
import pathlib
from tqdm import tqdm

import xml.etree.ElementTree as ET
from xml.dom import minidom

from PIL import Image, ImageDraw, ImageFont
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Subset
from nltk.metrics import edit_distance

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import XMLLmdbDataset, AlignCollate, tensor2im
from model import Model
from xmlparser import XMLRawDataset, SyntheticDataset, XMLRawDatasetWithCli
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def gen_dataset(db_type, db_path, opt, line_index=None, accept_empty=True, keep_remain=False):
    if db_type == 'xmllmdb':
        ds = ConcatDataset([XMLLmdbDataset(root=p, opt=opt) for p in db_path])
        if line_index is not None:
            ds = Subset(ds, opt.line_index)
    elif db_type == 'xmlraw':
        ds = XMLRawDataset.from_list(input_paths=db_path,
                                     image_type=XMLRawDataset.IMAGE_TYPE_GRAY_IMAGE,
                                     accept_empty=accept_empty, keep_remain=keep_remain)
        opt.workers = 0
    elif db_type == 'synth':
        ds = SyntheticDataset(opt.character, db_path)
    return ds


def _debug_char_prob(preds_prob, character):
    preds_v, preds_i = torch.topk(preds_prob, 3)
    for b in zip(preds_v.tolist(), preds_i.tolist()):
        for p3, i3 in zip(*b):
            if i3[0] == 0:
                continue
            for p, i in zip(p3, i3):
                if p > 0.01:
                    print(f'{p:.2f}', character[i-1], end=' ')
            print()
        print('--------')


class Inferencer:
    @staticmethod
    def get_argparser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
        parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
        parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
        """ Data processing """
        parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
        parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
        parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
        parser.add_argument('--rgb', action='store_true', help='use rgb input')
        parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
        parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
        parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
        parser.add_argument('--remove_char', default=None, help='remove the specified index class. ex. 〓')
        """ Model Architecture """
        parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
        parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
        parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
        parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
        parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
        parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
        parser.add_argument('--output_channel', type=int, default=512,
                            help='the number of output channel of Feature extractor')
        parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
        return parser

    def __init__(self, opt):
        """
        Args:
            opt
                上記get_parserによってparseされたargument
        """
        # model config
        if 'CTC' in opt.Prediction:
            converter = CTCLabelConverter(opt.character)
        else:
            converter = AttnLabelConverter(opt.character)
        opt.num_class = len(converter.character)
        if opt.remove_char is not None:
            opt.remove_char = opt.character.index(opt.remove_char) + 1

        if opt.rgb:
            opt.input_channel = 3
        model = Model(opt)
        print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
              opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
              opt.SequenceModeling, opt.Prediction)
        model = torch.nn.DataParallel(model).to(device)

        # load model
        print('loading pretrained model from %s' % opt.saved_model)
        model.load_state_dict(torch.load(opt.saved_model, map_location=device))

        self.model = model
        self.converter = converter
        self.aligncollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        self.opt = opt

    def gen(self, dataset, keep_remain=False, with_tqdm=False):
        """
        Args:
            dataset
                以下を生成するtorch.utils.data.Dataset
                PIL.Image(mode="L"), {'WIDTH': int, 'HEIGHT': int, 'STRING': string}
            keep_remain
                これが有効のとき、xmlraw dbは偶数週目に
                推論しない要素を吐くようになる
            with_tqdm
                これが有効のとき、進捗表示をする
        Yields:
            image
            groundtruth label
            prediction label
            confidence score
            appendix information
        """
        converter = self.converter

        demo_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=int(self.opt.workers),
            collate_fn=self.aligncollate, pin_memory=True)
        if with_tqdm:
            demo_loader = tqdm(demo_loader, ncols=80)

        # predict
        self.model.eval()
        with torch.no_grad():
            for image_tensors, labels, data in demo_loader:
                batch_size = image_tensors.size(0)
                image = image_tensors.to(device)
                # For max length prediction
                length_for_pred = torch.IntTensor([self.opt.batch_max_length] * batch_size).to(device)
                text_for_pred = torch.LongTensor(batch_size, self.opt.batch_max_length + 1).fill_(0).to(device)

                if 'CTC' in self.opt.Prediction:
                    preds = self.model(image, text_for_pred)
                    if self.opt.remove_char is not None:
                        preds[:, :, self.opt.remove_char] = -1e5

                    # Select max probabilty (greedy decoding) then decode index to character
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    _, preds_index = preds.max(2)
                    # preds_index = preds_index.view(-1)
                    preds_str = converter.decode(preds_index, preds_size)
                else:
                    preds = self.model(image, text_for_pred, is_train=False)

                    # select max probabilty (greedy decoding) then decode index to character
                    _, preds_index = preds.max(2)
                    preds_str = converter.decode(preds_index, length_for_pred)

                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)

                if 0:
                    _debug_char_prob(preds_prob, self.opt.character)

                for image, gt, pred, pred_max_prob, datum in zip(image, labels, preds_str, preds_max_prob, data):
                    if 'Attn' in self.opt.Prediction:
                        pred_EOS = pred.find('[s]')
                        pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                        pred_max_prob = pred_max_prob[:pred_EOS]

                    # calculate confidence score (= multiply of pred_max_prob)
                    try:
                        confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                    except Exception:
                        confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])

                    yield image, gt, pred, confidence_score, datum

            if keep_remain:
                for datum in dataset:
                    yield None, None, None, None, datum


class TR_WORKER:
    CHAR_DIFF_NONE = 0
    CHAR_DIFF_WRONG = 1
    CHAR_DIFF_ALL = 2

    def __init__(self,
                 accuracy=False,
                 levenshtein_distance=False,
                 char_diff=CHAR_DIFF_NONE,
                 render=False,
                 xml=None,
                 outimage_dir=None, font_path=None,
                 stat=False):
        self._task = []
        self._accuracy = accuracy
        self._char_diff = char_diff
        self._levenshtein_distance = levenshtein_distance
        self._xml = xml
        self._stat = stat

        self.nline = 0
        if accuracy:
            self.accuracy = 0
            self.ncorrect = 0
            self._task.append(self._facc)

        if levenshtein_distance:
            self.sum_dist = 0
            self.normalized_edit_distance = 0
            self._task.append(self._fld)

        if char_diff != self.CHAR_DIFF_NONE:
            self.counters = {
                'misstake': collections.Counter(),
                'tp': collections.Counter(),
                'fn': collections.Counter(),
                'fp': collections.Counter(),
            }
            self._task.append(self._fchar_diff)

            self.outimage_dir = outimage_dir
            if outimage_dir is None:
                self.outimage_dir = None
            else:
                assert font_path is not None
                self.outimage_dir = pathlib.Path(outimage_dir)
                self.outimage_dir.mkdir(exist_ok=True)
                dtmp = ImageDraw.Draw(Image.new('L', (400, 200)))
                self._font = ImageFont.truetype(font_path, 32)
                self._textsize = functools.partial(dtmp.multiline_textsize, font=self._font)

        if render:
            self._task.append(self._frender)
            assert font_path is not None
            assert outimage_dir is not None
            self.outimage_dir = pathlib.Path(outimage_dir)
            self.outimage_dir.mkdir(exist_ok=True)
            dtmp = ImageDraw.Draw(Image.new('L', (400, 200)))
            self._font = ImageFont.truetype(font_path, 32)
            self._textsize = functools.partial(dtmp.multiline_textsize, font=self._font)

        if xml:
            self.outxml_dir = pathlib.Path(xml)
            self.outxml_dir.mkdir(exist_ok=True)
            self._xml_data = {}
            self._task.append(self._fxml)

    def finalize(self):
        if self._accuracy:
            self.accuracy = self.ncorrect / self.nline
        if self._levenshtein_distance:
            self.normalized_edit_distance = self.sum_dist / self.nline
        if self._xml:
            self._fgenerate_xml()
        if self._stat:
            print('===== f measure =====')
            for c in self.counters['tp'].keys() | self.counters['fp'].keys() | self.counters['fn'].keys():
                tp = self.counters['tp'][c]
                precision = tp / (tp + self.counters['fp'][c] + 1e-9)
                recall = tp / (tp + self.counters['fn'][c] + 1e-9)
                print(c, f"{2 * precision * recall / (precision + recall + 1e-9):.3f}")
            print('===== misstake stat =====')
            for p, n in self.counters['misstake'].most_common():
                if p[1] == '-':
                    print(p, n, f"U+{ord(p[0]):X} U+{ord(p[2]):X}")
        return self

    def _facc(self, correct, *args):
        if correct:
            self.ncorrect += 1

    def _fld(self, correct, image, gt, pred, *args):
        d = edit_distance(gt, pred)
        if len(gt) == 0 and len(pred) == 0:
            self.sum_dist += 0
        elif len(gt) > len(pred):
            self.sum_dist += 1 - d / len(gt)
        else:
            self.sum_dist += 1 - d / len(pred)

    def _frender(self, correct, image, sa1, sb1, *args):
        image_pil = Image.fromarray(tensor2im(image))
        w, h = self._textsize(f'{sb1}')
        g = Image.new(image_pil.mode, (w, h), (255, 255, 255))
        d = ImageDraw.Draw(g)
        p = [0, 0]
        draw_escape_colored_text(sb1, d, p=p, font=self._font)
        if h * image_pil.width > image_pil.height * 2 * w:
            w = w * image_pil.height * 2 // h
            h = image_pil.height * 2
        else:
            h = h * image_pil.width // w
            w = image_pil.width
        g = g.resize((w, h))
        canvas = Image.new(image_pil.mode, (image_pil.width, image_pil.height + h), (255, 255, 255))
        canvas.paste(image_pil)
        canvas.paste(g, (0, image_pil.height))
        canvas.save(self.outimage_dir / f'{self.nline:09d}-{sb1.replace("/", "")}.png')

    def _fchar_diff(self, correct, image, sa1, sb1, *args):
        if correct and self._char_diff != self.CHAR_DIFF_ALL:
            if self._char_diff == self.CHAR_DIFF_ALL:
                print('------------------')
                print(sa1)
            return
        if sa1 is None:
            sa1 = ''
        sm = difflib.SequenceMatcher(None, sa1, sb1)
        sa2 = str()
        sb2 = str()
        reason = ''
        for tag, ia1, ia2, ib1, ib2 in sm.get_opcodes():
            if tag == 'equal':
                sa2 += "\033[0m"
                sb2 += "\033[0m"
                self.counters['tp'].update(list(sa1[ia1:ia2]))
            elif tag == 'replace':
                sa2 += "\033[31m"
                sb2 += "\033[31m"
                self.counters['fn'].update(list(sa1[ia1:ia2]))
                self.counters['fp'].update(list(sb1[ia1:ib2]))
                for ia, ib in zip(range(ia1, ia2), range(ib1, ib2)):
                    self.counters['misstake'].update([f'{sa1[ia]}-{sb1[ib]}'])
                    reason += f'{sa1[ia]}-{sb1[ib]},'
            elif tag == 'insert':
                sb2 += "\033[33m"
                self.counters['fp'].update(list(sb1[ia1:ib2]))
                for ia in range(ia1, ia2):
                    self.counters['misstake'].update([f'{sa1[ia]}>　'])
                    reason += f'{sa1[ia]}>　,'
            elif tag == 'delete':
                sa2 += "\033[33m"
                self.counters['fn'].update(list(sa1[ia1:ia2]))
                for ib in range(ib1, ib2):
                    self.counters['misstake'].update([f'　<{sb1[ib]}'])
                    reason += f'　<{sb1[ib]},'
            sa2 += sa1[ia1:ia2]
            sb2 += sb1[ib1:ib2]
        sa2 += '\033[0m'
        sb2 += '\033[0m'

        if self._char_diff != self.CHAR_DIFF_NONE:
            print(f'-{self.nline:09d}-----------------')
            print(sa2)
            print(sb2)

        if self.outimage_dir is not None:
            image_pil = Image.fromarray(tensor2im(image))
            w, h = self._textsize(f'{sa2}\n{sb2}')
            g = Image.new(image_pil.mode, (w, h), (255, 255, 255))
            d = ImageDraw.Draw(g)
            p = [0, 0]
            draw_escape_colored_text(sa2, d, p=p, font=self._font)
            draw_escape_colored_text(sb2, d, p=p, font=self._font)
            if h * image_pil.width > image_pil.height * 4 * w:
                w = w * image_pil.height * 4 // h
                h = image_pil.height * 4
            else:
                h = h * image_pil.width // w
                w = image_pil.width
            g = g.resize((w, h))
            canvas = Image.new(image_pil.mode, (image_pil.width, image_pil.height + h), (255, 255, 255))
            canvas.paste(image_pil)
            canvas.paste(g, (0, image_pil.height))
            # canvas.save(self.outimage_dir / f'{self.nline:09d}-{reason}.png')
            canvas.save(self.outimage_dir / f'{self.nline:09d}-{sa1.replace("/", "")}.png')

    def _fxml(self, _1, _2, _3, pred_str, conf, data):
        d = dict()
        for attr in ['tag', 'DIRECTION', 'TYPE', 'X', 'Y', 'WIDTH', 'HEIGHT', 'CONF']:
            if attr in data:
                d[attr] = f"{data[attr]}"
        if conf is not None:
            d['STR_CONF'] = f"{conf:.3f}"
        if pred_str is not None:
            d['STRING'] = pred_str
        pid = data['path'].parents[1].name
        imagename = data['path'].name

        if pid not in self._xml_data:
            self._xml_data[pid] = {}
        if imagename not in self._xml_data[pid]:
            self._xml_data[pid][imagename] = []

        self._xml_data[pid][imagename].append(d)

    def _fgenerate_xml(self):
        for pid, pages in self._xml_data.items():
            xml_data = ET.Element('OCRDATASET')
            ET.register_namespace('', 'NDLOCRDATASET')
            for p, lines in pages.items():
                page = ET.SubElement(xml_data, 'PAGE', attrib={'IMAGENAME': p})
                for line in lines:
                    line = ET.SubElement(page, line.pop('tag', 'LINE'), attrib=line)

            xml_str = minidom.parseString(ET.tostring(xml_data, encoding='utf-8', method='xml')).toprettyxml(indent=' ')
            out_xml_path = self.outxml_dir / (pid + '.xml')

            with out_xml_path.open(mode='w') as f:
                f.write(xml_str)

    def __call__(self, generator):
        for image, gt, pred, conf, data in generator:
            correct = gt == pred
            for t in self._task:
                t(correct, image, gt, pred, conf, data)
            self.nline += 1
        return self


def draw_escape_colored_text(t, d, p, font):
    get_textsize = functools.partial(d.textsize, font=font)
    it = iter(t)
    cl = (0, 0, 0)
    for c in it:
        if c == '\033':
            n = next(it)
            while n[-1] != 'm':
                n += next(it)
            if n == '[0m':
                cl = (0, 0, 0)
            elif n == '[31m':
                cl = (255, 0, 0)
            elif n == '[33m':
                cl = (255, 255, 0)
            continue
        else:
            size = get_textsize(c)

        d.text(p, c, font=font, fill=cl)
        p[0] += size[0]
    p[0], p[1] = 0, get_textsize(t)[1]


class InferencerWithCLI:
    def __init__(self, conf_dict, character):
        class EmptyOption():
            def __init__(self):
                return

        # create option dictionary from parser
        parser = Inferencer.get_argparser()
        option_key_dict = {}
        for action in parser._actions:
            for opt_str in action.option_strings:
                key_str = None
                if opt_str.startswith('--'):
                    key_str = opt_str[2:]
                    option_key_dict[key_str] = parser.get_default(key_str)

        # create option instance
        opt = EmptyOption()
        for k, v in option_key_dict.items():
            setattr(opt, k, v)
        opt.saved_model = conf_dict['saved_model']
        opt.batch_max_length = conf_dict['batch_max_length']
        opt.batch_size = conf_dict['batch_size']
        opt.character = character
        opt.imgW = conf_dict['imgW']
        opt.workers = conf_dict['workers']
        opt.xml = conf_dict['xml']
        opt.FeatureExtraction = conf_dict['FeatureExtraction']
        opt.Prediction = conf_dict['Prediction']
        opt.PAD = conf_dict['PAD']
        opt.SequenceModeling = conf_dict['SequenceModeling']
        opt.Transformation = conf_dict['Transformation']

        self.opt = opt
        self.inf = Inferencer(self.opt)

        return

    def inference_wich_cli(self, img_data, xml_data, accept_empty=False,
                           yield_block_pillar=True, yield_block_page_num=True):

        cudnn.benchmark = True
        cudnn.deterministic = True
        num_gpu = torch.cuda.device_count()
        dataset = XMLRawDatasetWithCli(img_data, xml_data,
                                       accept_empty=accept_empty,
                                       yield_block_pillar=yield_block_pillar,
                                       yield_block_page_num=yield_block_page_num)
        generator = self.inf.gen(dataset, keep_remain=self.opt.xml)

        result_list = []
        for image, gt, pred, conf, data in generator:
            result_list.append(pred)

        for xml_line in xml_data.getroot().find('PAGE'):
            if len(result_list) == 0:
                print('ERROR: mismatch num of predicted result and xml line')
                break
            if result_list[0] is None:
                print('No predicted STRING for this xml_line')
                print(xml_line.attrib)
                del result_list[0]
                continue
            xml_line.set('STRING', result_list.pop(0))

        return xml_data


if __name__ == '__main__':
    parser = Inferencer.get_argparser()
    g = parser.add_argument_group('db settings')
    g.add_argument('--db_path', required=True, nargs='+', help='データベースへのパス(複数指定可). synthの場合はfont pathを指定する')
    g.add_argument('--db_type', choices=['xmlraw', 'xmllmdb', 'synth'], help='データベースの種類', default='xmlraw')
    g.add_argument('--line_index', type=int, nargs='+', default=None, help='指定の行のみに対して推論. xmllmdb使用時のみ有効')
    action = parser.add_argument_group()
    action.add_argument('--diff', nargs='?', default='none', const='wrong', choices=['none', 'wrong', 'all'],
                        help='差分表示. 画像出力したい場合にはoutimage_dirとfont_pathを指定する')
    action.add_argument('--render', action='store_true', help='diffのgtなし番. outimage_dirとfont_pathが必要')
    action.add_argument('--leven', action='store_true', help='normalized edit distance')
    action.add_argument('--acc', action='store_true', help='accuracy')
    action.add_argument('--xml', default=None, help='xml出力を行う先を指定する')
    parser.add_argument('--stat', action='store_true', help='diff指定時の出力を詳細にする')
    parser.add_argument('--outimage_dir', default=None, help='diff指定時の画像出力先')
    parser.add_argument('--font_path', default=None, help='diff指定時画像出力する際に使用するttf font')
    parser.add_argument('--skip_empty', dest='accept_empty', action='store_false', help='GTが空行の推論を行わない')
    opt = parser.parse_args()

    assert opt.diff != 'none' or opt.render or opt.leven or opt.acc or opt.xml

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    dataset = gen_dataset(opt.db_type, opt.db_path, opt, line_index=opt.line_index,
                          accept_empty=opt.accept_empty, keep_remain=opt.xml)
    generator = Inferencer(opt).gen(dataset, keep_remain=opt.xml, with_tqdm=True)
    char_diff = {
        'none': TR_WORKER.CHAR_DIFF_NONE,
        'wrong': TR_WORKER.CHAR_DIFF_WRONG,
        'all': TR_WORKER.CHAR_DIFF_ALL,
    }[opt.diff]
    w = TR_WORKER(char_diff=char_diff, render=opt.render, stat=opt.stat,
                  accuracy=opt.acc, levenshtein_distance=opt.leven,
                  xml=opt.xml,
                  outimage_dir=opt.outimage_dir,
                  font_path=opt.font_path)(generator).finalize()
    if w._accuracy:
        print(f'Accuracy: {w.accuracy:.4f}')
    if w._levenshtein_distance:
        print(f'Normalized Edit Distance: {w.normalized_edit_distance:.4f}')
