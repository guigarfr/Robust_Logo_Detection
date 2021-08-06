import os.path as osp

import mmcv

from . import XMLDataset
from .builder import DATASETS

import xml.etree.ElementTree as ET

from PIL import Image


@DATASETS.register_module()
class LogosDataset(XMLDataset):

    def load_annotations(self, ann_file):
        """Load annotation from XML style ann_file.

        Args:
            ann_file (str): Path of XML file.

        Returns:
            list[dict]: Annotation info from XML file.
        """

        if not self.CLASSES:
            self.CLASSES = set()

        data_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        for img_id in img_ids:
            filename = f'JPEGImages/{img_id}.jpg'
            xml_path = osp.join(self.img_prefix, 'Annotations',
                                f'{img_id}.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Get image size data
            size = root.find('size')
            if size is not None:
                width = int(size.find('width').text)
                height = int(size.find('height').text)
            else:
                img_path = osp.join(self.img_prefix, 'JPEGImages',
                                    '{}.jpg'.format(img_id))
                img = Image.open(img_path)
                width, height = img.size

            # Get object classes
            self.CLASSES |= {x.text for x in tree.findall("object/name")}

            data_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height))

        self.CLASSES = sorted(list(self.CLASSES))

        return data_infos

