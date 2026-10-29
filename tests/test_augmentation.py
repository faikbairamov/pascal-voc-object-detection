"""Check geometric correctness of training-only horizontal flips."""
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from voc_detection.dataset import VOCDetectionDataset
from voc_detection.project import load_config, make_loader


class HorizontalFlipTests(unittest.TestCase):
    def test_box_and_image_are_flipped_together(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'JPEGImages').mkdir()
            (root / 'Annotations').mkdir()
            (root / 'split.txt').write_text('example\n')
            image = Image.new('RGB', (10, 6), 'black')
            for y in range(6):
                for x in range(3): image.putpixel((x,y),(255,0,0))
            image.save(root / 'JPEGImages/example.jpg')
            (root / 'Annotations/example.xml').write_text('''<annotation><object>
            <name>person</name><difficult>0</difficult><bndbox>
            <xmin>2</xmin><ymin>2</ymin><xmax>5</xmax><ymax>4</ymax>
            </bndbox></object></annotation>''')
            original, target = VOCDetectionDataset(root, root / 'split.txt')[0]
            flipped, flipped_target = VOCDetectionDataset(root, root / 'split.txt',1.)[0]
            self.assertTrue(torch.equal(flipped, original.flip(-1)))
            self.assertTrue(torch.equal(target['boxes'],torch.tensor([[1.,1.,5.,4.]])))
            self.assertTrue(torch.equal(flipped_target['boxes'],torch.tensor([[5.,1.,9.,4.]])))
            self.assertTrue(torch.equal(target['labels'],flipped_target['labels']))
            self.assertTrue(torch.equal(target['difficult'],flipped_target['difficult']))

    def test_augmentation_never_applies_to_dev(self):
        config = {**load_config(), 'horizontal_flip_prob': 1.}
        train, _ = make_loader(config, 'train')
        dev, _ = make_loader(config, 'dev')
        self.assertEqual(train.horizontal_flip_prob, 1.)
        self.assertEqual(dev.horizontal_flip_prob, 0.)


if __name__ == '__main__': unittest.main()
