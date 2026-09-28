import importlib.util
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile, ZIP_DEFLATED

from PIL import Image

spec = importlib.util.spec_from_file_location('fetch_swig_images',
    Path(__file__).resolve().parents[1] / 'scripts/fetch_swig_images.py')
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


class DownloadTests(unittest.TestCase):
    def test_selection_integrity_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = BytesIO()
            Image.new('RGB', (16, 16), 'blue').save(payload, format='JPEG')
            dataset = root / 'data.json'
            dataset.write_text(json.dumps([{'images': ['/old/path/example.jpg']}]))
            with ZipFile(root / 'images.zip', 'w', compression=ZIP_DEFLATED) as archive:
                archive.writestr('images_512/example.jpg', payload.getvalue())
                archive.writestr('images_512/unneeded.jpg', payload.getvalue())
            with ZipFile(root / 'images.zip') as archive:
                members = download.select_members(archive, download.required_images([dataset]))
                self.assertEqual(len(members), 1)
                raw_entry = (root / 'images.zip').read_bytes()[members[0].header_offset:]
                self.assertEqual(download.decode_member(raw_entry, members[0]), payload.getvalue())
                with self.assertRaisesRegex(ValueError, 'Truncated'):
                    download.decode_member(raw_entry[:80], members[0])
                first = download.fetch_member(archive, members[0], root)
                self.assertFalse(first['cached'])
                self.assertTrue(download.fetch_member(archive, members[0], root)['cached'])
                (root / 'example.jpg').write_bytes(b'corrupt')
                repaired = download.fetch_member(archive, members[0], root)
                self.assertFalse(repaired['cached'])
                self.assertEqual(first['sha256'], repaired['sha256'])
                with self.assertRaisesRegex(ValueError, 'absent'):
                    download.select_members(archive, {'missing.jpg'})
            with ZipFile(root / 'ambiguous.zip', 'w') as archive:
                archive.writestr('a/example.jpg', payload.getvalue())
                archive.writestr('b/example.jpg', payload.getvalue())
            with ZipFile(root / 'ambiguous.zip') as archive:
                with self.assertRaisesRegex(ValueError, 'Ambiguous'):
                    download.select_members(archive, {'example.jpg'})


if __name__ == '__main__':
    unittest.main()
