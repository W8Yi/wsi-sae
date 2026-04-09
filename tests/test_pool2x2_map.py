import unittest

import numpy as np

from wsi_sae.data.dataloader import _build_pool2x2_groups_from_coords, Pool2x2GeometryError


class Pool2x2MapTests(unittest.TestCase):
    def test_regular_2x2_grid_builds_one_group(self):
        coords = np.array(
            [
                [0, 0],
                [256, 0],
                [0, 256],
                [256, 256],
            ],
            dtype=np.int64,
        )
        groups, info = _build_pool2x2_groups_from_coords(coords, require_complete=True)
        self.assertEqual(groups.shape, (1, 4))
        self.assertTrue(np.array_equal(groups[0], np.array([0, 1, 2, 3], dtype=np.int64)))
        self.assertEqual(info["step_x"], 256)
        self.assertEqual(info["step_y"], 256)

    def test_missing_corner_drops_or_pads_group(self):
        coords = np.array(
            [
                [0, 0],
                [256, 0],
                [0, 256],
            ],
            dtype=np.int64,
        )
        groups_complete, _ = _build_pool2x2_groups_from_coords(coords, require_complete=True)
        self.assertEqual(groups_complete.shape, (0, 4))

        groups_partial, _ = _build_pool2x2_groups_from_coords(coords, require_complete=False)
        self.assertEqual(groups_partial.shape, (1, 4))
        self.assertTrue(np.array_equal(groups_partial[0], np.array([0, 1, 2, 2], dtype=np.int64)))

    def test_duplicate_coord_raises(self):
        coords = np.array(
            [
                [0, 0],
                [256, 0],
                [0, 256],
                [256, 256],
                [0, 0],  # duplicate quantized grid position
            ],
            dtype=np.int64,
        )
        with self.assertRaises(Pool2x2GeometryError):
            _build_pool2x2_groups_from_coords(coords, require_complete=True)


if __name__ == "__main__":
    unittest.main()
