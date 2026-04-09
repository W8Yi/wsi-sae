import unittest
from collections import Counter

import numpy as np

from wsi_sae.commands.mine import select_latents


class ParentBalancedSelectionTests(unittest.TestCase):
    def test_parent_balanced_selection_respects_cap_and_keeps_variable_counts(self):
        # 48 latents, 6 parents, 8 children each, with stronger early parents.
        parent = np.repeat(np.arange(6, dtype=np.int64), 8)
        scores = np.concatenate(
            [
                np.linspace(1.00, 0.86, num=8, dtype=np.float32),
                np.linspace(0.92, 0.72, num=8, dtype=np.float32),
                np.linspace(0.78, 0.58, num=8, dtype=np.float32),
                np.linspace(0.55, 0.35, num=8, dtype=np.float32),
                np.linspace(0.32, 0.20, num=8, dtype=np.float32),
                np.linspace(0.18, 0.05, num=8, dtype=np.float32),
            ]
        )
        global_stats = {
            "max": scores.copy(),
            "var": scores.copy(),
            "sparsity": scores.copy(),
        }

        selected, summary = select_latents(
            global_stats=global_stats,
            strategy="sdf_parent_balanced",
            n_latents=12,
            manual=None,
            parent_assignment_all_level1=parent,
            parent_max_children_per_selected_parent=6,
            parent_preferred_children_per_selected_parent=4,
            parent_target_count=3,
        )

        self.assertEqual(len(selected), 12)
        self.assertIsNotNone(summary)

        counts = Counter(int(parent[i]) for i in selected)
        self.assertEqual(len(counts), 3)
        self.assertTrue(all(1 <= c <= 6 for c in counts.values()))
        self.assertGreaterEqual(int(np.median(list(counts.values()))), 3)
        self.assertGreater(len(set(counts.values())), 1)

    def test_selection_is_deterministic_for_same_inputs(self):
        parent = np.repeat(np.arange(5, dtype=np.int64), 6)
        scores = np.linspace(0.0, 1.0, num=30, dtype=np.float32)
        global_stats = {
            "max": scores.copy(),
            "var": scores.copy(),
            "sparsity": scores.copy(),
        }

        a, sa = select_latents(
            global_stats=global_stats,
            strategy="sdf_parent_balanced",
            n_latents=16,
            manual=None,
            parent_assignment_all_level1=parent,
            parent_max_children_per_selected_parent=6,
            parent_preferred_children_per_selected_parent=4,
            parent_target_count=-1,
        )
        b, sb = select_latents(
            global_stats=global_stats,
            strategy="sdf_parent_balanced",
            n_latents=16,
            manual=None,
            parent_assignment_all_level1=parent,
            parent_max_children_per_selected_parent=6,
            parent_preferred_children_per_selected_parent=4,
            parent_target_count=-1,
        )

        self.assertEqual(a, b)
        self.assertEqual(sa, sb)


if __name__ == "__main__":
    unittest.main()
