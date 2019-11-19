import unittest
from match_konspect import MatchKonspekt


class TestMatchKonspect(unittest.TestCase):
    def test_match(self):
        mk = MatchKonspekt()
        pattern = "14"
        category, cat_pattern, subcategory = mk.find_category(pattern)
        self.assertEqual(category, 5)

    def test_unpack_dot(self):
        mk = MatchKonspekt()
        mdt = "639.2/.6"
        returned = mk.unpack_mdt(mdt)
        unpacked = ["639.2", "639.3", "639.4", "639.5", "639.6"]
        self.assertEqual(unpacked, returned)

    def test_unpack_dot_in(self):
        mk = MatchKonspekt()
        mdt = "821.81/.82.09"
        returned = mk.unpack_mdt(mdt)
        unpacked = ["821.81.09", "821.82.09"]
        self.assertEqual(unpacked, returned)

    # def test_unpack_dash(self):
    #     mk = MatchKonspekt()
    #     mdt = "36-1/-7"
    #     returned = mk.unpack_mdt(mdt)
    #     unpacked = ["36-1", "36-2", "36-3", "36-4", "36-5", "36-6", "36-7"]
    #     self.assertEqual(unpacked, returned)

    def test_unpack_parenthesis(self):
        mk = MatchKonspekt()
        mdt = "94(420/429)"
        returned = mk.unpack_mdt(mdt)
        unpacked = ["94(420)", "94(421)", "94(422)", "94(423)", "94(424)", "94(425)", "94(426)", "94(427)", "94(428)",
                    "94(429)"]
        self.assertEqual(unpacked, returned)

    def test_unpack_solo(self):
        mk = MatchKonspekt()
        mdt = "22/24"
        returned = mk.unpack_mdt(mdt)
        unpacked = ["22", "23", "24"]
        self.assertEqual(unpacked, returned)

    def test_unpack_zero(self):
        mk = MatchKonspekt()
        mdt = "022/024"
        returned = mk.unpack_mdt(mdt)
        unpacked = ["022", "023", "024"]
        self.assertEqual(unpacked, returned)

    def test_shorten_number(self):
        mk = MatchKonspekt()
        mdt = "123"
        returned = mk.shorten_from_right(mdt)
        expected = "12"
        self.assertEqual(expected, returned)

    def test_shorten_dot(self):
        mk = MatchKonspekt()
        mdt = "123.3"
        returned = mk.shorten_from_right(mdt)
        expected = "123"
        self.assertEqual(expected, returned)

    def test_shorten_dash(self):
        mk = MatchKonspekt()
        mdt = "123-3"
        returned = mk.shorten_from_right(mdt)
        expected = "123"
        self.assertEqual(expected, returned)

    def test_shorten_parenthesis(self):
        mk = MatchKonspekt()
        mdt = "94(420)"
        returned = mk.shorten_from_right(mdt)
        expected = "94(42)"
        self.assertEqual(expected, returned)

    def test_shorten_quotes(self):
        mk = MatchKonspekt()
        mdt = "14(100-15)\"15/20\""
        returned = mk.shorten_from_right(mdt)
        expected = "14(100-15)"
        self.assertEqual(expected, returned)

if __name__ == '__main__':
    unittest.main()
