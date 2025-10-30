import unittest
from two_sum_solution import Solution

class TestTwoSum(unittest.TestCase):

    def setUp(self):
        self.solution = Solution()

    def test_example_1(self):
        nums = [2, 7, 11, 15]
        target = 9
        self.assertEqual(self.solution.twoSum(nums, target), [0, 1])

    def test_example_2(self):
        nums = [3, 2, 4]
        target = 6
        self.assertEqual(self.solution.twoSum(nums, target), [1, 2])

    def test_example_3(self):
        nums = [3, 3]
        target = 6
        self.assertEqual(self.solution.twoSum(nums, target), [0, 1])

    # Edge Case Test Cases
    def test_min_length_array(self):
        nums = [5, 5]
        target = 10
        self.assertEqual(self.solution.twoSum(nums, target), [0, 1])

    def test_negative_numbers(self):
        nums = [-1, -2, -3, -4, -5]
        target = -8
        self.assertEqual(self.solution.twoSum(nums, target), [2, 4]) # -3 + -5 = -8

    def test_mixed_numbers(self):
        nums = [-10, 7, 19, 15, -3]
        target = 12
        self.assertEqual(self.solution.twoSum(nums, target), [3, 4]) # 15 (index 3) + -3 (index 4) = 12

    def test_large_numbers(self):
        nums = [10**9, 1, 2, 3, 10**9 - 5]
        target = 2 * 10**9 - 5
        self.assertEqual(self.solution.twoSum(nums, target), [0, 4])

    def test_zero_target(self):
        nums = [-5, 0, 5, 10]
        target = 0
        self.assertEqual(self.solution.twoSum(nums, target), [0, 2]) # -5 + 5 = 0

    def test_target_with_one_negative_one_positive(self):
        nums = [-100, 50, 200, -150]
        target = -50
        self.assertEqual(self.solution.twoSum(nums, target), [0, 1]) # -100 + 50 = -50

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)