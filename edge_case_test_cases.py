# Edge Case Test Cases for Two Sum Problem

# Test Case 1: Minimum length array
# nums = [5, 5], target = 10
# Expected Output: [0, 1]

# Test Case 2: Negative numbers
# nums = [-1, -2, -3, -4, -5], target = -8
# Expected Output: [2, 4] (because -3 + -5 = -8)

# Test Case 3: Mixed positive and negative numbers
# nums = [-10, 7, 19, 15, -3], target = 12
# Expected Output: [3, 4] (because 15 + -3 = 12)

# Test Case 4: Large numbers
# nums = [10**9, 1, 2, 3, 10**9 - 5], target = 2 * 10**9 - 5
# Expected Output: [0, 4] (because 10**9 + (10**9 - 5) = 2 * 10**9 - 5)

# Test Case 5: Zero target
# nums = [-5, 0, 5, 10], target = 0
# Expected Output: [0, 2] (because -5 + 5 = 0)

# Test Case 6: Target with one negative and one positive number
# nums = [-100, 50, 200, -150], target = -50
# Expected Output: [0, 1] (because -100 + 50 = -50)