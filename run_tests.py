#!/usr/bin/env python3
"""
Test runner script for grpo_code package.

This script provides a simple way to run the test suite with different options.
"""

import sys
import subprocess
import argparse


def run_tests(test_type="all", verbose=True, coverage=False):
    """Run the test suite with specified options."""
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    # Test type selection
    if test_type == "unit":
        cmd.extend(["-m", "not integration"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif test_type == "rewards":
        cmd.append("tests/test_rewards.py")
    elif test_type == "transforms":
        cmd.append("tests/test_transforms.py")
    # "all" runs everything by default
    
    # Coverage reporting
    if coverage:
        cmd.extend(["--cov=grpo_code", "--cov-report=term-missing"])
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=".")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run grpo_code tests")
    parser.add_argument(
        "--type", 
        choices=["all", "unit", "integration", "rewards", "transforms"],
        default="all",
        help="Type of tests to run (default: all)"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true",
        help="Include coverage reporting"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Run tests in quiet mode"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GRPO Code Test Runner")
    print("=" * 60)
    print(f"Test type: {args.type}")
    print(f"Coverage: {args.coverage}")
    print(f"Verbose: {not args.quiet}")
    print("=" * 60)
    
    exit_code = run_tests(
        test_type=args.type,
        verbose=not args.quiet,
        coverage=args.coverage
    )
    
    print("=" * 60)
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    print("=" * 60)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
