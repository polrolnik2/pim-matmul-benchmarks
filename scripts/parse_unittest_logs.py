#!/usr/bin/env python3
"""
Unittest log parser that checks compilation status and test results.
First checks if compilation succeeded, then parses test outcomes.
"""

import os
import re
import sys

# Directory containing logs
LOG_DIR = os.path.expanduser('~/workspace/pim-matmul-benchmarks/scratch')
COMPILE_LOG_DIR = os.path.join(LOG_DIR, 'compile_logs')
RUNTIME_LOG_DIR = os.path.join(LOG_DIR, 'runtime_logs')

# Compilation patterns
ERROR_PATTERN = re.compile(r'error:', re.IGNORECASE)
WARNING_PATTERN = re.compile(r'warning:', re.IGNORECASE)
LINKER_ERROR_PATTERN = re.compile(r'undefined reference|ld: symbol|cannot find', re.IGNORECASE)

# Test result patterns
PASS_PATTERN = re.compile(r'\[PASS\]')
FAIL_PATTERN = re.compile(r'\[FAIL\]')
TEST_FAILED_PATTERN = re.compile(r'(\d+) .* tests? failed')
ALL_PASSED_PATTERN = re.compile(r'All .* tests passed')

def check_compilation_status(compile_log_path):
    """Check if compilation succeeded by looking for errors."""
    if not os.path.exists(compile_log_path):
        return "NO_LOG", []
    
    with open(compile_log_path, 'r', errors='ignore') as f:
        content = f.read()
    
    errors = ERROR_PATTERN.findall(content)
    warnings = WARNING_PATTERN.findall(content)
    linker_errors = LINKER_ERROR_PATTERN.findall(content)
    
    # Extract actual error lines for details
    error_lines = []
    for line in content.split('\n'):
        if ERROR_PATTERN.search(line) or LINKER_ERROR_PATTERN.search(line):
            error_lines.append(line.strip())
    
    if errors or linker_errors:
        return "FAILED", error_lines
    elif warnings:
        return "SUCCESS_WITH_WARNINGS", error_lines
    else:
        return "SUCCESS", error_lines

def parse_test_results(runtime_log_path):
    """Parse test results from runtime log."""
    if not os.path.exists(runtime_log_path):
        return "NO_LOG", 0, 0, []
    
    with open(runtime_log_path, 'r', errors='ignore') as f:
        content = f.read()
    
    # Check if log is empty
    if not content.strip():
        return "EMPTY_LOG", 0, 0, ["Runtime log is empty - test may have crashed or not run"]
    
    passes = len(PASS_PATTERN.findall(content))
    fails = len(FAIL_PATTERN.findall(content))
    
    # Extract failure details
    failure_lines = []
    for line in content.split('\n'):
        if FAIL_PATTERN.search(line):
            failure_lines.append(line.strip())
    
    # Check overall test result
    if ALL_PASSED_PATTERN.search(content):
        status = "ALL_PASSED"
    elif TEST_FAILED_PATTERN.search(content):
        status = "SOME_FAILED"
    elif fails > 0:
        status = "SOME_FAILED"
    elif passes > 0:
        status = "ALL_PASSED"
    else:
        status = "NO_TESTS"
    
    return status, passes, fails, failure_lines

def main():
    print("==== UNITTEST COMPILATION AND TEST RESULTS ====\n")
    
    if not os.path.exists(COMPILE_LOG_DIR):
        print(f"Compile log directory not found: {COMPILE_LOG_DIR}")
        return 1
    
    # Get all test names from compile logs
    test_names = []
    for fname in os.listdir(COMPILE_LOG_DIR):
        if fname.endswith('.log'):
            test_name = fname[:-4]  # Remove .log extension
            test_names.append(test_name)
    
    if not test_names:
        print("No test logs found.")
        return 1
    
    overall_success = True
    
    for test_name in sorted(test_names):
        compile_log = os.path.join(COMPILE_LOG_DIR, f"{test_name}.log")
        runtime_log = os.path.join(RUNTIME_LOG_DIR, f"{test_name}.log")
        
        print(f"=== {test_name} ===")
        
        # Check compilation
        compile_status, error_lines = check_compilation_status(compile_log)
        print(f"Compilation: {compile_status}")
        
        if compile_status == "FAILED":
            overall_success = False
            print("Compilation errors:")
            for error in error_lines[:5]:  # Show first 5 errors
                print(f"  {error}")
            if len(error_lines) > 5:
                print(f"  ... and {len(error_lines) - 5} more errors")
            print("Skipping runtime check due to compilation failure.\n")
            continue
        elif compile_status == "SUCCESS_WITH_WARNINGS":
            print("Compilation succeeded with warnings.")
        
        # Check test results
        test_status, passes, fails, failure_lines = parse_test_results(runtime_log)
        print(f"Tests: {test_status} ({passes} PASS, {fails} FAIL)")
        
        if test_status == "SOME_FAILED":
            overall_success = False
            print("Test failures:")
            for failure in failure_lines[:5]:  # Show first 5 failures
                print(f"  {failure}")
            if len(failure_lines) > 5:
                print(f"  ... and {len(failure_lines) - 5} more failures")
        elif test_status == "NO_LOG":
            overall_success = False
            print("No runtime log found - test may not have run.")
        elif test_status == "EMPTY_LOG":
            overall_success = False
            print("Runtime log is empty - test may have crashed or failed to run.")
        elif test_status == "NO_TESTS":
            print("No test results found in runtime log.")
        
        print()
    
    # Overall summary
    print("==== OVERALL SUMMARY ====")
    if overall_success:
        print("All tests compiled and passed successfully!")
        return 0
    else:
        print("Some tests failed to compile or had test failures.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
