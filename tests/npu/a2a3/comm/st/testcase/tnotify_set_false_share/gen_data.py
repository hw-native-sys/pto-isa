#!/usr/bin/python3
# coding=utf-8
# run_st.py copies gen_data.py into the build dir and runs it. Other comm STs
# generate golden tensors here. This ST checks live GM words after Set, so
# the stub only creates the expected empty testcases/ directory.
import os

def main():
    os.makedirs("testcases", exist_ok=True)

if __name__ == "__main__":
    main()
