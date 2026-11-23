# Contributing Guide

We welcome developers to explore PTO Tile Lib and contribute to the project. Before participating in community contributions, please refer to [cann-community](https://gitcode.com/cann/community) to review the community code of conduct, complete the CLA process, and understand the contribution workflow used by the source repositories.

This document is organized into three parts:

1. **Common Contribution Scenarios**: explains which types of changes are suitable for contribution and how to get started with an Issue.
2. **Local Development**: covers local implementation, delivery files, and local checks.
3. **Submitting and Merging PRs**: describes pre-submission checks, PR submission, and the review and merge process.

## Common Contribution Scenarios

### Bug Fixes for Operators

If you find a bug in an operator implementation in this project and would like to fix it, you are welcome to open an Issue for tracking and resolution.

You can follow [Submit / Process Issues](https://gitcode.com/cann/community#提交Issue处理Issue任务) to create a `Bug-Report|缺陷反馈` Issue describing the problem, then enter `/assign` or `/assign @yourself` in a comment to assign the Issue to yourself.

### Operator Optimization

If you have ideas for improving generality or optimizing the performance of an existing operator implementation, you are welcome to contribute them.

You can follow [Submit / Process Issues](https://gitcode.com/cann/community#提交Issue处理Issue任务) to create a `Requirement|需求建议` Issue describing the optimization opportunity and your proposed design. Then enter `/assign` or `/assign @yourself` in a comment to assign the Issue to yourself for follow-up work.

### Contributing a New Operator

If you want to design and implement a brand-new operator for NPU, you are welcome to propose the idea and design in an Issue.

#### 1. Create an Issue

Please follow [Submit / Process Issues](https://gitcode.com/cann/community#提交Issue处理Issue任务) to create a `Requirement|需求建议` Issue and describe the proposed operator design.

The Issue should include:

- **Background**
- **Value / motivation**
- **Design proposal**

You should also comment `/assign` or `/assign @yourself` on the Issue to claim the task for later implementation.

#### 2. Requirement Review

SIG members will review the Issue and provide feedback. After addressing the comments, please reply in the Issue:

> "The review comments have been addressed. Requesting re-review."

If the requirement is accepted, SIG members will assign a suitable operator category path for you (for example `include/pto/npu/a5`) so that the new operator can be contributed in the appropriate location.

If no consensus is reached during the Issue discussion, it is recommended to request further discussion in the SIG biweekly meeting.

### Documentation Fixes

If you find incorrect descriptions in the operator documentation, you are welcome to open an Issue and submit a fix.

You can follow [Submit / Process Issues](https://gitcode.com/cann/community#提交Issue处理Issue任务) to create a `Documentation|文档反馈` Issue describing the problem, then enter `/assign` or `/assign @yourself` in a comment to assign the Issue to yourself.

### Helping Resolve Other Issues

If you have a suitable solution to an Issue raised by another community member, you are welcome to discuss it in the Issue and help resolve it together.

If the Issue requires code changes, you can enter `/assign` or `/assign @yourself` in the Issue comments to assign the task to yourself and help drive it to completion.

## Local Development

Once the design is confirmed, you can begin local development. For a new operator, the delivery contents can be adjusted according to the minimum required structure. The following layout reflects the current repository organization and can be used as a reference, where `${op_name}` is the operator name and `${op_class}` is the operator category path assigned during review:

```text
    docs/
    ├── isa/
    │   └── ${op_name}.md                                # operator documentation
    include/
    └── pto/
        ├── common/
        │   ├── pto_instr_impl.hpp                       # summary of operator implementations
        │   └── pto_instr.hpp                            # public interfaces
        └── ${op_class}/
            └── ${op_name}.hpp                           # operator implementation, comments, structures, logic
    tests/
    ├── ${op_class}/src/st/testcase/
    │   ├── ${op_name}/
    │   │   ├── ${op_name}.cpp                           # interface invocation file
    │   │   ├── main.cpp                                 # test entry file
    │   │   ├── gen_data.py                              # input / expected output generator
    │   │   └── CMakeLists.txt                           # build file
    │   └── CMakeLists.txt                               # testcase collection build file
    ├── run_st.sh                                        # ST execution script entry
    └── README.md                                        # test instructions
```

Before preparing a PR, please complete the following local development checks:

- Ensure the delivery files are complete, including ST testcases
- Make sure the code follows `.clang-format` and `pyproject.toml`; use `clang-format -i -style=file <file>` and `ruff format <file>` before submission

## Submitting and Merging PRs

When preparing local changes and submitting a PR, please pay particular attention to the following:

1. When submitting a PR, please complete the PR template carefully, including the business background, goals, and design of the change.
2. If your change is not a simple bug fix, but involves a new feature, a new interface, a new configuration parameter, or a modification to an existing workflow, please open an Issue for design discussion first to avoid unnecessary rework or rejection during review. If you are unsure whether your change qualifies as a “simple bug fix,” it is still recommended to start with an Issue.

### Local Checks Before Submission

Community contributors can use `pre-commit` to perform local checks before submitting changes.

#### Step 1: Install the `pre-commit` framework

```bash
# Install with pip (recommended)
pip install pre-commit

# Verify installation
pre-commit --version
# Output: pre-commit 3.x.x
```

For Windows users, make sure Python and pip are installed first.

#### Step 2: Enter the project directory

```bash
cd /path/to/your/pto-isa

# Example
cd d:\complianceRepo\CANN\pto-isa
```

#### Step 3: Install Git hooks

```bash
# Run in the project root directory
pre-commit install
```

#### Step 4: Verify the installation (optional)

This command is used to verify that the `pre-commit` hook is installed and can run successfully. If the checks pass, it creates an empty commit, so you can remove it afterward if it is only for testing. After `pre-commit` is installed, the same checks will run automatically on subsequent `git commit` operations.

```bash
# Test the hook with an empty commit
git commit --allow-empty -m "test pre-commit"
```

#### PR Submission and Review

Before submitting the PR, please also confirm the following:

- The PR is linked to the corresponding Issue
- The CLA has been signed
- Use the `compile` command in comments to trigger CI checks, and fix any issues reported by CI. If `codecheck` reports false positives, contact SIG members for suppression

After the checks pass, reply in the linked Issue:

> "The PR linked to this Issue is: XXX. Please review when available."

After SIG members provide review comments, please complete all requested changes and reply:

> "The PR linked to this Issue is: XXX. All requested changes have been addressed. Please review again."

After the committer review passes, the maintainer will perform the final review. Once approved, the PR will be merged with `/lgtm` and `/approve` labels.
