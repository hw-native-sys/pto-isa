# Security Notice

## Recommended User Account

For security reasons, avoid running commands as `root` (or other administrative accounts). Follow the principle of least privilege whenever possible.

## File Permission Hardening

- On the host (including the container host) and inside containers, set `umask` to `0027` or stricter. This makes newly created directories default to at most `750`, and newly created files default to at most `640`.
- Apply appropriate access control to sensitive assets such as personal data, proprietary data, source code, and intermediate artifacts generated during PTO instruction development. See “Appendix A” for recommended maximum permissions.
- During installation and usage, make sure your installation directory and any input data files have appropriate permissions configured.

## Build Security

When building this project from source, you compile it locally and intermediate build artifacts are generated. After the build completes, restrict permissions on those artifacts to protect sensitive data.

## Runtime Security

- If a PTO instruction implementation encounters a runtime error, it may terminate the process and print an error message. Use the error output to locate the root cause (for example, ensure required synchronization is present and inspect logs when available).

## Public Network Addresses

This repository contains references to the following public URLs:

| Type       | Open-source URL | File                              | Public URL                                                                                                                                    | Purpose                                      |
| :--------: | :-------------: | --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| Dependency | N/A             | cmake/fetch_cann_cmake.cmake      | https://gitcode.com/cann/cmake.git                                                                                                            | Pulls the shared CANN engineering cmake repo |
| Dependency | N/A             | cmake/third_party/json.cmake      | https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip                                                           | Downloads `json` headers as a build dep      |
| Dependency | N/A             | cmake/third_party/gtest.cmake     | https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz                                        | Downloads GoogleTest as a build/test dep     |

## Appendix

### Appendix A: Recommended Maximum Permissions

| Scenario                                          | Recommended maximum Linux permission |
| ------------------------------------------------- | ----------------------------------- |
| User home directory                                | 750 (`rwxr-x---`)                   |
| Program files (scripts, libraries, etc.)           | 550 (`r-xr-x---`)                   |
| Program directories                                | 550 (`r-xr-x---`)                   |
| Configuration files                                | 640 (`rw-r-----`)                   |
| Configuration directories                          | 750 (`rwxr-x---`)                   |
| Log files (archived / complete)                    | 440 (`r--r-----`)                   |
| Log files (actively written)                       | 640 (`rw-r-----`)                   |
| Log directories                                    | 750 (`rwxr-x---`)                   |
| Debug files                                        | 640 (`rw-r-----`)                   |
| Debug directories                                  | 750 (`rwxr-x---`)                   |
| Temporary directories                              | 750 (`rwxr-x---`)                   |
| Maintenance / upgrade directories                  | 770 (`rwxrwx---`)                   |
| Business data files                                | 640 (`rw-r-----`)                   |
| Business data directories                          | 750 (`rwxr-x---`)                   |
| Key material / private keys / certs / ciphertext dirs | 700 (`rwx------`)                   |
| Key material / private keys / certs / ciphertext files | 600 (`rw-------`)                   |
| Crypto interfaces and scripts                      | 500 (`r-x------`)                   |
